"""
Compact end-of-run system summary generation.

This module reads aggregated host and GPU system metrics from the
`system_samples` projection table and produces:

1. a compact text summary for end-of-run display and sharing
2. a structured JSON payload for future automation and compare features

Design goals
------------
- Keep the printed summary short and easy to scan
- Preserve richer structured fields in JSON
- Avoid changing the aggregation contract while improving presentation
"""

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

from traceml.aggregator.summaries.summary_formatting import (
    bytes_to_gb,
    duration_from_bounds,
    format_optional,
)
from traceml.aggregator.summaries.summary_io import (
    load_json_or_empty,
    write_json,
)


@dataclass
class SystemSummaryAgg:
    """
    Aggregated system metrics loaded from `system_samples`.

    Notes
    -----
    - Memory values remain in raw bytes while aggregating and are converted
      only at formatting / serialization time.
    - GPU fields are optional because CPU-only runs are fully supported.
    """

    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    system_samples: int = 0

    cpu_avg_percent: Optional[float] = None
    cpu_peak_percent: Optional[float] = None

    ram_avg_bytes: Optional[float] = None
    ram_peak_bytes: Optional[float] = None
    ram_total_bytes: Optional[float] = None

    gpu_available: Optional[bool] = None
    gpu_count: Optional[int] = None

    gpu_util_avg_percent: Optional[float] = None
    gpu_util_peak_percent: Optional[float] = None

    gpu_mem_avg_bytes: Optional[float] = None
    gpu_mem_peak_bytes: Optional[float] = None

    gpu_temp_avg_c: Optional[float] = None
    gpu_temp_peak_c: Optional[float] = None

    gpu_power_avg_w: Optional[float] = None
    gpu_power_peak_w: Optional[float] = None


def _load_system_summary_agg(
    conn: sqlite3.Connection,
    *,
    rank: Optional[int] = None,
    max_system_rows: int = 5_000,
) -> SystemSummaryAgg:
    """
    Load aggregated system metrics directly from `system_samples`.

    Parameters
    ----------
    conn:
        Open SQLite connection.
    rank:
        Optional rank filter. If None, aggregates across all ranks.
    max_system_rows:
        Safety cap on rows included in aggregation.

    Returns
    -------
    SystemSummaryAgg
        Aggregated summary values ready for formatting.
    """
    where_clause = ""
    params: list[Any] = []

    if rank is not None:
        where_clause = "WHERE rank = ?"
        params.append(int(rank))

    count_sql = f"""
        SELECT
            COUNT(*),
            MIN(sample_ts_s),
            MAX(sample_ts_s)
        FROM (
            SELECT sample_ts_s
            FROM system_samples
            {where_clause}
            ORDER BY id ASC
            LIMIT ?
        );
    """
    count_row = conn.execute(
        count_sql, (*params, int(max_system_rows))
    ).fetchone()

    n_rows = int(count_row[0] or 0)
    first_ts = float(count_row[1]) if count_row[1] is not None else None
    last_ts = float(count_row[2]) if count_row[2] is not None else None

    agg_sql = f"""
        SELECT
            AVG(cpu_percent),
            MAX(cpu_percent),

            AVG(ram_used_bytes),
            MAX(ram_used_bytes),
            MAX(ram_total_bytes),

            MAX(gpu_available),
            MAX(gpu_count),

            AVG(gpu_util_avg),
            MAX(gpu_util_peak),

            AVG(gpu_mem_used_avg_bytes),
            MAX(gpu_mem_used_peak_bytes),

            AVG(gpu_temp_avg_c),
            MAX(gpu_temp_peak_c),

            AVG(gpu_power_avg_w),
            MAX(gpu_power_peak_w)
        FROM (
            SELECT *
            FROM system_samples
            {where_clause}
            ORDER BY id ASC
            LIMIT ?
        );
    """
    row = conn.execute(agg_sql, (*params, int(max_system_rows))).fetchone()

    return SystemSummaryAgg(
        first_ts=first_ts,
        last_ts=last_ts,
        system_samples=n_rows,
        cpu_avg_percent=float(row[0]) if row[0] is not None else None,
        cpu_peak_percent=float(row[1]) if row[1] is not None else None,
        ram_avg_bytes=float(row[2]) if row[2] is not None else None,
        ram_peak_bytes=float(row[3]) if row[3] is not None else None,
        ram_total_bytes=float(row[4]) if row[4] is not None else None,
        gpu_available=bool(row[5]) if row[5] is not None else None,
        gpu_count=int(row[6]) if row[6] is not None else None,
        gpu_util_avg_percent=float(row[7]) if row[7] is not None else None,
        gpu_util_peak_percent=float(row[8]) if row[8] is not None else None,
        gpu_mem_avg_bytes=float(row[9]) if row[9] is not None else None,
        gpu_mem_peak_bytes=float(row[10]) if row[10] is not None else None,
        gpu_temp_avg_c=float(row[11]) if row[11] is not None else None,
        gpu_temp_peak_c=float(row[12]) if row[12] is not None else None,
        gpu_power_avg_w=float(row[13]) if row[13] is not None else None,
        gpu_power_peak_w=float(row[14]) if row[14] is not None else None,
    )


def _build_gpu_line(
    *,
    gpu_available: Optional[bool],
    gpu_count: Optional[int],
    gpu_util_avg_percent: Optional[float],
    gpu_util_peak_percent: Optional[float],
    gpu_mem_peak_gb: Optional[float],
    gpu_temp_peak_c: Optional[float],
) -> str:
    """
    Build one compact GPU line for the printed summary.

    The printed end-of-run summary is intentionally compact. We therefore keep:
    - availability / absence
    - utilization
    - peak memory
    - peak temperature

    More detailed GPU fields remain available in JSON.
    """
    if gpu_util_avg_percent is not None:
        parts = [
            f"util avg {format_optional(gpu_util_avg_percent, '%', 1)}",
            f"peak {format_optional(gpu_util_peak_percent, '%', 1)}",
        ]

        if gpu_mem_peak_gb is not None:
            parts.append(
                f"mem peak {format_optional(gpu_mem_peak_gb, ' GB', 1)}"
            )

        if gpu_temp_peak_c is not None:
            parts.append(
                f"temp peak {format_optional(gpu_temp_peak_c, ' C', 1)}"
            )

        return "GPU: " + " | ".join(parts)

    if gpu_available is False:
        return "GPU: unavailable"

    if (gpu_count or 0) > 0:
        return "GPU: detected, but no per-GPU samples were recorded"

    return "GPU: n/a"


def _build_system_card(agg: SystemSummaryAgg) -> tuple[str, Dict[str, Any]]:
    """
    Build a compact, shareable end-of-run system summary.

    Returns
    -------
    tuple[str, Dict[str, Any]]
        - text block for stdout / saved text summaries
        - structured JSON payload

    Notes
    -----
    The JSON payload keeps richer data than the printed text. This preserves
    future flexibility for compare views without making the default summary
    noisy.
    """
    duration_s = duration_from_bounds(agg.first_ts, agg.last_ts)

    ram_avg_gb = bytes_to_gb(agg.ram_avg_bytes)
    ram_peak_gb = bytes_to_gb(agg.ram_peak_bytes)
    ram_total_gb = bytes_to_gb(agg.ram_total_bytes)

    gpu_mem_avg_gb = bytes_to_gb(agg.gpu_mem_avg_bytes)
    gpu_mem_peak_gb = bytes_to_gb(agg.gpu_mem_peak_bytes)

    lines = [
        f"TraceML System Summary | duration {format_optional(duration_s, 's', 1)} | samples {agg.system_samples}",
        "System",
        f"- CPU: avg {format_optional(agg.cpu_avg_percent, '%', 1)}, peak {format_optional(agg.cpu_peak_percent, '%', 1)}",
        (
            f"- RAM: avg {format_optional(ram_avg_gb, ' GB', 1)}, "
            f"peak {format_optional(ram_peak_gb, ' GB', 1)} / {format_optional(ram_total_gb, ' GB', 1)}"
        ),
        (
            "- "
            + _build_gpu_line(
                gpu_available=agg.gpu_available,
                gpu_count=agg.gpu_count,
                gpu_util_avg_percent=agg.gpu_util_avg_percent,
                gpu_util_peak_percent=agg.gpu_util_peak_percent,
                gpu_mem_peak_gb=gpu_mem_peak_gb,
                gpu_temp_peak_c=agg.gpu_temp_peak_c,
            )
        ),
    ]
    card = "\n".join(lines)

    summary = {
        "duration_s": duration_s,
        "system_samples": agg.system_samples,
        "cpu_avg_percent": agg.cpu_avg_percent,
        "cpu_peak_percent": agg.cpu_peak_percent,
        "ram_avg_gb": ram_avg_gb,
        "ram_peak_gb": ram_peak_gb,
        "ram_total_gb": ram_total_gb,
        "gpu_available": agg.gpu_available,
        "gpu_count": agg.gpu_count,
        "gpu_util_avg_percent": agg.gpu_util_avg_percent,
        "gpu_util_peak_percent": agg.gpu_util_peak_percent,
        "gpu_mem_avg_gb": gpu_mem_avg_gb,
        "gpu_mem_peak_gb": gpu_mem_peak_gb,
        "gpu_temp_avg_c": agg.gpu_temp_avg_c,
        "gpu_temp_peak_c": agg.gpu_temp_peak_c,
        "gpu_power_avg_w": agg.gpu_power_avg_w,
        "gpu_power_peak_w": agg.gpu_power_peak_w,
        "units": {
            "memory": "GB",
            "temperature": "C",
            "power": "W",
            "util": "%",
        },
        # Keep `card` for backward compatibility with existing callers/files.
        "card": card,
    }
    return card, summary


def generate_system_summary_card(
    db_path: str,
    *,
    rank: Optional[int] = None,
    print_to_stdout: bool = True,
    max_system_rows: int = 5_000,
) -> Dict[str, Any]:
    """
    Generate a compact SYSTEM summary from SQL projection tables.

    Parameters
    ----------
    db_path:
        Path to the SQLite DB file.
    rank:
        Optional rank filter. If None, summarizes across all ranks.
    print_to_stdout:
        If True, print the rendered compact summary.
    max_system_rows:
        Safety cap on rows included in aggregation.

    Returns
    -------
    Dict[str, Any]
        Structured summary JSON including the rendered `card` text.
    """
    conn = sqlite3.connect(db_path)
    try:
        agg = _load_system_summary_agg(
            conn,
            rank=rank,
            max_system_rows=max_system_rows,
        )
    finally:
        conn.close()

    card, summary = _build_system_card(agg)

    with open(db_path + "_summary_card.txt", "w", encoding="utf-8") as f:
        f.write(card + "\n")

    existing = load_json_or_empty(db_path + "_summary_card.json")
    existing["system"] = summary
    write_json(db_path + "_summary_card.json", existing)

    if print_to_stdout:
        print(card)

    return summary
