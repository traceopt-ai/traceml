"""
Compact end-of-run system summary generation.

This module reads aggregated host and GPU system metrics from the
`system_samples` and `system_gpu_samples` projection tables and produces:

1. a compact text summary for end-of-run display and sharing
2. a structured JSON payload for automation, logging, and future dashboard use

Design goals
------------
- Keep the printed summary short and easy to scan
- Preserve richer machine-readable fields in JSON
- Use one clear canonical schema for system summary data
- Add explicit structured blocks for:
  - CPU rollup
  - RAM rollup
  - GPU rollup across all GPUs
  - per-GPU rollups

Notes
-----
- The printed text intentionally remains compact.
- The JSON summary is the richer source of truth for downstream systems.
- Compare should continue to work without changing its output because it will
  adapt internally to read the new nested system schema.
"""

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

from traceml.reporting.summaries.summary_formatting import (
    bytes_to_gb,
    duration_from_bounds,
    format_optional,
)
from traceml.reporting.summaries.summary_io import (
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


@dataclass
class PerGPUSummary:
    """
    Aggregated metrics for one physical GPU across the sampled summary window.

    Notes
    -----
    - Values are aggregated across all selected `system_gpu_samples` rows for
      the device index.
    - Memory values remain in raw bytes while aggregating and are converted
      only during final summary serialization.
    """

    gpu_idx: int

    util_avg_percent: Optional[float] = None
    util_peak_percent: Optional[float] = None

    mem_avg_bytes: Optional[float] = None
    mem_peak_bytes: Optional[float] = None
    mem_total_bytes: Optional[float] = None

    temp_avg_c: Optional[float] = None
    temp_peak_c: Optional[float] = None

    power_avg_w: Optional[float] = None
    power_peak_w: Optional[float] = None


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


def _load_per_gpu_summary(
    conn: sqlite3.Connection,
    *,
    rank: Optional[int] = None,
    max_system_rows: int = 5_000,
) -> Dict[int, PerGPUSummary]:
    """
    Load per-GPU aggregated metrics from `system_gpu_samples`.

    Parameters
    ----------
    conn:
        Open SQLite connection.
    rank:
        Optional rank filter. If None, aggregates across all ranks.
    max_system_rows:
        Safety cap on the number of parent `system_samples` rows included in the
        summary window. Per-GPU rows are restricted to those parent rows.

    Returns
    -------
    Dict[int, PerGPUSummary]
        Mapping from GPU index to aggregated summary metrics.

    Notes
    -----
    This function intentionally uses the same bounded `system_samples` window as
    the top-level system summary so that the rollup and per-GPU views describe
    the same time range.
    """
    where_clause = ""
    params: list[Any] = []

    if rank is not None:
        where_clause = "WHERE s.rank = ?"
        params.append(int(rank))

    sql = f"""
        SELECT
            g.gpu_idx,

            AVG(g.util),
            MAX(g.util),

            AVG(g.mem_used_bytes),
            MAX(g.mem_used_bytes),
            MAX(g.mem_total_bytes),

            AVG(g.temperature_c),
            MAX(g.temperature_c),

            AVG(g.power_usage_w),
            MAX(g.power_usage_w)

        FROM system_gpu_samples AS g
        INNER JOIN (
            SELECT s.rank, s.seq
            FROM system_samples AS s
            {where_clause}
            ORDER BY s.id ASC
            LIMIT ?
        ) AS recent
            ON g.rank IS recent.rank
           AND g.seq = recent.seq
        GROUP BY g.gpu_idx
        ORDER BY g.gpu_idx ASC;
    """

    rows = conn.execute(sql, (*params, int(max_system_rows))).fetchall()

    out: Dict[int, PerGPUSummary] = {}
    for row in rows:
        gpu_idx = int(row[0])
        out[gpu_idx] = PerGPUSummary(
            gpu_idx=gpu_idx,
            util_avg_percent=float(row[1]) if row[1] is not None else None,
            util_peak_percent=float(row[2]) if row[2] is not None else None,
            mem_avg_bytes=float(row[3]) if row[3] is not None else None,
            mem_peak_bytes=float(row[4]) if row[4] is not None else None,
            mem_total_bytes=float(row[5]) if row[5] is not None else None,
            temp_avg_c=float(row[6]) if row[6] is not None else None,
            temp_peak_c=float(row[7]) if row[7] is not None else None,
            power_avg_w=float(row[8]) if row[8] is not None else None,
            power_peak_w=float(row[9]) if row[9] is not None else None,
        )
    return out


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


def _per_gpu_to_json(
    per_gpu: Dict[int, PerGPUSummary],
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Convert per-GPU aggregates into a JSON-friendly dictionary keyed by GPU idx.
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}

    for gpu_idx, item in sorted(per_gpu.items()):
        out[str(gpu_idx)] = {
            "util_avg_percent": item.util_avg_percent,
            "util_peak_percent": item.util_peak_percent,
            "mem_avg_gb": bytes_to_gb(item.mem_avg_bytes),
            "mem_peak_gb": bytes_to_gb(item.mem_peak_bytes),
            "mem_total_gb": bytes_to_gb(item.mem_total_bytes),
            "temp_avg_c": item.temp_avg_c,
            "temp_peak_c": item.temp_peak_c,
            "power_avg_w": item.power_avg_w,
            "power_peak_w": item.power_peak_w,
        }

    return out


def _best_gpu_idx(
    per_gpu: Dict[int, PerGPUSummary],
    attr_name: str,
) -> Optional[int]:
    """
    Return the GPU index with the largest value for `attr_name`.
    """
    best_idx: Optional[int] = None
    best_value: Optional[float] = None

    for gpu_idx, item in per_gpu.items():
        value = getattr(item, attr_name, None)
        if value is None:
            continue
        if best_value is None or float(value) > float(best_value):
            best_idx = int(gpu_idx)
            best_value = float(value)

    return best_idx


def _build_system_card(
    agg: SystemSummaryAgg,
    *,
    per_gpu: Dict[int, PerGPUSummary],
) -> tuple[str, Dict[str, Any]]:
    """
    Build a compact, shareable end-of-run system summary.

    Returns
    -------
    tuple[str, Dict[str, Any]]
        - text block for stdout / saved text summaries
        - structured JSON payload

    Notes
    -----
    The printed summary is intentionally compact and continues to show a single
    GPU line for readability.

    The structured JSON summary is richer and includes:
    - nested CPU and RAM rollups
    - a `gpu_rollup` block summarizing the overall GPU subsystem
    - a `per_gpu` block preserving per-device aggregates for dashboards,
      logging, and future analysis
    """
    duration_s = duration_from_bounds(agg.first_ts, agg.last_ts)

    ram_avg_gb = bytes_to_gb(agg.ram_avg_bytes)
    ram_peak_gb = bytes_to_gb(agg.ram_peak_bytes)
    ram_total_gb = bytes_to_gb(agg.ram_total_bytes)

    gpu_mem_avg_gb = bytes_to_gb(agg.gpu_mem_avg_bytes)
    gpu_mem_peak_gb = bytes_to_gb(agg.gpu_mem_peak_bytes)

    hottest_gpu_idx = _best_gpu_idx(per_gpu, "temp_peak_c")
    highest_mem_gpu_idx = _best_gpu_idx(per_gpu, "mem_peak_bytes")
    highest_util_gpu_idx = _best_gpu_idx(per_gpu, "util_peak_percent")

    hottest_gpu_temp_peak_c = (
        per_gpu[hottest_gpu_idx].temp_peak_c
        if hottest_gpu_idx is not None
        else None
    )
    highest_mem_peak_gb = (
        bytes_to_gb(per_gpu[highest_mem_gpu_idx].mem_peak_bytes)
        if highest_mem_gpu_idx is not None
        else None
    )
    highest_util_peak_percent = (
        per_gpu[highest_util_gpu_idx].util_peak_percent
        if highest_util_gpu_idx is not None
        else None
    )

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
        "cpu": {
            "avg_percent": agg.cpu_avg_percent,
            "peak_percent": agg.cpu_peak_percent,
        },
        "ram": {
            "avg_gb": ram_avg_gb,
            "peak_gb": ram_peak_gb,
            "total_gb": ram_total_gb,
        },
        "gpu_rollup": {
            "available": agg.gpu_available,
            "count": agg.gpu_count,
            "util_avg_percent": agg.gpu_util_avg_percent,
            "util_peak_percent": agg.gpu_util_peak_percent,
            "mem_avg_gb": gpu_mem_avg_gb,
            "mem_peak_gb": gpu_mem_peak_gb,
            "temp_avg_c": agg.gpu_temp_avg_c,
            "temp_peak_c": agg.gpu_temp_peak_c,
            "power_avg_w": agg.gpu_power_avg_w,
            "power_peak_w": agg.gpu_power_peak_w,
            "hottest_gpu_idx": hottest_gpu_idx,
            "hottest_gpu_temp_peak_c": hottest_gpu_temp_peak_c,
            "highest_mem_gpu_idx": highest_mem_gpu_idx,
            "highest_mem_peak_gb": highest_mem_peak_gb,
            "highest_util_gpu_idx": highest_util_gpu_idx,
            "highest_util_peak_percent": highest_util_peak_percent,
        },
        "per_gpu": _per_gpu_to_json(per_gpu),
        "units": {
            "memory": "GB",
            "temperature": "C",
            "power": "W",
            "util": "%",
        },
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
        per_gpu = _load_per_gpu_summary(
            conn,
            rank=rank,
            max_system_rows=max_system_rows,
        )
    finally:
        conn.close()

    card, summary = _build_system_card(agg, per_gpu=per_gpu)

    with open(db_path + "_summary_card.txt", "w", encoding="utf-8") as f:
        f.write(card + "\n")

    existing = load_json_or_empty(db_path + "_summary_card.json")
    existing["system"] = summary
    write_json(db_path + "_summary_card.json", existing)

    if print_to_stdout:
        print(card)

    return summary
