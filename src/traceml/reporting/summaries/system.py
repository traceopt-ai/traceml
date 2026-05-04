"""End-of-run system summary generation."""

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

from traceml.diagnostics.bands import Band
from traceml.diagnostics.system.policy import DEFAULT_SYSTEM_POLICY
from traceml.reporting.summaries.summary_formatting import bytes_to_gb
from traceml.reporting.summaries.summary_io import (
    load_json_or_empty,
    write_json,
)

MAX_SUMMARY_ROWS = 10_000


def _table_has_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
    return any(str(row[1]) == column_name for row in rows)


def _percent(
    numerator: Optional[float],
    denominator: Optional[float],
) -> Optional[float]:
    if numerator is None or denominator is None or float(denominator) <= 0.0:
        return None
    return max(0.0, float(numerator) / float(denominator) * 100.0)


def _band_name(band: Optional[Band]) -> Optional[str]:
    return None if band is None else str(band)


def _format_percent_stat(
    label: str,
    value: Optional[float],
) -> Optional[str]:
    if value is None:
        return None
    return f"{label} {float(value):.0f}%"


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
    power_limit_w: Optional[float] = None


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

    power_limit_expr = (
        "MAX(g.power_limit_w)"
        if _table_has_column(conn, "system_gpu_samples", "power_limit_w")
        else "NULL"
    )

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
            MAX(g.power_usage_w),
            {power_limit_expr}

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
            power_limit_w=float(row[10]) if row[10] is not None else None,
        )
    return out


def _highest_gpu_memory_percent(
    per_gpu: Dict[int, PerGPUSummary],
) -> Optional[float]:
    values = [
        _percent(item.mem_peak_bytes, item.mem_total_bytes)
        for item in per_gpu.values()
    ]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _build_stats_line(
    agg: SystemSummaryAgg,
    *,
    per_gpu: Dict[int, PerGPUSummary],
) -> str:
    ram_peak_percent = _percent(agg.ram_peak_bytes, agg.ram_total_bytes)
    gpu_mem_peak_percent = _highest_gpu_memory_percent(per_gpu)

    parts = [
        _format_percent_stat("CPU avg", agg.cpu_avg_percent),
        _format_percent_stat("RAM peak", ram_peak_percent),
        _format_percent_stat("GPU util avg", agg.gpu_util_avg_percent),
        _format_percent_stat("GPU memory peak", gpu_mem_peak_percent),
    ]
    rendered = [part for part in parts if part is not None]
    return " | ".join(rendered) if rendered else "unavailable"


def _per_gpu_to_json(
    per_gpu: Dict[int, PerGPUSummary],
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Convert per-GPU aggregates into a JSON-friendly dictionary keyed by GPU idx.
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}

    for gpu_idx, item in sorted(per_gpu.items()):
        mem_peak_percent = _percent(item.mem_peak_bytes, item.mem_total_bytes)
        power_avg_limit_percent = _percent(
            item.power_avg_w,
            item.power_limit_w,
        )
        out[str(gpu_idx)] = {
            "util_avg_percent": item.util_avg_percent,
            "util_peak_percent": item.util_peak_percent,
            "util_avg_band": _band_name(
                DEFAULT_SYSTEM_POLICY.gpu_util_avg_percent.classify(
                    item.util_avg_percent
                )
            ),
            "mem_avg_gb": bytes_to_gb(item.mem_avg_bytes),
            "mem_peak_gb": bytes_to_gb(item.mem_peak_bytes),
            "mem_total_gb": bytes_to_gb(item.mem_total_bytes),
            "mem_peak_percent": mem_peak_percent,
            "mem_peak_band": _band_name(
                DEFAULT_SYSTEM_POLICY.gpu_memory_peak_percent.classify(
                    mem_peak_percent
                )
            ),
            "temp_avg_c": item.temp_avg_c,
            "temp_peak_c": item.temp_peak_c,
            "temp_peak_band": _band_name(
                DEFAULT_SYSTEM_POLICY.gpu_temp_peak_c.classify(
                    item.temp_peak_c
                )
            ),
            "power_avg_w": item.power_avg_w,
            "power_peak_w": item.power_peak_w,
            "power_limit_w": item.power_limit_w,
            "power_avg_limit_percent": power_avg_limit_percent,
            "power_avg_band": _band_name(
                DEFAULT_SYSTEM_POLICY.gpu_power_avg_limit_percent.classify(
                    power_avg_limit_percent
                )
            ),
        }

    return out


def _per_gpu_to_diagnosis_input(
    per_gpu: Dict[int, PerGPUSummary],
) -> Dict[int, Dict[str, Optional[float]]]:
    """
    Convert per-GPU summary objects into the bytes-based shape used by system
    diagnosis rules.

    Notes
    -----
    Summary JSON intentionally exposes memory in GB for readability, while the
    diagnosis layer keeps raw bytes so pressure calculations remain precise.
    """
    out: Dict[int, Dict[str, Optional[float]]] = {}

    for gpu_idx, item in sorted(per_gpu.items()):
        out[int(gpu_idx)] = {
            "util_avg_percent": item.util_avg_percent,
            "util_peak_percent": item.util_peak_percent,
            "mem_avg_bytes": item.mem_avg_bytes,
            "mem_peak_bytes": item.mem_peak_bytes,
            "mem_total_bytes": item.mem_total_bytes,
            "temp_avg_c": item.temp_avg_c,
            "temp_peak_c": item.temp_peak_c,
            "power_avg_w": item.power_avg_w,
            "power_peak_w": item.power_peak_w,
            "power_limit_w": item.power_limit_w,
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


def generate_system_summary_card(
    db_path: str,
    *,
    rank: Optional[int] = None,
    print_to_stdout: bool = True,
    max_system_rows: int = MAX_SUMMARY_ROWS,
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
    from traceml.reporting.sections.system import SystemSummarySection

    result = SystemSummarySection(
        rank=rank,
        max_system_rows=max_system_rows,
    ).build(db_path)
    card = result.text
    summary = result.payload

    with open(db_path + "_summary_card.txt", "w", encoding="utf-8") as f:
        f.write(card + "\n")

    existing = load_json_or_empty(db_path + "_summary_card.json")
    existing["system"] = summary
    write_json(db_path + "_summary_card.json", existing)

    if print_to_stdout:
        print(card)

    return summary
