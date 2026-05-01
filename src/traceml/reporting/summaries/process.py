"""End-of-run process summary generation."""

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

from traceml.diagnostics.bands import Band
from traceml.diagnostics.common import diagnosis_to_dict
from traceml.diagnostics.process import build_process_diagnosis_result
from traceml.diagnostics.process.policy import DEFAULT_PROCESS_POLICY
from traceml.reporting.summaries.issue_summary import (
    issues_by_metric_json,
    issues_by_rank_json,
    issues_to_json,
)
from traceml.reporting.summaries.summary_formatting import (
    bytes_to_gb,
    duration_from_bounds,
    format_optional,
    share_percent,
)
from traceml.reporting.summaries.summary_io import (
    append_text,
    load_json_or_empty,
    write_json,
)

MAX_SUMMARY_ROWS = 10_000


def _band_name(band: Optional[Band]) -> Optional[str]:
    return None if band is None else str(band)


def _format_percent_stat(
    label: str,
    value: Optional[float],
) -> Optional[str]:
    if value is None:
        return None
    return f"{label} {float(value):.0f}%"


def _format_memory_stat(
    label: str,
    value_gb: Optional[float],
    total_gb: Optional[float],
) -> Optional[str]:
    if value_gb is None:
        return None
    if total_gb is None:
        return f"{label} {float(value_gb):.1f} GB"
    return f"{label} {float(value_gb):.1f} / {float(total_gb):.1f} GB"


@dataclass
class ProcessSummaryAgg:
    """
    Aggregated process metrics loaded from `process_samples`.

    Notes
    -----
    - Memory values remain in raw bytes during aggregation and are converted
      only at formatting / serialization time.
    - Fields are intentionally broad enough to support both single-process and
      distributed runs.
    """

    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    process_samples: int = 0

    distinct_ranks: int = 0
    distinct_pids: int = 0

    cpu_avg_percent: Optional[float] = None
    cpu_peak_percent: Optional[float] = None
    cpu_logical_core_count: Optional[int] = None

    ram_avg_bytes: Optional[float] = None
    ram_peak_bytes: Optional[float] = None
    ram_total_bytes: Optional[float] = None

    gpu_available: Optional[bool] = None
    gpu_count: Optional[int] = None
    gpu_device_index: Optional[int] = None

    gpu_mem_used_avg_bytes: Optional[float] = None
    gpu_mem_used_peak_bytes: Optional[float] = None
    gpu_mem_reserved_avg_bytes: Optional[float] = None
    gpu_mem_reserved_peak_bytes: Optional[float] = None
    gpu_mem_total_bytes: Optional[float] = None
    gpu_mem_reserved_overhang_ratio: Optional[float] = None


@dataclass
class PerRankProcessSummary:
    """
    Aggregated traced-process metrics for one rank across the sampled summary
    window.

    Notes
    -----
    - Values are aggregated across all selected `process_samples` rows for the
      rank.
    - Memory values remain in raw bytes while aggregating and are converted only
      during final summary serialization.
    - `pid_count` is included because a rank may emit from more than one traced
      process over the sampled window in some environments.
    """

    rank: int
    pid_count: int = 0

    cpu_avg_percent: Optional[float] = None
    cpu_peak_percent: Optional[float] = None
    cpu_logical_core_count: Optional[int] = None

    ram_avg_bytes: Optional[float] = None
    ram_peak_bytes: Optional[float] = None
    ram_total_bytes: Optional[float] = None

    gpu_available: Optional[bool] = None
    gpu_count: Optional[int] = None
    gpu_device_index: Optional[int] = None

    gpu_mem_used_avg_bytes: Optional[float] = None
    gpu_mem_used_peak_bytes: Optional[float] = None
    gpu_mem_reserved_avg_bytes: Optional[float] = None
    gpu_mem_reserved_peak_bytes: Optional[float] = None
    gpu_mem_total_bytes: Optional[float] = None
    gpu_mem_reserved_overhang_ratio: Optional[float] = None


def _load_process_summary_agg(
    conn: sqlite3.Connection,
    *,
    rank: Optional[int] = None,
    max_process_rows: int = 10_000,
) -> ProcessSummaryAgg:
    """
    Load aggregated process metrics directly from `process_samples`.

    Parameters
    ----------
    conn:
        Open SQLite connection.
    rank:
        Optional rank filter. If None, aggregates across all ranks.
    max_process_rows:
        Safety cap on rows included in aggregation.

    Returns
    -------
    ProcessSummaryAgg
        Aggregated summary values ready for formatting.
    """
    where_clause = ""
    params: list[Any] = []

    if rank is not None:
        where_clause = "WHERE rank = ?"
        params.append(int(rank))

    base_sql = f"""
        FROM (
            SELECT *
            FROM process_samples
            {where_clause}
            ORDER BY id ASC
            LIMIT ?
        )
    """

    count_row = conn.execute(
        f"""
        SELECT
            COUNT(*),
            MIN(sample_ts_s),
            MAX(sample_ts_s),
            COUNT(DISTINCT rank),
            COUNT(DISTINCT pid)
        {base_sql};
        """,
        (*params, int(max_process_rows)),
    ).fetchone()

    n_rows = int(count_row[0] or 0)
    first_ts = float(count_row[1]) if count_row[1] is not None else None
    last_ts = float(count_row[2]) if count_row[2] is not None else None
    distinct_ranks = int(count_row[3] or 0)
    distinct_pids = int(count_row[4] or 0)

    row = conn.execute(
        f"""
        SELECT
            AVG(cpu_percent),
            MAX(cpu_percent),
            MAX(cpu_logical_core_count),

            AVG(ram_used_bytes),
            MAX(ram_used_bytes),
            MAX(ram_total_bytes),

            MAX(gpu_available),
            MAX(gpu_count),
            MIN(gpu_device_index),

            AVG(gpu_mem_used_bytes),
            MAX(gpu_mem_used_bytes),
            AVG(gpu_mem_reserved_bytes),
            MAX(gpu_mem_reserved_bytes),
            MAX(gpu_mem_total_bytes)
        {base_sql};
        """,
        (*params, int(max_process_rows)),
    ).fetchone()

    return ProcessSummaryAgg(
        first_ts=first_ts,
        last_ts=last_ts,
        process_samples=n_rows,
        distinct_ranks=distinct_ranks,
        distinct_pids=distinct_pids,
        cpu_avg_percent=float(row[0]) if row[0] is not None else None,
        cpu_peak_percent=float(row[1]) if row[1] is not None else None,
        cpu_logical_core_count=int(row[2]) if row[2] is not None else None,
        ram_avg_bytes=float(row[3]) if row[3] is not None else None,
        ram_peak_bytes=float(row[4]) if row[4] is not None else None,
        ram_total_bytes=float(row[5]) if row[5] is not None else None,
        gpu_available=bool(row[6]) if row[6] is not None else None,
        gpu_count=int(row[7]) if row[7] is not None else None,
        gpu_device_index=int(row[8]) if row[8] is not None else None,
        gpu_mem_used_avg_bytes=float(row[9]) if row[9] is not None else None,
        gpu_mem_used_peak_bytes=(
            float(row[10]) if row[10] is not None else None
        ),
        gpu_mem_reserved_avg_bytes=(
            float(row[11]) if row[11] is not None else None
        ),
        gpu_mem_reserved_peak_bytes=(
            float(row[12]) if row[12] is not None else None
        ),
        gpu_mem_total_bytes=float(row[13]) if row[13] is not None else None,
    )


def _load_per_rank_process_summary(
    conn: sqlite3.Connection,
    *,
    rank: Optional[int] = None,
    max_process_rows: int = 10_000,
) -> Dict[int, PerRankProcessSummary]:
    """
    Load per-rank aggregated process metrics from `process_samples`.

    Parameters
    ----------
    conn:
        Open SQLite connection.
    rank:
        Optional rank filter. If None, aggregates across all ranks.
    max_process_rows:
        Safety cap on rows included in aggregation.

    Returns
    -------
    Dict[int, PerRankProcessSummary]
        Mapping rank -> aggregated traced-process metrics.

    Notes
    -----
    This uses the same bounded `process_samples` window as the top-level
    process summary so that the rollup and per-rank views describe the same
    time range.
    """
    where_clause = ""
    params: list[Any] = []

    if rank is not None:
        where_clause = "WHERE rank = ?"
        params.append(int(rank))

    sql = f"""
        SELECT
            rank,
            COUNT(DISTINCT pid),

            AVG(cpu_percent),
            MAX(cpu_percent),
            MAX(cpu_logical_core_count),

            AVG(ram_used_bytes),
            MAX(ram_used_bytes),
            MAX(ram_total_bytes),

            MAX(gpu_available),
            MAX(gpu_count),
            MIN(gpu_device_index),

            AVG(gpu_mem_used_bytes),
            MAX(gpu_mem_used_bytes),
            AVG(gpu_mem_reserved_bytes),
            MAX(gpu_mem_reserved_bytes),
            MAX(gpu_mem_total_bytes),
            MAX(
                CASE
                    WHEN gpu_mem_used_bytes IS NOT NULL
                     AND gpu_mem_used_bytes > 0
                     AND gpu_mem_reserved_bytes IS NOT NULL
                    THEN gpu_mem_reserved_bytes / gpu_mem_used_bytes
                    ELSE NULL
                END
            )

        FROM (
            SELECT *
            FROM process_samples
            {where_clause}
            ORDER BY id ASC
            LIMIT ?
        )
        WHERE rank IS NOT NULL
        GROUP BY rank
        ORDER BY rank ASC;
    """

    rows = conn.execute(sql, (*params, int(max_process_rows))).fetchall()

    out: Dict[int, PerRankProcessSummary] = {}
    for row in rows:
        rank_id = int(row[0])
        out[rank_id] = PerRankProcessSummary(
            rank=rank_id,
            pid_count=int(row[1] or 0),
            cpu_avg_percent=float(row[2]) if row[2] is not None else None,
            cpu_peak_percent=float(row[3]) if row[3] is not None else None,
            cpu_logical_core_count=int(row[4]) if row[4] is not None else None,
            ram_avg_bytes=float(row[5]) if row[5] is not None else None,
            ram_peak_bytes=float(row[6]) if row[6] is not None else None,
            ram_total_bytes=float(row[7]) if row[7] is not None else None,
            gpu_available=bool(row[8]) if row[8] is not None else None,
            gpu_count=int(row[9]) if row[9] is not None else None,
            gpu_device_index=int(row[10]) if row[10] is not None else None,
            gpu_mem_used_avg_bytes=(
                float(row[11]) if row[11] is not None else None
            ),
            gpu_mem_used_peak_bytes=(
                float(row[12]) if row[12] is not None else None
            ),
            gpu_mem_reserved_avg_bytes=(
                float(row[13]) if row[13] is not None else None
            ),
            gpu_mem_reserved_peak_bytes=(
                float(row[14]) if row[14] is not None else None
            ),
            gpu_mem_total_bytes=(
                float(row[15]) if row[15] is not None else None
            ),
            gpu_mem_reserved_overhang_ratio=(
                float(row[16]) if row[16] is not None else None
            ),
        )
    return out


def _cpu_capacity_percent(agg: ProcessSummaryAgg) -> Optional[float]:
    if (
        agg.cpu_avg_percent is None
        or agg.cpu_logical_core_count is None
        or agg.cpu_logical_core_count <= 0
    ):
        return None
    return max(
        0.0,
        float(agg.cpu_avg_percent)
        / (100.0 * float(agg.cpu_logical_core_count))
        * 100.0,
    )


def _build_stats_line(
    agg: ProcessSummaryAgg,
    *,
    ram_peak_gb: Optional[float],
    ram_total_gb: Optional[float],
    gpu_mem_used_peak_pct: Optional[float],
    gpu_mem_reserved_peak_pct: Optional[float],
) -> str:
    gpu_pct = gpu_mem_reserved_peak_pct or gpu_mem_used_peak_pct
    gpu_label = (
        "GPU reserved peak"
        if gpu_mem_reserved_peak_pct is not None
        else "GPU used peak"
    )
    parts = [
        f"ranks {agg.distinct_ranks}",
        f"pids {agg.distinct_pids}",
        _format_percent_stat("CPU avg", agg.cpu_avg_percent),
        _format_memory_stat("RSS peak", ram_peak_gb, ram_total_gb),
        _format_percent_stat(gpu_label, gpu_pct),
    ]
    rendered = [part for part in parts if part is not None]
    return " | ".join(rendered) if rendered else "unavailable"


def _per_rank_to_json(
    per_rank: Dict[int, PerRankProcessSummary],
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Convert per-rank aggregates into a JSON-friendly dictionary keyed by rank.
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}

    for rank_id, item in sorted(per_rank.items()):
        used_peak_pct = share_percent(
            item.gpu_mem_used_peak_bytes,
            item.gpu_mem_total_bytes,
        )
        reserved_peak_pct = share_percent(
            item.gpu_mem_reserved_peak_bytes,
            item.gpu_mem_total_bytes,
        )
        ram_peak_pct = share_percent(item.ram_peak_bytes, item.ram_total_bytes)

        total_gb = bytes_to_gb(item.gpu_mem_total_bytes)
        used_peak_gb = bytes_to_gb(item.gpu_mem_used_peak_bytes)
        reserved_peak_gb = bytes_to_gb(item.gpu_mem_reserved_peak_bytes)

        out[str(rank_id)] = {
            "pid_count": float(item.pid_count),
            "gpu_device_index": (
                float(item.gpu_device_index)
                if item.gpu_device_index is not None
                else None
            ),
            "cpu_avg_percent": item.cpu_avg_percent,
            "cpu_peak_percent": item.cpu_peak_percent,
            "ram_avg_gb": bytes_to_gb(item.ram_avg_bytes),
            "ram_peak_gb": bytes_to_gb(item.ram_peak_bytes),
            "ram_total_gb": bytes_to_gb(item.ram_total_bytes),
            "ram_peak_percent": ram_peak_pct,
            "ram_peak_band": _band_name(
                DEFAULT_PROCESS_POLICY.rss_peak_percent.classify(ram_peak_pct)
            ),
            "gpu_mem_used_avg_gb": bytes_to_gb(item.gpu_mem_used_avg_bytes),
            "gpu_mem_used_peak_gb": used_peak_gb,
            "gpu_mem_reserved_avg_gb": bytes_to_gb(
                item.gpu_mem_reserved_avg_bytes
            ),
            "gpu_mem_reserved_peak_gb": reserved_peak_gb,
            "gpu_mem_total_gb": total_gb,
            "gpu_mem_used_peak_pct": used_peak_pct,
            "gpu_mem_used_peak_band": _band_name(
                DEFAULT_PROCESS_POLICY.gpu_memory_peak_percent.classify(
                    used_peak_pct
                )
            ),
            "gpu_mem_reserved_peak_pct": reserved_peak_pct,
            "gpu_mem_reserved_peak_band": _band_name(
                DEFAULT_PROCESS_POLICY.gpu_memory_peak_percent.classify(
                    reserved_peak_pct
                )
            ),
            "gpu_mem_reserved_overhang_ratio": (
                item.gpu_mem_reserved_overhang_ratio
            ),
            "gpu_mem_reserved_overhang_band": _band_name(
                DEFAULT_PROCESS_POLICY.gpu_reserved_overhang_ratio.classify(
                    item.gpu_mem_reserved_overhang_ratio
                )
            ),
            "gpu_mem_headroom_gb": (
                max(total_gb - reserved_peak_gb, 0.0)
                if total_gb is not None and reserved_peak_gb is not None
                else None
            ),
        }

    return out


def _per_rank_to_diagnosis_input(
    per_rank: Dict[int, PerRankProcessSummary],
) -> Dict[int, Dict[str, Optional[float]]]:
    """
    Convert per-rank process summaries into the bytes-based shape used by
    process diagnosis rules.

    Notes
    -----
    Summary JSON intentionally exposes memory in GB for readability, while the
    diagnosis layer keeps raw bytes so pressure calculations remain precise.
    """
    out: Dict[int, Dict[str, Optional[float]]] = {}

    for rank_id, item in sorted(per_rank.items()):
        out[int(rank_id)] = {
            "pid_count": float(item.pid_count),
            "gpu_device_index": (
                float(item.gpu_device_index)
                if item.gpu_device_index is not None
                else None
            ),
            "cpu_avg_percent": item.cpu_avg_percent,
            "cpu_peak_percent": item.cpu_peak_percent,
            "ram_avg_bytes": item.ram_avg_bytes,
            "ram_peak_bytes": item.ram_peak_bytes,
            "ram_total_bytes": item.ram_total_bytes,
            "gpu_mem_used_avg_bytes": item.gpu_mem_used_avg_bytes,
            "gpu_mem_used_peak_bytes": item.gpu_mem_used_peak_bytes,
            "gpu_mem_reserved_avg_bytes": item.gpu_mem_reserved_avg_bytes,
            "gpu_mem_reserved_peak_bytes": item.gpu_mem_reserved_peak_bytes,
            "gpu_mem_total_bytes": item.gpu_mem_total_bytes,
            "gpu_mem_reserved_overhang_ratio": (
                item.gpu_mem_reserved_overhang_ratio
            ),
        }
    return out


def _best_rank_idx(
    per_rank: Dict[int, PerRankProcessSummary],
    attr_name: str,
) -> Optional[int]:
    """
    Return the rank id with the largest value for `attr_name`.
    """
    best_idx: Optional[int] = None
    best_value: Optional[float] = None

    for rank_id, item in per_rank.items():
        value = getattr(item, attr_name, None)
        if value is None:
            continue
        if best_value is None or float(value) > float(best_value):
            best_idx = int(rank_id)
            best_value = float(value)

    return best_idx


def _build_process_card(
    agg: ProcessSummaryAgg,
    *,
    per_rank: Dict[int, PerRankProcessSummary],
) -> tuple[str, Dict[str, Any]]:
    """
    Build a compact, shareable end-of-run process summary.

    Returns
    -------
    tuple[str, Dict[str, Any]]
        - text block for stdout / saved text summaries
        - structured JSON payload

    Notes
    -----
    The printed text is intentionally compact and continues to show one concise
    process GPU line for readability.

    The structured JSON summary is richer and includes:
    - a `scope` block for traced ranks and pids
    - nested CPU and RAM rollups
    - a `gpu_rollup` block summarizing the traced workload GPU memory behavior
    - a `per_rank` block preserving per-rank process aggregates for dashboards,
      logging, and future analysis
    """
    duration_s = duration_from_bounds(agg.first_ts, agg.last_ts)

    ram_avg_gb = bytes_to_gb(agg.ram_avg_bytes)
    ram_peak_gb = bytes_to_gb(agg.ram_peak_bytes)
    ram_total_gb = bytes_to_gb(agg.ram_total_bytes)

    gpu_mem_used_avg_gb = bytes_to_gb(agg.gpu_mem_used_avg_bytes)
    gpu_mem_used_peak_gb = bytes_to_gb(agg.gpu_mem_used_peak_bytes)
    gpu_mem_reserved_avg_gb = bytes_to_gb(agg.gpu_mem_reserved_avg_bytes)
    gpu_mem_reserved_peak_gb = bytes_to_gb(agg.gpu_mem_reserved_peak_bytes)
    gpu_mem_total_gb = bytes_to_gb(agg.gpu_mem_total_bytes)

    gpu_mem_used_peak_pct = share_percent(
        agg.gpu_mem_used_peak_bytes,
        agg.gpu_mem_total_bytes,
    )
    gpu_mem_reserved_peak_pct = share_percent(
        agg.gpu_mem_reserved_peak_bytes,
        agg.gpu_mem_total_bytes,
    )
    cpu_capacity_percent = _cpu_capacity_percent(agg)
    ram_peak_percent = share_percent(agg.ram_peak_bytes, agg.ram_total_bytes)
    per_rank_for_diagnosis = _per_rank_to_diagnosis_input(per_rank)
    gpu_reserved_overhang_ratio = max(
        (
            item.gpu_mem_reserved_overhang_ratio
            for item in per_rank.values()
            if item.gpu_mem_reserved_overhang_ratio is not None
        ),
        default=None,
    )
    highest_overhang_rank = _best_rank_idx(
        per_rank,
        "gpu_mem_reserved_overhang_ratio",
    )

    highest_used_rank = _best_rank_idx(per_rank, "gpu_mem_used_peak_bytes")
    highest_reserved_rank = _best_rank_idx(
        per_rank, "gpu_mem_reserved_peak_bytes"
    )

    highest_used_peak_gb = (
        bytes_to_gb(per_rank[highest_used_rank].gpu_mem_used_peak_bytes)
        if highest_used_rank is not None
        else None
    )
    highest_reserved_peak_gb = (
        bytes_to_gb(
            per_rank[highest_reserved_rank].gpu_mem_reserved_peak_bytes
        )
        if highest_reserved_rank is not None
        else None
    )

    least_headroom_rank: Optional[int] = None
    least_headroom_gb: Optional[float] = None
    for rank_id, item in per_rank.items():
        total_gb = bytes_to_gb(item.gpu_mem_total_bytes)
        reserved_gb = bytes_to_gb(item.gpu_mem_reserved_peak_bytes)
        if total_gb is None or reserved_gb is None:
            continue
        headroom_gb = max(total_gb - reserved_gb, 0.0)
        if least_headroom_gb is None or headroom_gb < least_headroom_gb:
            least_headroom_rank = int(rank_id)
            least_headroom_gb = headroom_gb

    diagnosis_result = build_process_diagnosis_result(
        duration_s=duration_s,
        process_samples=agg.process_samples,
        distinct_ranks=agg.distinct_ranks,
        distinct_pids=agg.distinct_pids,
        cpu_avg_percent=agg.cpu_avg_percent,
        cpu_peak_percent=agg.cpu_peak_percent,
        cpu_logical_core_count=agg.cpu_logical_core_count,
        ram_avg_bytes=agg.ram_avg_bytes,
        ram_peak_bytes=agg.ram_peak_bytes,
        ram_total_bytes=agg.ram_total_bytes,
        gpu_available=agg.gpu_available,
        gpu_count=agg.gpu_count,
        gpu_device_index=agg.gpu_device_index,
        gpu_mem_used_avg_bytes=agg.gpu_mem_used_avg_bytes,
        gpu_mem_used_peak_bytes=agg.gpu_mem_used_peak_bytes,
        gpu_mem_reserved_avg_bytes=agg.gpu_mem_reserved_avg_bytes,
        gpu_mem_reserved_peak_bytes=agg.gpu_mem_reserved_peak_bytes,
        gpu_mem_total_bytes=agg.gpu_mem_total_bytes,
        per_rank=per_rank_for_diagnosis,
    )
    primary_diagnosis = diagnosis_result.primary
    issues = diagnosis_result.issues
    issues_by_rank, unassigned_issues = issues_by_rank_json(
        issues,
        rank_keys=per_rank.keys(),
    )
    issues_by_metric, metric_unassigned = issues_by_metric_json(issues)
    per_rank_json = _per_rank_to_json(per_rank)
    for rank_key, entry in per_rank_json.items():
        entry["issues"] = issues_by_rank.get(rank_key, [])

    lines = [
        f"TraceML Process Summary | duration {format_optional(duration_s, 's', 1)} | samples {agg.process_samples}",
        "Process",
        f"- Diagnosis: {primary_diagnosis.status}",
        (
            "- Stats: "
            + _build_stats_line(
                agg,
                ram_peak_gb=ram_peak_gb,
                ram_total_gb=ram_total_gb,
                gpu_mem_used_peak_pct=gpu_mem_used_peak_pct,
                gpu_mem_reserved_peak_pct=gpu_mem_reserved_peak_pct,
            )
        ),
        f"- Why: {primary_diagnosis.reason}",
    ]
    card = "\n".join(lines)

    global_summary = {
        "scope": {
            "ranks": agg.distinct_ranks,
            "pids": agg.distinct_pids,
        },
        "cpu": {
            "avg_percent": agg.cpu_avg_percent,
            "peak_percent": agg.cpu_peak_percent,
            "logical_core_count": agg.cpu_logical_core_count,
            "capacity_percent": cpu_capacity_percent,
            "capacity_band": _band_name(
                DEFAULT_PROCESS_POLICY.cpu_capacity_percent.classify(
                    cpu_capacity_percent
                )
            ),
        },
        "ram": {
            "avg_gb": ram_avg_gb,
            "peak_gb": ram_peak_gb,
            "total_gb": ram_total_gb,
            "peak_percent": ram_peak_percent,
            "peak_band": _band_name(
                DEFAULT_PROCESS_POLICY.rss_peak_percent.classify(
                    ram_peak_percent
                )
            ),
        },
        "gpu_rollup": {
            "available": agg.gpu_available,
            "count": agg.gpu_count,
            "device_index": agg.gpu_device_index,
            "used_avg_gb": gpu_mem_used_avg_gb,
            "used_peak_gb": gpu_mem_used_peak_gb,
            "reserved_avg_gb": gpu_mem_reserved_avg_gb,
            "reserved_peak_gb": gpu_mem_reserved_peak_gb,
            "total_gb": gpu_mem_total_gb,
            "used_peak_pct": gpu_mem_used_peak_pct,
            "used_peak_band": _band_name(
                DEFAULT_PROCESS_POLICY.gpu_memory_peak_percent.classify(
                    gpu_mem_used_peak_pct
                )
            ),
            "reserved_peak_pct": gpu_mem_reserved_peak_pct,
            "reserved_peak_band": _band_name(
                DEFAULT_PROCESS_POLICY.gpu_memory_peak_percent.classify(
                    gpu_mem_reserved_peak_pct
                )
            ),
            "reserved_overhang_ratio": gpu_reserved_overhang_ratio,
            "reserved_overhang_band": _band_name(
                DEFAULT_PROCESS_POLICY.gpu_reserved_overhang_ratio.classify(
                    gpu_reserved_overhang_ratio
                )
            ),
            "highest_overhang_rank": highest_overhang_rank,
            "highest_used_rank": highest_used_rank,
            "highest_used_peak_gb": highest_used_peak_gb,
            "highest_reserved_rank": highest_reserved_rank,
            "highest_reserved_peak_gb": highest_reserved_peak_gb,
            "least_headroom_rank": least_headroom_rank,
            "least_headroom_gb": least_headroom_gb,
        },
    }

    summary = {
        "overview": {
            "duration_s": duration_s,
            "samples": agg.process_samples,
            "ranks_seen": agg.distinct_ranks,
            "pids_seen": agg.distinct_pids,
        },
        "primary_diagnosis": diagnosis_to_dict(
            primary_diagnosis,
            drop_none=True,
            include_action=False,
        ),
        "issues": issues_to_json(issues),
        "issues_by_rank": issues_by_rank,
        "issues_by_metric": issues_by_metric,
        "unassigned_issues": unassigned_issues + metric_unassigned,
        "global": global_summary,
        "per_rank": per_rank_json,
        "units": {
            "memory": "GB",
            "cpu": "%",
        },
        "card": card,
    }
    return card, summary


def generate_process_summary_card(
    db_path: str,
    *,
    rank: Optional[int] = None,
    print_to_stdout: bool = True,
    max_process_rows: int = MAX_SUMMARY_ROWS,
) -> Dict[str, Any]:
    """
    Generate a compact PROCESS summary from SQL projection tables.

    Parameters
    ----------
    db_path:
        Path to the SQLite DB file.
    rank:
        Optional rank filter. If None, summarizes across all ranks.
    print_to_stdout:
        If True, print the rendered summary.
    max_process_rows:
        Safety cap on rows included in aggregation.

    Returns
    -------
    Dict[str, Any]
        Structured summary JSON including the rendered `card`.
    """
    from traceml.reporting.sections.process import ProcessSummarySection

    result = ProcessSummarySection(
        rank=rank,
        max_process_rows=max_process_rows,
    ).build(db_path)
    card = result.text
    process_summary = result.payload

    append_text(db_path + "_summary_card.txt", card)

    existing = load_json_or_empty(db_path + "_summary_card.json")
    existing["process"] = process_summary
    write_json(db_path + "_summary_card.json", existing)

    if print_to_stdout:
        print(card)

    return process_summary
