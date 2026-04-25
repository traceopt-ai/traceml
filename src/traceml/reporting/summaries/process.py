"""
Compact end-of-run process summary generation.

This module reads traced workload process metrics from the `process_samples`
projection table and produces:

1. a compact text summary for end-of-run display and sharing
2. a structured JSON payload for automation, logging, and future dashboard use

Design goals
------------
- Keep the printed summary short and easy to scan
- Use one clear canonical schema for process summary data
- Preserve richer machine-readable fields in JSON
- Use the schema 1.2 section contract:
  - `overview` for scope metadata
  - `primary_diagnosis` for the concise user-facing diagnosis
  - `global` for workload-level CPU, RAM, GPU, and takeaway data
  - `per_rank` for traced-rank detail

Notes
-----
- The printed text intentionally remains compact.
- The JSON summary is the richer source of truth for downstream systems.
"""

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

from traceml.diagnostics.common import diagnosis_to_dict
from traceml.diagnostics.process import build_process_diagnosis_result
from traceml.reporting.summaries.issue_summary import (
    issues_by_metric_json,
    issues_by_rank_json,
    issues_compact_text,
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
            MAX(gpu_mem_total_bytes)

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
        )
    return out


def _build_takeaway(agg: ProcessSummaryAgg) -> str:
    """
    Build one concise process-centric takeaway line.

    The goal is to highlight a useful process-local interpretation rather than
    merely restating raw averages.
    """
    if agg.process_samples <= 0:
        return "n/a"

    if agg.gpu_available and agg.gpu_mem_total_bytes is not None:
        used_peak_pct = share_percent(
            agg.gpu_mem_used_peak_bytes,
            agg.gpu_mem_total_bytes,
        )
        if (
            agg.gpu_mem_reserved_peak_bytes is not None
            and agg.gpu_mem_used_peak_bytes is not None
            and agg.gpu_mem_reserved_peak_bytes
            > agg.gpu_mem_used_peak_bytes * 1.25
        ):
            return "reserved GPU memory exceeds active use."

        if used_peak_pct is not None and used_peak_pct >= 90.0:
            return "GPU memory use is close to capacity"
        if used_peak_pct is not None and used_peak_pct <= 20.0:
            return "GPU memory use stayed low"

    if (
        agg.cpu_avg_percent is not None
        and agg.cpu_logical_core_count is not None
        and agg.cpu_logical_core_count > 0
    ):
        approx_single_core_pct = 100.0 / agg.cpu_logical_core_count
        if agg.cpu_avg_percent <= approx_single_core_pct * 1.5:
            return "CPU usage stayed low"

    return "stable overall"


def _build_gpu_line(
    *,
    gpu_available: Optional[bool],
    gpu_count: Optional[int],
    gpu_device_index: Optional[int],
    gpu_mem_used_peak_gb: Optional[float],
    gpu_mem_total_gb: Optional[float],
    gpu_mem_used_peak_pct: Optional[float],
    gpu_mem_reserved_peak_gb: Optional[float],
    gpu_mem_reserved_peak_pct: Optional[float],
) -> str:
    """
    Build one compact GPU line for the printed summary.

    This keeps the printed summary concise while still surfacing the process
    GPU memory information that matters most for performance and efficiency:
    - whether the process touched a GPU
    - which device it reported against
    - peak used memory
    - how close peak used memory came to the limit
    - whether reserved memory materially exceeded used memory
    """
    if gpu_available and gpu_mem_total_gb is not None:
        device_text = (
            f"device {gpu_device_index}"
            if gpu_device_index is not None
            else "device n/a"
        )
        parts = [
            device_text,
            f"used peak {format_optional(gpu_mem_used_peak_gb, ' GB', 1)}"
            f" / {format_optional(gpu_mem_total_gb, ' GB', 1)}",
        ]

        if gpu_mem_used_peak_pct is not None:
            parts.append(
                f"{format_optional(gpu_mem_used_peak_pct, '%', 1)} of limit"
            )

        if (
            gpu_mem_reserved_peak_gb is not None
            and gpu_mem_used_peak_gb is not None
            and gpu_mem_reserved_peak_gb > gpu_mem_used_peak_gb * 1.25
        ):
            parts.append(
                f"reserved peak {format_optional(gpu_mem_reserved_peak_gb, ' GB', 1)}"
            )
            if gpu_mem_reserved_peak_pct is not None:
                parts.append(
                    f"reserved {format_optional(gpu_mem_reserved_peak_pct, '%', 1)}"
                )

        return "GPU: " + " | ".join(parts)

    if gpu_available is False:
        return "GPU: no GPU process samples were recorded"

    if (gpu_count or 0) > 0:
        return "GPU: detected, but no process GPU memory samples were recorded"

    return "GPU: n/a"


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
            "gpu_mem_used_avg_gb": bytes_to_gb(item.gpu_mem_used_avg_bytes),
            "gpu_mem_used_peak_gb": used_peak_gb,
            "gpu_mem_reserved_avg_gb": bytes_to_gb(
                item.gpu_mem_reserved_avg_bytes
            ),
            "gpu_mem_reserved_peak_gb": reserved_peak_gb,
            "gpu_mem_total_gb": total_gb,
            "gpu_mem_used_peak_pct": used_peak_pct,
            "gpu_mem_reserved_peak_pct": reserved_peak_pct,
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
    per_rank_for_diagnosis = _per_rank_to_diagnosis_input(per_rank)

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

    takeaway = _build_takeaway(agg)
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
        (f"- Scope: ranks {agg.distinct_ranks} | pids {agg.distinct_pids}"),
        (
            f"- CPU: avg {format_optional(agg.cpu_avg_percent, '%', 1)}, "
            f"peak {format_optional(agg.cpu_peak_percent, '%', 1)}"
            + (
                f" | cores {format_optional(float(agg.cpu_logical_core_count), '', 0)}"
                if agg.cpu_logical_core_count is not None
                else ""
            )
        ),
        (
            f"- RSS: avg {format_optional(ram_avg_gb, ' GB', 1)}, "
            f"peak {format_optional(ram_peak_gb, ' GB', 1)} / {format_optional(ram_total_gb, ' GB', 1)}"
        ),
        (
            "- "
            + _build_gpu_line(
                gpu_available=agg.gpu_available,
                gpu_count=agg.gpu_count,
                gpu_device_index=agg.gpu_device_index,
                gpu_mem_used_peak_gb=gpu_mem_used_peak_gb,
                gpu_mem_total_gb=gpu_mem_total_gb,
                gpu_mem_used_peak_pct=gpu_mem_used_peak_pct,
                gpu_mem_reserved_peak_gb=gpu_mem_reserved_peak_gb,
                gpu_mem_reserved_peak_pct=gpu_mem_reserved_peak_pct,
            )
        ),
        f"- Takeaway: {takeaway}",
        f"- Diagnosis: {primary_diagnosis.status}",
        f"- Why: {primary_diagnosis.reason}",
        f"- Next: {primary_diagnosis.action}",
    ]
    issue_text = issues_compact_text(issues, max_items=4)
    if issue_text:
        lines.append(f"- Issues: {issue_text}")
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
        },
        "ram": {
            "avg_gb": ram_avg_gb,
            "peak_gb": ram_peak_gb,
            "total_gb": ram_total_gb,
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
            "reserved_peak_pct": gpu_mem_reserved_peak_pct,
            "highest_used_rank": highest_used_rank,
            "highest_used_peak_gb": highest_used_peak_gb,
            "highest_reserved_rank": highest_reserved_rank,
            "highest_reserved_peak_gb": highest_reserved_peak_gb,
            "least_headroom_rank": least_headroom_rank,
            "least_headroom_gb": least_headroom_gb,
        },
        "takeaway": takeaway,
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
    max_process_rows = min(max(1, int(max_process_rows)), MAX_SUMMARY_ROWS)
    conn = sqlite3.connect(db_path)
    try:
        agg = _load_process_summary_agg(
            conn,
            rank=rank,
            max_process_rows=max_process_rows,
        )
        per_rank = _load_per_rank_process_summary(
            conn,
            rank=rank,
            max_process_rows=max_process_rows,
        )
    finally:
        conn.close()

    card, process_summary = _build_process_card(agg, per_rank=per_rank)

    append_text(db_path + "_summary_card.txt", card)

    existing = load_json_or_empty(db_path + "_summary_card.json")
    existing["process"] = process_summary
    write_json(db_path + "_summary_card.json", existing)

    if print_to_stdout:
        print(card)

    return process_summary
