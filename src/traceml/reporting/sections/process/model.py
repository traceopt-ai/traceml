# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Process aggregation helpers for the final-report section."""

import sqlite3
from dataclasses import dataclass
from statistics import median
from typing import Any, Callable, Dict, Optional

from traceml.diagnostics.bands import Band
from traceml.diagnostics.process.policy import DEFAULT_PROCESS_POLICY
from traceml.reporting.summaries.summary_formatting import (
    bytes_to_gb,
    share_percent,
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
    Aggregated traced-process metrics for one global rank.

    Notes
    -----
    - Values are aggregated across all selected `process_samples` rows for the
      globally unique worker rank.
    - Memory values remain in raw bytes while aggregating and are converted only
      during final summary serialization.
    """

    global_rank: int
    local_rank: Optional[int] = None
    world_size: Optional[int] = None
    local_world_size: Optional[int] = None
    node_rank: Optional[int] = None
    hostname: Optional[str] = None

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
    global_rank: Optional[int] = None,
    max_process_rows: int = 10_000,
) -> ProcessSummaryAgg:
    """
    Load aggregated process metrics directly from `process_samples`.

    Parameters
    ----------
    conn:
        Open SQLite connection.
    global_rank:
        Optional global-rank filter. If None, aggregates across all ranks.
    max_process_rows:
        Safety cap on rows included in aggregation.

    Returns
    -------
    ProcessSummaryAgg
        Aggregated summary values ready for formatting.
    """
    where_clause = ""
    params: list[Any] = []

    if global_rank is not None:
        where_clause = "WHERE global_rank = ?"
        params.append(int(global_rank))

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
            COUNT(DISTINCT global_rank)
        {base_sql};
        """,
        (*params, int(max_process_rows)),
    ).fetchone()

    n_rows = int(count_row[0] or 0)
    first_ts = float(count_row[1]) if count_row[1] is not None else None
    last_ts = float(count_row[2]) if count_row[2] is not None else None
    distinct_ranks = int(count_row[3] or 0)

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
    global_rank: Optional[int] = None,
    max_process_rows: int = 10_000,
) -> Dict[int, PerRankProcessSummary]:
    """
    Load per-rank aggregated process metrics from `process_samples`.

    Parameters
    ----------
    conn:
        Open SQLite connection.
    global_rank:
        Optional global-rank filter. If None, aggregates across all ranks.
    max_process_rows:
        Safety cap on rows included in aggregation.

    Returns
    -------
    Dict[int, PerRankProcessSummary]
        Mapping global rank -> aggregated traced-process metrics.

    Notes
    -----
    This uses the same bounded `process_samples` window as the top-level
    process summary so that the rollup and per-rank views describe the same
    time range.
    """
    where_clause = ""
    params: list[Any] = []

    if global_rank is not None:
        where_clause = "WHERE global_rank = ?"
        params.append(int(global_rank))

    sql = f"""
        SELECT
            global_rank,
            MAX(local_rank),
            MAX(world_size),
            MAX(local_world_size),
            MAX(node_rank),
            MAX(hostname),

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
        WHERE global_rank IS NOT NULL
        GROUP BY global_rank
        ORDER BY global_rank ASC;
    """

    rows = conn.execute(sql, (*params, int(max_process_rows))).fetchall()

    out: Dict[int, PerRankProcessSummary] = {}
    for row in rows:
        rank_id = int(row[0])
        out[rank_id] = PerRankProcessSummary(
            global_rank=rank_id,
            local_rank=int(row[1]) if row[1] is not None else None,
            world_size=int(row[2]) if row[2] is not None else None,
            local_world_size=int(row[3]) if row[3] is not None else None,
            node_rank=int(row[4]) if row[4] is not None else None,
            hostname=str(row[5]) if row[5] is not None else None,
            cpu_avg_percent=float(row[6]) if row[6] is not None else None,
            cpu_peak_percent=float(row[7]) if row[7] is not None else None,
            cpu_logical_core_count=int(row[8]) if row[8] is not None else None,
            ram_avg_bytes=float(row[9]) if row[9] is not None else None,
            ram_peak_bytes=float(row[10]) if row[10] is not None else None,
            ram_total_bytes=float(row[11]) if row[11] is not None else None,
            gpu_available=bool(row[12]) if row[12] is not None else None,
            gpu_count=int(row[13]) if row[13] is not None else None,
            gpu_device_index=int(row[14]) if row[14] is not None else None,
            gpu_mem_used_avg_bytes=(
                float(row[15]) if row[15] is not None else None
            ),
            gpu_mem_used_peak_bytes=(
                float(row[16]) if row[16] is not None else None
            ),
            gpu_mem_reserved_avg_bytes=(
                float(row[17]) if row[17] is not None else None
            ),
            gpu_mem_reserved_peak_bytes=(
                float(row[18]) if row[18] is not None else None
            ),
            gpu_mem_total_bytes=(
                float(row[19]) if row[19] is not None else None
            ),
            gpu_mem_reserved_overhang_ratio=(
                float(row[20]) if row[20] is not None else None
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
        f"global ranks {agg.distinct_ranks}",
        _format_percent_stat("CPU avg", agg.cpu_avg_percent),
        _format_memory_stat("RSS peak", ram_peak_gb, ram_total_gb),
        _format_percent_stat(gpu_label, gpu_pct),
    ]
    rendered = [part for part in parts if part is not None]
    return " | ".join(rendered) if rendered else "unavailable"


def _per_rank_to_json(
    per_rank: Dict[int, PerRankProcessSummary],
) -> Dict[str, Dict[str, Any]]:
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
            "identity": {
                "global_rank": item.global_rank,
                "local_rank": item.local_rank,
                "node_rank": item.node_rank,
                "hostname": item.hostname,
                "local_world_size": item.local_world_size,
                "world_size": item.world_size,
            },
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


def _metric_rollup(
    per_rank: Dict[int, PerRankProcessSummary],
    value: Callable[[PerRankProcessSummary], Optional[float]],
    *,
    higher_is_worse: bool = True,
) -> Optional[Dict[str, Optional[float]]]:
    """
    Return median/worst/skew for one metric across global ranks.

    ``higher_is_worse`` is false for headroom-style metrics where the smallest
    value is the limiting case.
    """
    pairs: list[tuple[float, int]] = []
    for global_rank, item in per_rank.items():
        raw = value(item)
        if raw is None:
            continue
        pairs.append((float(raw), int(global_rank)))

    if not pairs:
        return None

    values = [metric for metric, _global_rank in pairs]
    worst_value, worst_global_rank = (
        max(pairs, key=lambda pair: pair[0])
        if higher_is_worse
        else min(pairs, key=lambda pair: pair[0])
    )
    median_value = float(median(values))

    skew_percent: Optional[float]
    if median_value == 0.0:
        skew_percent = None
    elif higher_is_worse:
        skew_percent = (
            (float(worst_value) - median_value) / abs(median_value) * 100.0
        )
    else:
        skew_percent = (
            (median_value - float(worst_value)) / abs(median_value) * 100.0
        )

    return {
        "median": median_value,
        "worst": float(worst_value),
        "worst_global_rank": int(worst_global_rank),
        "skew_percent": skew_percent,
    }


def _global_rank_rollup_to_json(
    per_rank: Dict[int, PerRankProcessSummary],
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Build comparison metrics across global ranks for Process summary JSON.
    """

    def headroom_gb(item: PerRankProcessSummary) -> Optional[float]:
        total_gb = bytes_to_gb(item.gpu_mem_total_bytes)
        reserved_gb = bytes_to_gb(item.gpu_mem_reserved_peak_bytes)
        if total_gb is None or reserved_gb is None:
            return None
        return max(total_gb - reserved_gb, 0.0)

    metrics = {
        "cpu_avg_percent": _metric_rollup(
            per_rank,
            lambda item: item.cpu_avg_percent,
        ),
        "ram_peak_gb": _metric_rollup(
            per_rank,
            lambda item: bytes_to_gb(item.ram_peak_bytes),
        ),
        "gpu_mem_used_peak_gb": _metric_rollup(
            per_rank,
            lambda item: bytes_to_gb(item.gpu_mem_used_peak_bytes),
        ),
        "gpu_mem_reserved_peak_gb": _metric_rollup(
            per_rank,
            lambda item: bytes_to_gb(item.gpu_mem_reserved_peak_bytes),
        ),
        "gpu_mem_headroom_gb": _metric_rollup(
            per_rank,
            headroom_gb,
            higher_is_worse=False,
        ),
    }
    return {
        name: rollup for name, rollup in metrics.items() if rollup is not None
    }
