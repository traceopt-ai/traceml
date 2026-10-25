# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Read and reduce traced-process measurements."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Optional

from traceml_ai.measurements.window import AnalysisWindow


@dataclass
class ProcessSummaryAgg:
    """Process measurements aggregated across the selected ranks."""

    # Sample timestamps are Unix seconds from the traced process.
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    process_samples: int = 0
    distinct_global_ranks: int = 0

    cpu_avg_percent: Optional[float] = None
    cpu_peak_percent: Optional[float] = None
    cpu_logical_core_count: Optional[int] = None

    ram_avg_bytes: Optional[float] = None
    ram_peak_bytes: Optional[float] = None
    ram_total_bytes: Optional[float] = None

    gpu_available: Optional[bool] = None
    gpu_count: Optional[int] = None
    gpu_mem_used_avg_bytes: Optional[float] = None
    gpu_mem_used_peak_bytes: Optional[float] = None
    gpu_mem_reserved_avg_bytes: Optional[float] = None
    gpu_mem_reserved_peak_bytes: Optional[float] = None
    gpu_mem_total_bytes: Optional[float] = None


@dataclass
class PerRankProcessSummary:
    """Process measurements aggregated for one global rank."""

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
    gpu_mem_used_avg_bytes: Optional[float] = None
    gpu_mem_used_peak_bytes: Optional[float] = None
    gpu_mem_reserved_avg_bytes: Optional[float] = None
    gpu_mem_reserved_peak_bytes: Optional[float] = None
    gpu_mem_total_bytes: Optional[float] = None
    # Max CUDA allocator reserved/allocated ratio in the selected window.
    gpu_mem_reserved_overhang_ratio: Optional[float] = None


@dataclass(frozen=True)
class ProcessMeasurements:
    """Selected Process aggregates used by reporting and diagnosis."""

    aggregate: ProcessSummaryAgg
    per_global_rank: Dict[int, PerRankProcessSummary]


def cpu_capacity_percent(
    cpu_avg_percent: Optional[float],
    cpu_logical_core_count: Optional[int],
) -> Optional[float]:
    """Return average CPU usage normalized by logical host core capacity."""
    if (
        cpu_avg_percent is None
        or cpu_logical_core_count is None
        or cpu_logical_core_count <= 0
    ):
        return None
    return max(
        0.0,
        float(cpu_avg_percent)
        / (100.0 * float(cpu_logical_core_count))
        * 100.0,
    )


def _process_samples_cte(
    analysis_window: Optional[AnalysisWindow],
) -> tuple[str, tuple[int, ...]]:
    params: tuple[int, ...] = ()
    time_filter = ""
    if (
        analysis_window is not None
        and analysis_window.start_time_unix_ns is not None
        and analysis_window.end_time_unix_ns is not None
    ):
        time_filter = "AND timestamp_unix_ns BETWEEN ? AND ?"
        params = (
            analysis_window.start_time_unix_ns,
            analysis_window.end_time_unix_ns,
        )
    return (
        f"""
        WITH selected_process_samples AS (
            SELECT p.*
            FROM process_samples AS p
            WHERE p.global_rank IS NOT NULL
              {time_filter}
        )
    """,
        params,
    )


def _load_process_summary_aggregate(
    connection: sqlite3.Connection,
    *,
    analysis_window: Optional[AnalysisWindow],
) -> ProcessSummaryAgg:
    sample_cte, params = _process_samples_cte(analysis_window)
    count_row = connection.execute(
        sample_cte + """
        SELECT
            COUNT(*),
            MIN(timestamp_unix_ns),
            MAX(timestamp_unix_ns),
            COUNT(DISTINCT global_rank)
        FROM selected_process_samples;
        """,
        params,
    ).fetchone()

    n_rows = int(count_row[0] or 0)
    first_ts = (
        count_row[1] / 1_000_000_000 if count_row[1] is not None else None
    )
    last_ts = (
        count_row[2] / 1_000_000_000 if count_row[2] is not None else None
    )
    distinct_global_ranks = int(count_row[3] or 0)

    row = connection.execute(
        sample_cte + """
        SELECT
            AVG(cpu_percent),
            MAX(cpu_percent),
            MAX(cpu_logical_core_count),

            AVG(ram_used_bytes),
            MAX(ram_used_bytes),
            MAX(ram_total_bytes),

            MAX(gpu_available),
            MAX(gpu_count),

            AVG(gpu_mem_used_bytes),
            MAX(gpu_mem_used_bytes),
            AVG(gpu_mem_reserved_bytes),
            MAX(gpu_mem_reserved_bytes),
            MAX(gpu_mem_total_bytes)
        FROM selected_process_samples;
        """,
        params,
    ).fetchone()

    return ProcessSummaryAgg(
        first_ts=first_ts,
        last_ts=last_ts,
        process_samples=n_rows,
        distinct_global_ranks=distinct_global_ranks,
        cpu_avg_percent=float(row[0]) if row[0] is not None else None,
        cpu_peak_percent=float(row[1]) if row[1] is not None else None,
        cpu_logical_core_count=int(row[2]) if row[2] is not None else None,
        ram_avg_bytes=float(row[3]) if row[3] is not None else None,
        ram_peak_bytes=float(row[4]) if row[4] is not None else None,
        ram_total_bytes=float(row[5]) if row[5] is not None else None,
        gpu_available=bool(row[6]) if row[6] is not None else None,
        gpu_count=int(row[7]) if row[7] is not None else None,
        gpu_mem_used_avg_bytes=float(row[8]) if row[8] is not None else None,
        gpu_mem_used_peak_bytes=(
            float(row[9]) if row[9] is not None else None
        ),
        gpu_mem_reserved_avg_bytes=(
            float(row[10]) if row[10] is not None else None
        ),
        gpu_mem_reserved_peak_bytes=(
            float(row[11]) if row[11] is not None else None
        ),
        gpu_mem_total_bytes=float(row[12]) if row[12] is not None else None,
    )


def _load_per_global_rank_process_summary(
    connection: sqlite3.Connection,
    *,
    analysis_window: Optional[AnalysisWindow],
) -> Dict[int, PerRankProcessSummary]:
    sample_cte, params = _process_samples_cte(analysis_window)
    rows = connection.execute(
        sample_cte + """
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

        FROM selected_process_samples
        GROUP BY global_rank
        ORDER BY global_rank ASC;
        """,
        params,
    ).fetchall()

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
            gpu_mem_used_avg_bytes=(
                float(row[14]) if row[14] is not None else None
            ),
            gpu_mem_used_peak_bytes=(
                float(row[15]) if row[15] is not None else None
            ),
            gpu_mem_reserved_avg_bytes=(
                float(row[16]) if row[16] is not None else None
            ),
            gpu_mem_reserved_peak_bytes=(
                float(row[17]) if row[17] is not None else None
            ),
            gpu_mem_total_bytes=(
                float(row[18]) if row[18] is not None else None
            ),
            gpu_mem_reserved_overhang_ratio=(
                float(row[19]) if row[19] is not None else None
            ),
        )
    return out


def read_process_measurements(
    connection: sqlite3.Connection,
    *,
    analysis_window: Optional[AnalysisWindow] = None,
) -> ProcessMeasurements:
    """Read selected Process measurements from an existing connection."""
    return ProcessMeasurements(
        aggregate=_load_process_summary_aggregate(
            connection,
            analysis_window=analysis_window,
        ),
        per_global_rank=_load_per_global_rank_process_summary(
            connection,
            analysis_window=analysis_window,
        ),
    )


__all__ = [
    "PerRankProcessSummary",
    "ProcessMeasurements",
    "ProcessSummaryAgg",
    "cpu_capacity_percent",
    "read_process_measurements",
]
