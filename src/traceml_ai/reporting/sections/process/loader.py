# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""SQLite loader for the final-report process section."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict

from traceml_ai.reporting.config import normalize_summary_window_rows
from traceml_ai.reporting.sections.process.model import (
    MAX_SUMMARY_ROWS,
    PerRankProcessSummary,
    ProcessSummaryAgg,
)


@dataclass(frozen=True)
class ProcessSectionData:
    """Loaded inputs for the process final-report section."""

    aggregate: ProcessSummaryAgg
    per_global_rank: Dict[int, PerRankProcessSummary]


def _recent_process_samples_cte() -> str:
    """
    Return the CTE used for the process summary window.

    Process summaries keep the latest N samples per global rank. Rows without a
    resolved global rank are ignored so aggregate and per-rank views agree.
    """
    return """
        WITH recent_process_samples AS (
            SELECT *
            FROM (
                SELECT
                    p.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.global_rank
                        ORDER BY p.id DESC
                    ) AS row_num
                FROM process_samples AS p
                WHERE p.global_rank IS NOT NULL
            )
            WHERE row_num <= ?
        )
    """


def load_process_summary_aggregate(
    conn: sqlite3.Connection,
    *,
    max_process_rows: int = MAX_SUMMARY_ROWS,
) -> ProcessSummaryAgg:
    """Load aggregate process metrics from `process_samples`."""
    sample_cte = _recent_process_samples_cte()
    params = (int(max_process_rows),)

    # Read window size, time bounds, and how many ranks are represented.
    count_row = conn.execute(
        sample_cte
        + """
        SELECT
            COUNT(*),
            MIN(sample_ts_s),
            MAX(sample_ts_s),
            COUNT(DISTINCT global_rank)
        FROM recent_process_samples;
        """,
        params,
    ).fetchone()

    n_rows = int(count_row[0] or 0)
    # Convert nullable SQLite scalars into typed summary fields.
    first_ts = float(count_row[1]) if count_row[1] is not None else None
    last_ts = float(count_row[2]) if count_row[2] is not None else None
    distinct_global_ranks = int(count_row[3] or 0)

    # Compute aggregate CPU, RAM, and GPU process metrics over the same window.
    row = conn.execute(
        sample_cte
        + """
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
        FROM recent_process_samples;
        """,
        params,
    ).fetchone()

    # Map the SQL aggregate row into the typed process summary model.
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


def load_per_global_rank_process_summary(
    conn: sqlite3.Connection,
    *,
    max_process_rows: int = MAX_SUMMARY_ROWS,
) -> Dict[int, PerRankProcessSummary]:
    """Load per-global-rank process metrics from `process_samples`."""
    sample_cte = _recent_process_samples_cte()
    params = (int(max_process_rows),)
    sql = (
        sample_cte
        + """
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

        FROM recent_process_samples
        GROUP BY global_rank
        ORDER BY global_rank ASC;
    """
    )

    # Build one typed process summary row for each observed global rank.
    rows = conn.execute(sql, params).fetchall()

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


def load_process_section_data(
    db_path: str,
    *,
    max_process_rows: int = MAX_SUMMARY_ROWS,
) -> ProcessSectionData:
    """
    Load bounded process-section data from the SQLite history database.
    """
    row_limit = normalize_summary_window_rows(max_process_rows)
    conn = sqlite3.connect(db_path)
    try:
        aggregate = load_process_summary_aggregate(
            conn,
            max_process_rows=row_limit,
        )
        per_global_rank = load_per_global_rank_process_summary(
            conn,
            max_process_rows=row_limit,
        )
    finally:
        conn.close()

    return ProcessSectionData(
        aggregate=aggregate,
        per_global_rank=per_global_rank,
    )


__all__ = [
    "ProcessSectionData",
    "load_per_global_rank_process_summary",
    "load_process_summary_aggregate",
    "load_process_section_data",
]
