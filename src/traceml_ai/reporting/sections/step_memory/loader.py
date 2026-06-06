# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""SQLite loader for the final-report step-memory section."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Optional

from traceml_ai.renderers.step_memory.schema import StepMemoryCombinedMetric
from traceml_ai.reporting.config import normalize_summary_window_rows
from traceml_ai.reporting.sections.step_memory.model import (
    MAX_SUMMARY_WINDOW_ROWS,
    StepMemoryAlignedWindow,
    StepMemoryGlobalRankIdentity,
    StepMemoryGlobalRankSummary,
    StepMemoryRankWindow,
    StepMemoryStepMetrics,
    build_combined_metrics_from_window,
    build_global_rank_summaries_from_window,
    metric_sort_key,
)
from traceml_ai.utils.step_windows import common_suffix_steps


@dataclass(frozen=True)
class StepMemorySectionData:
    """Loaded inputs for the step-memory final-report section."""

    training_steps: int
    latest_step_observed: Optional[int]
    metrics: list[StepMemoryCombinedMetric]
    gpu_total_bytes: Optional[float]
    no_gpu_detected: bool
    per_global_rank: Dict[str, StepMemoryGlobalRankSummary]
    aligned_window: StepMemoryAlignedWindow


@dataclass(frozen=True)
class _StepMemoryCandidateRow:
    """Latest deduplicated memory row for one `(global_rank, step)` pair."""

    identity: StepMemoryGlobalRankIdentity
    device: Optional[str]
    metrics: StepMemoryStepMetrics


def _load_latest_step_observed(conn: sqlite3.Connection) -> Optional[int]:
    """Return the latest observed memory step across all ranks."""
    row = conn.execute("SELECT MAX(step) FROM step_memory_samples;").fetchone()
    if not row or row[0] is None:
        return None
    return int(row[0])


def _load_gpu_total_bytes(conn: sqlite3.Connection) -> Optional[float]:
    """
    Best-effort device total memory from process telemetry.

    This is optional but useful for HIGH_PRESSURE diagnosis.
    """
    try:
        row = conn.execute(
            "SELECT MAX(gpu_mem_total_bytes) FROM process_samples;"
        ).fetchone()
    except Exception:
        return None

    if not row or row[0] is None:
        return None
    try:
        value = float(row[0])
    except Exception:
        return None
    return value if value > 0.0 else None


def _load_gpu_available(conn: sqlite3.Connection) -> Optional[bool]:
    """Best-effort check for whether the run reported GPU availability."""
    saw_signal = False
    for query in (
        "SELECT MAX(gpu_available) FROM process_samples;",
        "SELECT MAX(gpu_available) FROM system_samples;",
    ):
        try:
            row = conn.execute(query).fetchone()
        except Exception:
            continue
        if not row or row[0] is None:
            continue
        saw_signal = True
        try:
            if bool(int(row[0])):
                return True
        except Exception:
            if bool(row[0]):
                return True
    return False if saw_signal else None


def _load_global_ranks_seen(conn: sqlite3.Connection) -> int:
    """Return number of global ranks with step-memory rows."""
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT global_rank)
        FROM step_memory_samples
        WHERE global_rank IS NOT NULL
          AND step IS NOT NULL;
        """
    ).fetchone()
    return int(row[0] or 0) if row else 0


def _load_recent_candidate_rows(
    conn: sqlite3.Connection,
    *,
    max_candidate_steps_per_rank: int,
) -> Dict[int, Dict[int, _StepMemoryCandidateRow]]:
    """Load recent complete memory rows, deduped by `(global_rank, step)`."""
    rows = conn.execute(
        """
        WITH latest_per_step AS (
            SELECT
                global_rank,
                local_rank,
                node_rank,
                hostname,
                local_world_size,
                world_size,
                step,
                device,
                peak_alloc_bytes,
                peak_reserved_bytes,
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY global_rank, step
                    ORDER BY id DESC
                ) AS duplicate_row
            FROM step_memory_samples
            WHERE global_rank IS NOT NULL
              AND step IS NOT NULL
              AND peak_alloc_bytes IS NOT NULL
              AND peak_reserved_bytes IS NOT NULL
        ),
        recent_per_rank AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY global_rank
                    ORDER BY step DESC, id DESC
                ) AS recent_row
            FROM latest_per_step
            WHERE duplicate_row = 1
        )
        SELECT
            global_rank,
            local_rank,
            node_rank,
            hostname,
            local_world_size,
            world_size,
            step,
            device,
            peak_alloc_bytes,
            peak_reserved_bytes
        FROM recent_per_rank
        WHERE recent_row <= ?
        ORDER BY global_rank ASC, step ASC;
        """,
        (int(max_candidate_steps_per_rank),),
    ).fetchall()

    candidates: Dict[int, Dict[int, _StepMemoryCandidateRow]] = {}
    for row in rows:
        try:
            global_rank = int(row[0])
            step = int(row[6])
            peak_allocated = float(row[8])
            peak_reserved = float(row[9])
        except Exception:
            continue

        identity = StepMemoryGlobalRankIdentity(
            global_rank=global_rank,
            local_rank=int(row[1]) if row[1] is not None else None,
            node_rank=int(row[2]) if row[2] is not None else None,
            hostname=str(row[3]) if row[3] is not None else None,
            local_world_size=int(row[4]) if row[4] is not None else None,
            world_size=int(row[5]) if row[5] is not None else None,
        )
        candidates.setdefault(global_rank, {})[step] = _StepMemoryCandidateRow(
            identity=identity,
            device=str(row[7]) if row[7] else None,
            metrics=StepMemoryStepMetrics(
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
            ),
        )

    return candidates


def _build_aligned_window(
    conn: sqlite3.Connection,
    *,
    window_size: int,
) -> StepMemoryAlignedWindow:
    """Build the latest common step-memory window across global ranks."""
    global_ranks_seen = _load_global_ranks_seen(conn)
    candidate_limit = max(int(window_size) * 20, int(window_size) + 1)
    candidates = _load_recent_candidate_rows(
        conn,
        max_candidate_steps_per_rank=candidate_limit,
    )
    common_steps = common_suffix_steps(candidates, max_rows=window_size)
    if not common_steps:
        return StepMemoryAlignedWindow(
            steps=(),
            per_global_rank={},
            window_size=int(window_size),
            global_ranks_seen=global_ranks_seen,
        )

    per_global_rank: Dict[int, StepMemoryRankWindow] = {}
    for global_rank, step_rows in sorted(candidates.items()):
        aligned_rows = {
            int(step): step_rows[step]
            for step in common_steps
            if step in step_rows
        }
        if len(aligned_rows) != len(common_steps):
            continue

        latest_row = aligned_rows[max(aligned_rows.keys())]
        per_global_rank[int(global_rank)] = StepMemoryRankWindow(
            identity=latest_row.identity,
            device=latest_row.device,
            step_metrics={
                int(step): row.metrics
                for step, row in sorted(aligned_rows.items())
            },
        )

    return StepMemoryAlignedWindow(
        steps=tuple(int(step) for step in common_steps),
        per_global_rank=per_global_rank,
        window_size=int(window_size),
        global_ranks_seen=global_ranks_seen,
    )


def load_step_memory_section_data(
    db_path: str,
    *,
    window_size: int = MAX_SUMMARY_WINDOW_ROWS,
) -> StepMemorySectionData:
    """
    Load bounded step-memory section data from the SQLite history database.
    """
    bounded_window = normalize_summary_window_rows(window_size)
    conn = sqlite3.connect(db_path)

    try:
        latest_step_observed = _load_latest_step_observed(conn)
        training_steps = (
            latest_step_observed + 1 if latest_step_observed is not None else 0
        )

        gpu_total_bytes = _load_gpu_total_bytes(conn)
        gpu_available = _load_gpu_available(conn)

        aligned_window = _build_aligned_window(
            conn,
            window_size=bounded_window,
        )
        metrics = sorted(
            build_combined_metrics_from_window(aligned_window),
            key=metric_sort_key,
        )

        per_global_rank = build_global_rank_summaries_from_window(
            aligned_window
        )
    finally:
        conn.close()

    return StepMemorySectionData(
        training_steps=training_steps,
        latest_step_observed=latest_step_observed,
        metrics=metrics,
        gpu_total_bytes=gpu_total_bytes,
        no_gpu_detected=bool(gpu_available is False),
        per_global_rank=per_global_rank,
        aligned_window=aligned_window,
    )


__all__ = [
    "StepMemorySectionData",
    "load_step_memory_section_data",
]
