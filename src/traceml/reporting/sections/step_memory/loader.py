# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""SQLite loader for the final-report step-memory section."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

from traceml.diagnostics.step_memory import (
    SUMMARY_STEP_MEMORY_POLICY,
    StepMemoryDiagnosis,
    build_step_memory_diagnosis,
    build_step_memory_summary_diagnosis_result,
)
from traceml.renderers.step_memory.common import (
    StepMemoryMetricsDB,
    build_step_memory_combined_result,
)
from traceml.renderers.step_memory.schema import StepMemoryCombinedMetric
from traceml.reporting.config import normalize_summary_window_rows
from traceml.reporting.sections.step_memory.model import (
    MAX_SUMMARY_WINDOW_ROWS,
    StepMemoryGlobalRankIdentity,
    StepMemoryGlobalRankSummary,
    metric_output_name,
    metric_sort_key,
)


@dataclass(frozen=True)
class StepMemorySectionData:
    """Loaded inputs for the step-memory final-report section."""

    training_steps: int
    latest_step_observed: Optional[int]
    metrics: list[StepMemoryCombinedMetric]
    diagnosis: Optional[StepMemoryDiagnosis]
    diagnosis_result: Optional[Any]
    no_gpu_detected: bool
    per_global_rank: Dict[str, StepMemoryGlobalRankSummary]


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


def _load_global_rank_identities(
    conn: sqlite3.Connection,
    global_ranks: list[int],
) -> Dict[int, StepMemoryGlobalRankIdentity]:
    """Load latest runtime identity metadata for step-memory ranks."""
    identities: Dict[int, StepMemoryGlobalRankIdentity] = {}
    for global_rank in global_ranks:
        row = conn.execute(
            """
            SELECT local_rank, node_rank, hostname, local_world_size,
                   world_size
            FROM step_memory_samples
            WHERE global_rank = ?
            ORDER BY sample_ts_s DESC, id DESC
            LIMIT 1;
            """,
            (int(global_rank),),
        ).fetchone()
        if row is None:
            continue
        identities[int(global_rank)] = StepMemoryGlobalRankIdentity(
            global_rank=int(global_rank),
            local_rank=int(row[0]) if row[0] is not None else None,
            node_rank=int(row[1]) if row[1] is not None else None,
            hostname=str(row[2]) if row[2] is not None else None,
            local_world_size=int(row[3]) if row[3] is not None else None,
            world_size=int(row[4]) if row[4] is not None else None,
        )
    return identities


def _default_identity(global_rank: int) -> StepMemoryGlobalRankIdentity:
    """Create a minimal identity when older rows lack runtime metadata."""
    return StepMemoryGlobalRankIdentity(
        global_rank=int(global_rank),
        local_rank=None,
        node_rank=None,
        hostname=None,
        local_world_size=None,
        world_size=None,
    )


def _load_per_global_rank_summaries(
    conn: sqlite3.Connection,
    *,
    db: StepMemoryMetricsDB,
    metrics: list[StepMemoryCombinedMetric],
    window_size: int,
) -> Dict[str, StepMemoryGlobalRankSummary]:
    """
    Load per-rank metrics for the same aligned steps used globally.

    The combined metrics decide the common completed-step window. This loader
    reads the per-rank values for those exact steps, so row metrics and
    global.window describe the same data.
    """
    latest_per_global_rank = db.fetch_latest_step_per_global_rank(conn)
    if not latest_per_global_rank:
        return {}

    scan_span = max(int(window_size) * 20, int(window_size) + 1)
    identities = _load_global_rank_identities(
        conn,
        sorted(latest_per_global_rank.keys()),
    )
    per_global_rank: Dict[str, StepMemoryGlobalRankSummary] = {}
    metric_values: Dict[str, Dict[str, float]] = {}

    for metric in metrics:
        common_steps = [int(step) for step in metric.series.steps]
        if not common_steps:
            continue

        rank_maps, _ = db.fetch_global_rank_step_maps(
            conn,
            metric_key=metric.metric,
            start_step=min(common_steps),
            end_step=max(common_steps),
            max_unique_steps_per_rank=scan_span,
        )
        if not rank_maps:
            continue

        public_metric_name = metric_output_name(metric.metric)
        for rank in sorted(rank_maps.keys()):
            step_map = rank_maps.get(rank, {})
            values = [
                float(step_map[step])
                for step in common_steps
                if step in step_map
            ]
            if len(values) != len(common_steps) or not values:
                continue

            rank_key = str(rank)
            metric_values.setdefault(rank_key, {})[public_metric_name] = sum(
                values
            ) / len(values)

    for rank_key, metrics_by_name in sorted(
        metric_values.items(),
        key=lambda item: int(item[0]),
    ):
        global_rank = int(rank_key)
        per_global_rank[rank_key] = StepMemoryGlobalRankSummary(
            identity=identities.get(
                global_rank, _default_identity(global_rank)
            ),
            metrics=metrics_by_name,
        )

    return per_global_rank


def _per_rank_for_diagnostics(
    per_global_rank: Dict[str, StepMemoryGlobalRankSummary],
) -> Dict[str, Any]:
    """Expose typed rank summaries to diagnostics without leaking JSON rows."""
    return {
        rank_key: {
            "metrics": dict(summary.metrics),
            "identity": {
                "global_rank": summary.identity.global_rank,
                "local_rank": summary.identity.local_rank,
                "node_rank": summary.identity.node_rank,
                "hostname": summary.identity.hostname,
                "local_world_size": summary.identity.local_world_size,
                "world_size": summary.identity.world_size,
            },
        }
        for rank_key, summary in per_global_rank.items()
    }


def load_step_memory_section_data(
    db_path: str,
    *,
    window_size: int = MAX_SUMMARY_WINDOW_ROWS,
) -> StepMemorySectionData:
    """
    Load bounded step-memory section data from the SQLite history database.
    """
    bounded_window = normalize_summary_window_rows(window_size)
    db = StepMemoryMetricsDB(db_path=db_path)
    conn = db.connect()

    try:
        latest_step_observed = _load_latest_step_observed(conn)
        training_steps = (
            latest_step_observed + 1 if latest_step_observed is not None else 0
        )

        gpu_total_bytes = _load_gpu_total_bytes(conn)
        gpu_available = db.detect_gpu_available(conn)

        result = build_step_memory_combined_result(
            conn,
            db=db,
            window_size=bounded_window,
        )
        metrics = sorted(result.metrics, key=metric_sort_key)

        diagnosis = None
        if metrics:
            diagnosis = build_step_memory_diagnosis(
                metrics,
                gpu_total_bytes=gpu_total_bytes,
                thresholds=SUMMARY_STEP_MEMORY_POLICY.thresholds,
            )

        per_global_rank = _load_per_global_rank_summaries(
            conn,
            db=db,
            metrics=metrics,
            window_size=bounded_window,
        )

        diagnosis_result = None
        if metrics:
            diagnosis_result = build_step_memory_summary_diagnosis_result(
                metrics,
                gpu_total_bytes=gpu_total_bytes,
                per_rank=_per_rank_for_diagnostics(per_global_rank),
                thresholds=SUMMARY_STEP_MEMORY_POLICY.thresholds,
            )
    finally:
        conn.close()

    return StepMemorySectionData(
        training_steps=training_steps,
        latest_step_observed=latest_step_observed,
        metrics=metrics,
        diagnosis=diagnosis,
        diagnosis_result=diagnosis_result,
        no_gpu_detected=bool(
            gpu_available is False and latest_step_observed is not None
        ),
        per_global_rank=per_global_rank,
    )


__all__ = [
    "StepMemorySectionData",
    "load_step_memory_section_data",
]
