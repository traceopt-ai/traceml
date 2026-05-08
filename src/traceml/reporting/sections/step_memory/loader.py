"""SQLite loader for the final-report step-memory section."""

from __future__ import annotations

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
from traceml.reporting.sections.step_memory.model import (
    MAX_SUMMARY_WINDOW_ROWS,
    _gpu_total_bytes,
    _latest_step_observed,
    _load_per_rank_summary,
    _metric_sort_key,
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
    per_rank: Dict[str, Any]


def load_step_memory_section_data(
    db_path: str,
    *,
    window_size: int = 400,
) -> StepMemorySectionData:
    """
    Load bounded step-memory section data from the SQLite history database.
    """
    bounded_window = min(max(1, int(window_size)), MAX_SUMMARY_WINDOW_ROWS)
    db = StepMemoryMetricsDB(db_path=db_path)
    conn = db.connect()

    try:
        latest_step_observed = _latest_step_observed(conn)
        training_steps = (
            latest_step_observed + 1 if latest_step_observed is not None else 0
        )

        gpu_total_bytes = _gpu_total_bytes(conn)
        gpu_available = db.detect_gpu_available(conn)

        result = build_step_memory_combined_result(
            conn,
            db=db,
            window_size=bounded_window,
        )
        metrics = sorted(result.metrics, key=_metric_sort_key)

        diagnosis = None
        if metrics:
            diagnosis = build_step_memory_diagnosis(
                metrics,
                gpu_total_bytes=gpu_total_bytes,
                thresholds=SUMMARY_STEP_MEMORY_POLICY.thresholds,
            )

        per_rank = _load_per_rank_summary(
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
                per_rank=per_rank,
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
        per_rank=per_rank,
    )


__all__ = [
    "StepMemorySectionData",
    "load_step_memory_section_data",
]
