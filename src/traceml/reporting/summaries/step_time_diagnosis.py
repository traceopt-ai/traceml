"""Compatibility exports for step-time summary diagnosis helpers."""

from traceml.reporting.sections.step_time.diagnosis import (
    RankStepSignals,
    build_summary_step_diagnosis,
    build_summary_step_diagnosis_result,
)

__all__ = [
    "RankStepSignals",
    "build_summary_step_diagnosis",
    "build_summary_step_diagnosis_result",
]
