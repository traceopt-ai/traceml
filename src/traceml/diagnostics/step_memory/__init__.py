"""
Public step-memory diagnostics package.

This package contains both live and summary-oriented step-memory diagnosis.
The public import path remains ``traceml.diagnostics.step_memory`` while the
implementation is split into:

- ``api`` for primary builders and diagnosis payloads
- ``adapters`` for renderer/summary input normalization
- ``rules`` for summary diagnostic rules
- ``trend`` for conservative memory-creep trend helpers
"""

from .api import (
    DEFAULT_STEP_MEMORY_THRESHOLDS,
    StepMemoryDiagnosis,
    StepMemoryDiagnosisKind,
    StepMemoryDiagnosisThresholds,
    build_step_memory_diagnosis,
    build_step_memory_summary_diagnosis_result,
)

__all__ = [
    "StepMemoryDiagnosisKind",
    "StepMemoryDiagnosisThresholds",
    "DEFAULT_STEP_MEMORY_THRESHOLDS",
    "StepMemoryDiagnosis",
    "build_step_memory_diagnosis",
    "build_step_memory_summary_diagnosis_result",
]
