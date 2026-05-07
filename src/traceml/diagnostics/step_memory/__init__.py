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
    StepMemoryDiagnosis,
    StepMemoryDiagnosisKind,
    build_step_memory_diagnosis,
    build_step_memory_summary_diagnosis_result,
)
from .policy import (
    DEFAULT_STEP_MEMORY_THRESHOLDS,
    LIVE_STEP_MEMORY_POLICY,
    SUMMARY_STEP_MEMORY_POLICY,
    StepMemoryDiagnosisPolicy,
    StepMemoryDiagnosisThresholds,
)

__all__ = [
    "StepMemoryDiagnosisKind",
    "StepMemoryDiagnosisPolicy",
    "StepMemoryDiagnosisThresholds",
    "LIVE_STEP_MEMORY_POLICY",
    "SUMMARY_STEP_MEMORY_POLICY",
    "DEFAULT_STEP_MEMORY_THRESHOLDS",
    "StepMemoryDiagnosis",
    "build_step_memory_diagnosis",
    "build_step_memory_summary_diagnosis_result",
]
