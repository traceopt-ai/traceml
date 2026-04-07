"""
Backward-compatible import shim for step-memory diagnosis.
"""

from traceml.diagnostics.step_memory import (
    DEFAULT_STEP_MEMORY_THRESHOLDS,
    StepMemoryDiagnosis,
    StepMemoryDiagnosisKind,
    StepMemoryDiagnosisThresholds,
    build_step_memory_diagnosis,
)
from traceml.diagnostics.step_memory_formatters import (
    format_cli_diagnosis,
    format_dashboard_diagnosis,
)

__all__ = [
    "StepMemoryDiagnosisKind",
    "StepMemoryDiagnosisThresholds",
    "DEFAULT_STEP_MEMORY_THRESHOLDS",
    "StepMemoryDiagnosis",
    "build_step_memory_diagnosis",
    "format_cli_diagnosis",
    "format_dashboard_diagnosis",
]
