"""
Public step-time diagnostics package.

This package keeps the external import path stable as:

    traceml.diagnostics.step_time

while organizing the implementation into smaller modules:
- `api` for the public entry points and primary diagnosis dataclasses
- `context` for prepared analysis state
- `rules` for registered diagnosis rules
- `formatters` for CLI / dashboard presentation
"""

from .api import (
    ComputeSignal,
    DiagnosisKind,
    Severity,
    StepDiagnosis,
    build_step_diagnosis,
    build_step_diagnosis_result,
)
from .formatters import format_cli_diagnosis, format_dashboard_diagnosis
from .policy import (
    DEFAULT_THRESHOLDS,
    LIVE_STEP_TIME_POLICY,
    SUMMARY_STEP_TIME_POLICY,
    DiagnosisThresholds,
    StepTimeDiagnosisPolicy,
)

__all__ = [
    "Severity",
    "DiagnosisKind",
    "DiagnosisThresholds",
    "DEFAULT_THRESHOLDS",
    "LIVE_STEP_TIME_POLICY",
    "SUMMARY_STEP_TIME_POLICY",
    "StepTimeDiagnosisPolicy",
    "StepDiagnosis",
    "ComputeSignal",
    "build_step_diagnosis",
    "build_step_diagnosis_result",
    "format_cli_diagnosis",
    "format_dashboard_diagnosis",
]
