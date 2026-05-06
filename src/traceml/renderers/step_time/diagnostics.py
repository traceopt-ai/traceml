"""
Backward-compatible import shim for step-time diagnosis.

The implementation lives in `traceml.diagnostics.step_time` so it can be reused
by both live renderers and summary generation without layering violations.
"""

from traceml.diagnostics.step_time import (
    DEFAULT_THRESHOLDS,
    LIVE_STEP_TIME_POLICY,
    SUMMARY_STEP_TIME_POLICY,
    ComputeSignal,
    DiagnosisKind,
    DiagnosisThresholds,
    Severity,
    StepDiagnosis,
    StepTimeDiagnosisPolicy,
    build_step_diagnosis,
    build_step_diagnosis_result,
    format_cli_diagnosis,
    format_dashboard_diagnosis,
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
