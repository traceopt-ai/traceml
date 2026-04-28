"""
Central diagnostics package for shared diagnosis contracts and engines.
"""

from .common import (
    BaseDiagnosis,
    DiagnosticIssue,
    DiagnosticResult,
    DiagnosticRule,
    Severity,
    diagnosis_to_dict,
)
from .model_diagnostics import (
    ModelDiagnosisItem,
    ModelDiagnosticsPayload,
    build_model_diagnostics_payload,
)

__all__ = [
    "Severity",
    "BaseDiagnosis",
    "DiagnosticIssue",
    "DiagnosticResult",
    "DiagnosticRule",
    "diagnosis_to_dict",
    "ModelDiagnosisItem",
    "ModelDiagnosticsPayload",
    "build_model_diagnostics_payload",
]
