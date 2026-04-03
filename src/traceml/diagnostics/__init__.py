"""
Central diagnostics package for shared diagnosis contracts and engines.
"""

from .common import BaseDiagnosis, Severity, diagnosis_to_dict
from .model_diagnostics import (
    ModelDiagnosisItem,
    ModelDiagnosticsPayload,
    build_model_diagnostics_payload,
)

__all__ = [
    "Severity",
    "BaseDiagnosis",
    "diagnosis_to_dict",
    "ModelDiagnosisItem",
    "ModelDiagnosticsPayload",
    "build_model_diagnostics_payload",
]
