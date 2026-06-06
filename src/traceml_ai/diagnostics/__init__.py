"""
Central diagnostics package for shared diagnosis contracts and engines.
"""

from .common import (
    BaseDiagnosis,
    DiagnosticIssue,
    DiagnosticResult,
    DiagnosticRule,
    Severity,
)
from .model_diagnostics import (
    DEFAULT_MODEL_DIAGNOSTIC_REGISTRY,
    ModelDiagnosisItem,
    ModelDiagnosticsPayload,
    build_model_diagnostics_payload,
)
from .registry import (
    DiagnosticDomainRegistry,
    DiagnosticDomainSpec,
    ModelDiagnosticBuilder,
    ModelDiagnosticContext,
)

__all__ = [
    "Severity",
    "BaseDiagnosis",
    "DiagnosticIssue",
    "DiagnosticResult",
    "DiagnosticRule",
    "DiagnosticDomainRegistry",
    "DiagnosticDomainSpec",
    "ModelDiagnosticBuilder",
    "ModelDiagnosticContext",
    "DEFAULT_MODEL_DIAGNOSTIC_REGISTRY",
    "ModelDiagnosisItem",
    "ModelDiagnosticsPayload",
    "build_model_diagnostics_payload",
]
