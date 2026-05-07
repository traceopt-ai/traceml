"""
Public system diagnostics package.

The current system diagnostics are summary-oriented and conservative. They are
intended primarily for end-of-run interpretation and machine-readable summary
payloads rather than live runtime UI changes.
"""

from .api import SystemDiagnosis, build_system_diagnosis_result
from .policy import DEFAULT_SYSTEM_POLICY, SystemDiagnosisPolicy

__all__ = [
    "DEFAULT_SYSTEM_POLICY",
    "SystemDiagnosisPolicy",
    "SystemDiagnosis",
    "build_system_diagnosis_result",
]
