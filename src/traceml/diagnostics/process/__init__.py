"""
Public process diagnostics package.

The current process diagnostics are summary-oriented and conservative. They are
intended primarily for end-of-run interpretation and machine-readable summary
payloads rather than live runtime UI changes.
"""

from .api import ProcessDiagnosis, build_process_diagnosis_result
from .policy import DEFAULT_PROCESS_POLICY, ProcessDiagnosisPolicy

__all__ = [
    "DEFAULT_PROCESS_POLICY",
    "ProcessDiagnosisPolicy",
    "ProcessDiagnosis",
    "build_process_diagnosis_result",
]
