"""
Public process diagnostics package.

The current process diagnostics are summary-oriented and conservative. They are
intended primarily for end-of-run interpretation and machine-readable summary
payloads rather than live runtime UI changes.
"""

from .api import ProcessDiagnosis, build_process_diagnosis_result

__all__ = [
    "ProcessDiagnosis",
    "build_process_diagnosis_result",
]
