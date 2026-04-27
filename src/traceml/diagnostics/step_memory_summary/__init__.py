"""
Public summary-oriented step-memory diagnostics package.

These diagnostics are intended for end-of-run summary interpretation and
machine-readable summary payloads. Live step-memory diagnosis continues to use
the existing runtime-focused module.
"""

from .api import build_step_memory_summary_diagnosis_result

__all__ = [
    "build_step_memory_summary_diagnosis_result",
]
