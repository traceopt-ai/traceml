"""
Compatibility shim for summary-oriented step-memory diagnostics.

New code should import from ``traceml.diagnostics.step_memory``. This module is
kept as a thin internal bridge while reporting imports are migrated in slices.
"""

from traceml.diagnostics.step_memory import (
    build_step_memory_summary_diagnosis_result,
)

__all__ = [
    "build_step_memory_summary_diagnosis_result",
]
