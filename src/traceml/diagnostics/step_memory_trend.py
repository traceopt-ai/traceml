"""
Compatibility shim for step-memory trend helpers.

New code should import from ``traceml.diagnostics.step_memory.trend``.
"""

from traceml.diagnostics.step_memory.trend import (
    DEFAULT_STEP_MEMORY_TREND_HEURISTICS,
    StepMemoryTrendEvidence,
    StepMemoryTrendHeuristics,
    evaluate_step_memory_creep,
)

__all__ = [
    "StepMemoryTrendHeuristics",
    "DEFAULT_STEP_MEMORY_TREND_HEURISTICS",
    "StepMemoryTrendEvidence",
    "evaluate_step_memory_creep",
]
