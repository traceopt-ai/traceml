"""
Step-memory diagnostic context aliases.

Live diagnosis context currently remains inside ``step_memory.api`` to avoid a
large behavioral rewrite. Summary-oriented normalized contexts live in
``step_memory.adapters`` because they adapt renderer payloads into rule inputs.
This module gives contributors one discoverable place to import context types
while later migrations continue to split live context more deeply.
"""

from .adapters import (
    StepMemorySummaryMetricSignals,
    StepMemorySummaryTrendSignals,
    build_step_memory_summary_signals,
)
from .api import MetricAssessment, WindowCreepEvidence

__all__ = [
    "WindowCreepEvidence",
    "MetricAssessment",
    "StepMemorySummaryTrendSignals",
    "StepMemorySummaryMetricSignals",
    "build_step_memory_summary_signals",
]
