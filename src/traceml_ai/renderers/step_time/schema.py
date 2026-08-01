"""Backward-compatible imports for Step Time domain contracts.

New code should import these types from :mod:`traceml_ai.step_time.model`.
This module remains temporarily so downstream integrations can migrate without
an abrupt import-path break.
"""

from traceml_ai.step_time.model import (
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeResult,
    StepCombinedTimeSeries,
    StepCombinedTimeSummary,
)

__all__ = [
    "StepCombinedTimeCoverage",
    "StepCombinedTimeMetric",
    "StepCombinedTimeResult",
    "StepCombinedTimeSeries",
    "StepCombinedTimeSummary",
]
