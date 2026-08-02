"""Backward-compatible imports for Step Time domain contracts.

New code should import these types from :mod:`traceml_ai.step_time.model`.
This module remains temporarily so downstream integrations can migrate without
an abrupt import-path break.
"""

from traceml_ai.step_time.model import (
    StepTimeCoverage as StepCombinedTimeCoverage,
)
from traceml_ai.step_time.model import StepTimeMetric as StepCombinedTimeMetric
from traceml_ai.step_time.model import StepTimeResult as StepCombinedTimeResult
from traceml_ai.step_time.model import StepTimeSeries as StepCombinedTimeSeries
from traceml_ai.step_time.model import (
    StepTimeSummary as StepCombinedTimeSummary,
)

__all__ = [
    "StepCombinedTimeCoverage",
    "StepCombinedTimeMetric",
    "StepCombinedTimeResult",
    "StepCombinedTimeSeries",
    "StepCombinedTimeSummary",
]
