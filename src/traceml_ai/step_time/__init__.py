"""Shared domain contracts for the Step Time telemetry pipeline.

Import model types from this package or from :mod:`traceml_ai.step_time.model`.
Data loading, diagnosis, and presentation remain in their respective layers.
"""

from .model import (
    DIAGNOSIS_CLOCK_KEY,
    STEP_TIME_EVENT_NAMES,
    DiagnosisClock,
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeResult,
    StepCombinedTimeSeries,
    StepCombinedTimeSummary,
    StepTimeClockValues,
    StepTimeLoadRequest,
    StepTimeRankIdentity,
    StepTimeRepositorySnapshot,
    StepTimeSourceCursor,
    StepTimeSourceRow,
    StepTimeWindow,
)

__all__ = [
    "DIAGNOSIS_CLOCK_KEY",
    "DiagnosisClock",
    "STEP_TIME_EVENT_NAMES",
    "StepCombinedTimeCoverage",
    "StepCombinedTimeMetric",
    "StepCombinedTimeResult",
    "StepCombinedTimeSeries",
    "StepCombinedTimeSummary",
    "StepTimeClockValues",
    "StepTimeLoadRequest",
    "StepTimeRankIdentity",
    "StepTimeRepositorySnapshot",
    "StepTimeSourceCursor",
    "StepTimeSourceRow",
    "StepTimeWindow",
]
