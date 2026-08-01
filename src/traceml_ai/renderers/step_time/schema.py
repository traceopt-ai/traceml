from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from traceml_ai.utils.step_time_window import StepTimeWindow


@dataclass(frozen=True)
class StepCombinedTimeSeries:
    steps: List[int]
    median: List[float]
    worst: List[float]
    sum: List[float]


@dataclass(frozen=True)
class StepCombinedTimeSummary:
    window_size: int
    steps_used: int
    median_total: float
    worst_total: float
    worst_rank: Optional[int]
    skew_ratio: Optional[float]
    skew_pct: Optional[float]


@dataclass(frozen=True)
class StepCombinedTimeCoverage:
    expected_steps: int
    steps_used: int
    completed_step: int
    world_size: int
    ranks_present: int
    incomplete: bool


@dataclass(frozen=True)
class StepCombinedTimeMetric:
    metric: str
    clock: str  # "cpu" | "gpu" | "mixed"
    series: Optional[StepCombinedTimeSeries]
    summary: StepCombinedTimeSummary
    coverage: StepCombinedTimeCoverage


@dataclass(frozen=True)
class StepCombinedTimeResult:
    """Live Step Time state wrapping one canonical analyzed window."""

    status_message: str = "OK"
    window: Optional[StepTimeWindow] = None
    training_strategy: str = "ddp"
    had_ok: bool = False
