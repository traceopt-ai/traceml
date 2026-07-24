from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
    skew_ratio: float
    skew_pct: float


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
    status_message: str = "OK"
    per_rank_timing: Dict[int, Dict[str, float]] = field(default_factory=dict)
    diagnosis_clock: str = "cpu"
    training_strategy: str = "ddp"
    diagnosis_metrics: List[StepCombinedTimeMetric] = field(
        default_factory=list
    )
