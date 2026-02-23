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


# -----------------------------
# New: dashboard rank heatmap data
# -----------------------------

@dataclass(frozen=True)
class StepCombinedRankRow:
    """
    One row in the rank heatmap table.

    All values are window sums over the last K fully-common steps.
    """
    rank: int
    sums_ms: Dict[str, float]  # metric_key -> window sum in ms


@dataclass(frozen=True)
class StepCombinedRankHeatmap:
    """
    Rank x Metric heatmap payload (dashboard-oriented).

    - rows are ranks
    - columns are metrics (metric_keys)
    - each cell is sum over last K fully-common steps

    Sorting:
    - rows are sorted by (step_time_ms desc, dataloader_fetch desc) by default
    """
    window_size: int
    steps_used: int
    metric_keys: List[str]
    rows: List[StepCombinedRankRow]
    sort_by: List[str] = field(default_factory=lambda: ["step_time_ms", "dataloader_fetch"])


@dataclass(frozen=True)
class StepCombinedTimeResult:
    metrics: List[StepCombinedTimeMetric]
    status_message: str = "OK"

    # Optional dashboard payload (safe for older consumers)
    rank_heatmap: Optional[StepCombinedRankHeatmap] = None