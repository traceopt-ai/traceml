"""
Renderer-facing schema for step combined.

This schema represents a *capacity / tail-latency oriented* view of
step-level execution time aggregated across DDP ranks.

Design goals
------------
- Renderer/UI consumes ONLY these dataclasses
- No mixed-clock ambiguity (CPU vs GPU is explicit)
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class StepCombinedTimeSeries:
    """
    Per-metric time series (renderer-facing).

    Semantics
    ---------
    - Values are aggregated across ranks per step
    - `median` = typical rank
    - `worst`  = gating / tail rank
    - `sum`    = sum across ranks (cluster-wide proxy)
    """

    steps: List[int]

    median: List[float]
    worst: List[float]
    sum: List[float]


@dataclass(frozen=True)
class StepCombinedTimeSummary:
    """
    Window-level summary over last N complete steps.
    """

    window_size: int
    steps_used: int

    median_total: float
    worst_total: float

    worst_rank: Optional[int]

    skew_ratio: float
    skew_pct: float


@dataclass(frozen=True)
class StepCombinedTimeCoverage:
    """
    Coverage / health information for this metric.
    """

    expected_steps: int
    steps_used: int
    completed_step: Optional[int]

    world_size: int
    ranks_present: int

    incomplete: bool


@dataclass(frozen=True)
class StepCombinedTimeMetric:
    """
    Renderer-ready payload for ONE step combined time metric.

    Example metrics:
    - dataloader_fetch_cpu_ms
    - forward_gpu_ms
    - backward_cpu_ms
    - optimizer_gpu_ms
    """

    metric: str            # metric key / name
    clock: str             # "cpu" or "gpu"

    series: StepCombinedTimeSeries
    summary: StepCombinedTimeSummary
    coverage: StepCombinedTimeCoverage


@dataclass(frozen=True)
class StepCombinedTimeResult:
    """
    Final renderer-facing payload.

    This is the ONLY object UI layers should consume.
    """

    metrics: List[StepCombinedTimeMetric]
    status_message: str
