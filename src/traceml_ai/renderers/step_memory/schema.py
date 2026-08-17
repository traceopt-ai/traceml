"""Renderer-facing schema for combined step memory."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class StepMemoryCombinedSeries:
    """Per-step aggregated memory time series."""

    steps: List[int]
    median: List[float]
    worst: List[float]


@dataclass(frozen=True)
class StepMemoryCombinedSummary:
    """Window-level summary over the latest complete steps."""

    window_size: int
    steps_used: int

    median_peak: float
    worst_peak: float
    worst_rank: Optional[int]

    skew_ratio: float
    skew_pct: float


@dataclass(frozen=True)
class StepMemoryCombinedCoverage:
    """
    Coverage / health information for the combined memory metric.
    """

    expected_steps: int
    steps_used: int
    completed_step: Optional[int]

    world_size: int
    ranks_present: int
    incomplete: bool


@dataclass(frozen=True)
class StepMemoryCombinedMetric:
    """
    Renderer-ready payload for ONE combined step memory metric.

    metric
      - "peak_allocated"
      - "peak_reserved"
    """

    metric: str
    device: Optional[str]  # e.g. "cuda:0" or None if unknown/mixed

    series: StepMemoryCombinedSeries
    summary: StepMemoryCombinedSummary
    coverage: StepMemoryCombinedCoverage


@dataclass(frozen=True)
class StepMemoryCombinedResult:
    """
    Final renderer-facing payload for step memory combined.

    This is the ONLY object UI layers should consume.
    """

    metrics: List[StepMemoryCombinedMetric]
    status_message: str
