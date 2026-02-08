"""
Renderer-facing schema for step memory combined.

This schema represents a *capacity / tail-risk oriented* view of
step-level peak memory aggregated across DDP ranks.

Design goals
------------
- Renderer/UI consumes ONLY these dataclasses
- Stable, explicit semantics (MB units)
- Step alignment across ranks (intersection-based)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class StepMemoryCombinedSeries:
    """
    Per-step aggregated memory time series (renderer-facing).

    Semantics
    ---------
    - Aggregated across ranks per step
    - `median` = typical rank memory
    - `worst`  = maximum across ranks (tail / gating rank)
    """

    steps: List[int]
    median: List[float]
    worst: List[float]


@dataclass(frozen=True)
class StepMemoryCombinedSummary:
    """
    Window-level summary over last N complete steps.

    Semantics
    ---------
    - `median_peak_mb` is the median across ranks of their max(memory) over the window.
    - `worst_peak_mb` is the max across ranks of their max(memory) over the window.
    - `worst_rank` is the rank achieving `worst_peak_mb`.
    - `skew_pct` measures imbalance between worst and median peak.
    """

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
      - "peak_allocated_mb"
      - "peak_reserved_mb"
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
