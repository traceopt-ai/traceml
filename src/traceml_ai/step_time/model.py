"""Presentation-independent data contracts for the Step Time pipeline.

The objects in this module describe analyzed Step Time data shared by the
CLI, dashboard, diagnostics, and final summary.  They intentionally contain
no database access, diagnosis policy, or presentation behavior.  Keeping
these contracts at the bottom of the dependency graph lets every surface use
the same facts without making core analysis depend on a renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence

DiagnosisClock = Literal["cpu", "gpu"]
"""Clock selected consistently across one analysis window."""

DIAGNOSIS_CLOCK_KEY = "diagnosis_clock"
"""Stable final-summary key that records the selected diagnosis clock."""


@dataclass(frozen=True)
class StepCombinedTimeSeries:
    """Per-step aggregate values for one metric across participating ranks.

    Attributes:
        steps: Aligned completed step identifiers.
        median: Median rank value for each aligned step.
        worst: Largest rank value for each aligned step.
        sum: Sum of rank values for each aligned step.
    """

    steps: List[int]
    median: List[float]
    worst: List[float]
    sum: List[float]


@dataclass(frozen=True)
class StepCombinedTimeSummary:
    """Window-level statistics for one measured Step Time metric."""

    window_size: int
    steps_used: int
    median_total: float
    worst_total: float
    worst_rank: Optional[int]
    skew_ratio: Optional[float]
    skew_pct: Optional[float]


@dataclass(frozen=True)
class StepCombinedTimeCoverage:
    """Completeness metadata for an aligned Step Time window."""

    expected_steps: int
    steps_used: int
    completed_step: int
    world_size: int
    ranks_present: int
    incomplete: bool


@dataclass(frozen=True)
class StepCombinedTimeMetric:
    """One canonical metric with optional series and aggregate statistics."""

    metric: str
    clock: str  # "cpu" | "gpu" | "mixed"
    series: Optional[StepCombinedTimeSeries]
    summary: StepCombinedTimeSummary
    coverage: StepCombinedTimeCoverage


@dataclass(frozen=True)
class StepTimeWindow:
    """Aligned selected-clock facts for one Step Time analysis window.

    Metric rows are intentionally sparse: a missing key means that the signal
    was unavailable, while a present ``0.0`` means it was measured as zero.
    Consumers should use :meth:`ranks_for` or :meth:`eligible_ranks` instead
    of recreating availability rules.
    """

    clock: DiagnosisClock = "cpu"
    steps: list[int] = field(default_factory=list)
    expected_ranks: tuple[int, ...] = ()
    coverage: StepCombinedTimeCoverage = field(
        default_factory=lambda: StepCombinedTimeCoverage(
            expected_steps=0,
            steps_used=0,
            completed_step=0,
            world_size=0,
            ranks_present=0,
            incomplete=False,
        )
    )
    per_rank_step_timing: Dict[int, Dict[int, Dict[str, float]]] = field(
        default_factory=dict
    )
    per_rank_timing: Dict[int, Dict[str, float]] = field(default_factory=dict)
    metrics: list[StepCombinedTimeMetric] = field(default_factory=list)

    @property
    def rank_universe(self) -> tuple[int, ...]:
        """Return expected ranks, or observed ranks for direct fixtures."""
        if self.expected_ranks:
            return self.expected_ranks
        return tuple(sorted(int(rank) for rank in self.per_rank_timing))

    def ranks_for(self, metric: str) -> tuple[int, ...]:
        """Return ranks carrying one canonical sparse metric."""
        key = str(metric)
        return tuple(
            rank
            for rank in self.rank_universe
            if key in self.per_rank_timing.get(rank, {})
        )

    def eligible_ranks(self, metrics: Sequence[str]) -> tuple[int, ...]:
        """Return ranks carrying every requested metric in this window."""
        keys = tuple(str(metric) for metric in metrics)
        if not keys:
            return self.rank_universe
        return tuple(
            rank
            for rank in self.rank_universe
            if all(key in self.per_rank_timing.get(rank, {}) for key in keys)
        )

    def is_complete(self, metric: str) -> bool:
        """Return whether every rank in the window measured one metric."""
        ranks = self.rank_universe
        return bool(ranks) and self.ranks_for(metric) == ranks

    def to_json(self) -> Dict[str, Any]:
        """Return the aligned-window metadata used by ``final_summary``."""
        return {
            "alignment": "common_steps",
            "aligned_steps_analyzed": int(self.coverage.steps_used),
            "start_step": self.steps[0] if self.steps else None,
            "end_step": self.steps[-1] if self.steps else None,
            "window_size": int(self.coverage.expected_steps),
            DIAGNOSIS_CLOCK_KEY: self.clock,
        }


@dataclass(frozen=True)
class StepCombinedTimeResult:
    """Live Step Time state wrapping one canonical analyzed window."""

    status_message: str = "OK"
    window: Optional[StepTimeWindow] = None
    training_strategy: str = "ddp"
    had_ok: bool = False


__all__ = [
    "DIAGNOSIS_CLOCK_KEY",
    "DiagnosisClock",
    "StepCombinedTimeCoverage",
    "StepCombinedTimeMetric",
    "StepCombinedTimeResult",
    "StepCombinedTimeSeries",
    "StepCombinedTimeSummary",
    "StepTimeWindow",
]
