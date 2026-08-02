"""Presentation-independent data contracts for the Step Time pipeline.

The objects in this module describe analyzed Step Time data shared by the
CLI, dashboard, diagnostics, and final summary.  They intentionally contain
no database access, diagnosis policy, or presentation behavior.  Keeping
these contracts at the bottom of the dependency graph lets every surface use
the same facts without making core analysis depend on a renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
)

DiagnosisClock = Literal["cpu", "gpu"]
"""Clock selected consistently across one analysis window."""

DIAGNOSIS_CLOCK_KEY = "diagnosis_clock"
"""Stable final-summary key that records the selected diagnosis clock."""

STEP_TIME_EVENT_NAMES: Mapping[str, str] = {
    "input_wait": "_traceml_internal:dataloader_next",
    "h2d": "_traceml_internal:h2d_time",
    "forward": "_traceml_internal:forward_time",
    "backward": "_traceml_internal:backward_time",
    "optimizer_step": "_traceml_internal:optimizer_step",
    "step_time": "_traceml_internal:step_time",
}
"""Persisted event names keyed by their canonical Step Time metric."""


@dataclass(frozen=True, slots=True)
class StepTimeLoadRequest:
    """Selection parameters for one bounded Step Time repository read.

    ``window_size`` is the final analysis window. ``lookback_factor`` expands
    the distinct per-rank steps returned to the alignment layer without
    changing that final size. ``rank_filter=None`` selects every global rank;
    an empty tuple deliberately selects none.
    """

    window_size: int
    lookback_factor: int = 1
    rank_filter: Optional[tuple[int, ...]] = None


class StepTimeClockValues(NamedTuple):
    """Normalized CPU and GPU durations for one metric occurrence."""

    cpu_ms: Optional[float] = None
    gpu_ms: Optional[float] = None


class StepTimeSourceRow(NamedTuple):
    """One decoded, deduplicated global-rank step from persisted telemetry."""

    source_id: int
    global_rank: int
    step: int
    metrics: Mapping[str, StepTimeClockValues]


@dataclass(frozen=True, slots=True)
class StepTimeSourceCursor:
    """Append-oriented source position returned with a repository snapshot."""

    last_row_id: Optional[int] = None
    latest_step: Optional[int] = None


@dataclass(frozen=True, slots=True)
class StepTimeRankIdentity:
    """Latest persisted distributed identity for one global rank."""

    global_rank: int
    local_rank: Optional[int] = None
    node_rank: Optional[int] = None
    hostname: Optional[str] = None
    local_world_size: Optional[int] = None
    world_size: Optional[int] = None


@dataclass(frozen=True, slots=True)
class StepTimeRepositorySnapshot:
    """Consistent source facts returned by one repository load.

    The repository does not align steps, choose a clock, derive metrics, or
    diagnose. Those responsibilities remain in later pipeline layers. Live
    snapshots intentionally leave summary-only identity, progress, and cursor
    fields empty.
    """

    rows: tuple[StepTimeSourceRow, ...] = ()
    global_ranks: tuple[int, ...] = ()
    identities: Mapping[int, StepTimeRankIdentity] = field(
        default_factory=dict
    )
    latest_step_observed: Optional[int] = None
    cursor: StepTimeSourceCursor = field(default_factory=StepTimeSourceCursor)
    training_strategy: str = "ddp"


@dataclass(frozen=True)
class StepTimeSeries:
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
class StepTimeSummary:
    """Window-level statistics for one measured Step Time metric."""

    window_size: int
    steps_used: int
    median_total: float
    worst_total: float
    worst_rank: Optional[int]
    skew_ratio: Optional[float]
    skew_pct: Optional[float]


@dataclass(frozen=True)
class StepTimeCoverage:
    """Completeness metadata for an aligned Step Time window."""

    expected_steps: int
    steps_used: int
    completed_step: int
    world_size: int
    ranks_present: int
    incomplete: bool


@dataclass(frozen=True)
class StepTimeMetric:
    """One canonical metric with optional series and aggregate statistics."""

    metric: str
    clock: str  # "cpu" | "gpu" | "mixed"
    series: Optional[StepTimeSeries]
    summary: StepTimeSummary
    coverage: StepTimeCoverage


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
    coverage: StepTimeCoverage = field(
        default_factory=lambda: StepTimeCoverage(
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
    metrics: list[StepTimeMetric] = field(default_factory=list)

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
class StepTimeResult:
    """Live Step Time state wrapping one canonical analyzed window."""

    status_message: str = "OK"
    window: Optional[StepTimeWindow] = None
    training_strategy: str = "ddp"
    had_ok: bool = False


__all__ = [
    "DIAGNOSIS_CLOCK_KEY",
    "DiagnosisClock",
    "STEP_TIME_EVENT_NAMES",
    "StepTimeClockValues",
    "StepTimeCoverage",
    "StepTimeLoadRequest",
    "StepTimeMetric",
    "StepTimeRankIdentity",
    "StepTimeRepositorySnapshot",
    "StepTimeResult",
    "StepTimeSeries",
    "StepTimeSourceCursor",
    "StepTimeSourceRow",
    "StepTimeSummary",
    "StepTimeWindow",
]
