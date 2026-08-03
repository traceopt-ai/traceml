"""Presentation-independent data contracts for the Step Time pipeline.

The objects in this module describe analyzed Step Time data shared by the
CLI, dashboard, diagnostics, and final summary.  They intentionally contain
no database access, diagnosis policy, or presentation behavior.  Keeping
these contracts at the bottom of the dependency graph lets every surface use
the same facts without making core analysis depend on a renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Mapping, NamedTuple, Optional

DiagnosisClock = Literal["cpu", "gpu"]
"""Clock selected consistently across one analysis window."""

STEP_TIME_EVENT_NAMES: Mapping[str, str] = {
    "input_wait": "_traceml_internal:dataloader_next",
    "h2d": "_traceml_internal:h2d_time",
    "forward": "_traceml_internal:forward_time",
    "backward": "_traceml_internal:backward_time",
    "optimizer_step": "_traceml_internal:optimizer_step",
    # The raw event is renamed by the instrumentation migration. Until then,
    # this decoder maps its historical spelling to the canonical inner metric.
    "traced_step_time": "_traceml_internal:step_time",
}
"""Persisted event names keyed by canonical timing metrics."""

_VALUE_FIELDS: Mapping[str, str] = {
    "input_wait": "input_wait_ms",
    "h2d": "h2d_ms",
    "forward": "forward_ms",
    "backward": "backward_ms",
    "optimizer_step": "optimizer_step_ms",
    "step_time": "step_time_ms",
    "traced_step_time": "traced_step_time_ms",
    "compute": "compute_ms",
    "residual_proxy": "residual_ms",
    "step_time_cpu": "step_time_cpu_ms",
    "step_time_gpu": "step_time_gpu_ms",
    "traced_step_time_cpu": "traced_step_time_cpu_ms",
    "traced_step_time_gpu": "traced_step_time_gpu_ms",
    "dataloader_fetch": "dataloader_fetch_cpu_ms",
}


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
    snapshots omit summary-only identities but retain their selected source
    cursor so the later live-session cache can detect unchanged input.
    """

    rows: tuple[StepTimeSourceRow, ...] = ()
    global_ranks: tuple[int, ...] = ()
    identities: Mapping[int, StepTimeRankIdentity] = field(
        default_factory=dict
    )
    cursor: StepTimeSourceCursor = field(default_factory=StepTimeSourceCursor)
    training_strategy: str = "ddp"


@dataclass(frozen=True, slots=True)
class StepTimeValues:
    """Typed timing values for one step or one rank-window average.

    The phase fields use the clock selected for the complete analysis window.
    ``step_time_ms`` is the selected-clock outer duration and
    ``traced_step_time_ms`` is the selected-clock inner trace envelope.
    Clock-qualified fields preserve both aggregate clocks independently.
    ``None`` means the signal was unavailable; a measured zero remains
    ``0.0``.
    """

    input_wait_ms: Optional[float] = None
    h2d_ms: Optional[float] = None
    forward_ms: Optional[float] = None
    backward_ms: Optional[float] = None
    optimizer_step_ms: Optional[float] = None
    step_time_ms: Optional[float] = None
    traced_step_time_ms: Optional[float] = None
    compute_ms: Optional[float] = None
    residual_ms: Optional[float] = None
    step_time_cpu_ms: Optional[float] = None
    step_time_gpu_ms: Optional[float] = None
    traced_step_time_cpu_ms: Optional[float] = None
    traced_step_time_gpu_ms: Optional[float] = None
    dataloader_fetch_cpu_ms: Optional[float] = None

    def value(self, metric: str) -> Optional[float]:
        """Return one canonical value by its internal metric key."""
        field_name = _VALUE_FIELDS.get(str(metric))
        return getattr(self, field_name) if field_name is not None else None


@dataclass(frozen=True, slots=True)
class StepTimeStepFacts:
    """Canonical selected-clock facts for one aligned training step."""

    step: int
    values: StepTimeValues


@dataclass(frozen=True, slots=True)
class StepTimeRankFacts:
    """Canonical per-step and averaged facts for one global rank."""

    global_rank: int
    steps: tuple[StepTimeStepFacts, ...] = ()
    average: StepTimeValues = field(default_factory=StepTimeValues)


@dataclass(frozen=True)
class StepTimeSeries:
    """Per-step aggregate values for one metric across participating ranks.

    Attributes:
        median: Median rank value for each aligned step.
        worst: Largest rank value for each aligned step.
    """

    median: List[float]
    worst: List[float]


@dataclass(frozen=True)
class StepTimeCoverage:
    """Completeness metadata for an aligned Step Time window."""

    expected_steps: int
    steps_used: int
    world_size: int
    ranks_present: int


@dataclass(frozen=True)
class StepTimeMetric:
    """Series and window statistics for one measured Step Time signal."""

    metric: str
    series: Optional[StepTimeSeries]
    median_total: float
    worst_total: float
    worst_rank: Optional[int]
    skew_pct: Optional[float]
    representative_rank: Optional[int] = None
    representative_total: Optional[float] = None


@dataclass(frozen=True)
class StepTimeWindow:
    """Aligned selected-clock facts for one Step Time analysis window.

    ``rank_facts`` is the sole canonical rank representation. Missing fields
    are unavailable and measured zeroes remain ``0.0``. The explicit cohorts,
    rank statistics, and component shares are computed once by the analyzer.
    Consumers should use :meth:`ranks_for` instead of recreating availability
    rules.
    """

    clock: DiagnosisClock = "cpu"
    training_strategy: str = "ddp"
    steps: list[int] = field(default_factory=list)
    expected_ranks: tuple[int, ...] = ()
    coverage: StepTimeCoverage = field(
        default_factory=lambda: StepTimeCoverage(
            expected_steps=0,
            steps_used=0,
            world_size=0,
            ranks_present=0,
        )
    )
    rank_facts: tuple[StepTimeRankFacts, ...] = ()
    metrics: list[StepTimeMetric] = field(default_factory=list)
    step_ranks: tuple[int, ...] = ()
    compute_ranks: tuple[int, ...] = ()
    straggler_ranks: tuple[int, ...] = ()
    composition_representative_rank: Optional[int] = None
    input_wait_share: Optional[float] = None
    h2d_share: Optional[float] = None
    compute_share: Optional[float] = None
    residual_share: Optional[float] = None

    @property
    def rank_universe(self) -> tuple[int, ...]:
        """Return expected ranks, or observed ranks for direct fixtures."""
        if self.expected_ranks:
            return self.expected_ranks
        return tuple(sorted(facts.global_rank for facts in self.rank_facts))

    def rank(self, global_rank: int) -> Optional[StepTimeRankFacts]:
        """Return canonical facts for one global rank, when observed."""
        rank = int(global_rank)
        return next(
            (facts for facts in self.rank_facts if facts.global_rank == rank),
            None,
        )

    def ranks_for(self, metric: str) -> tuple[int, ...]:
        """Return ranks carrying one canonical sparse metric."""
        key = str(metric)
        return tuple(
            facts.global_rank
            for facts in self.rank_facts
            if facts.average.value(key) is not None
        )


__all__ = [
    "DiagnosisClock",
    "STEP_TIME_EVENT_NAMES",
    "StepTimeClockValues",
    "StepTimeCoverage",
    "StepTimeLoadRequest",
    "StepTimeMetric",
    "StepTimeRankIdentity",
    "StepTimeRankFacts",
    "StepTimeRepositorySnapshot",
    "StepTimeSeries",
    "StepTimeSourceCursor",
    "StepTimeSourceRow",
    "StepTimeStepFacts",
    "StepTimeValues",
    "StepTimeWindow",
]
