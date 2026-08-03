"""Canonical Step Time analysis over normalized repository snapshots.

The analyzer is the only production owner of Step Time alignment, clock
selection, signal availability, derived values, rank averages, cohorts, and
statistics. It consumes flat :class:`StepTimeSourceRow` objects and emits one
typed :class:`StepTimeWindow`; it neither reads SQLite nor knows presentation
labels, diagnosis thresholds, or reporting schemas.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple, Optional, Sequence

import numpy as np

from traceml_ai.step_time.model import (
    DiagnosisClock,
    StepTimeCoverage,
    StepTimeMetric,
    StepTimeRankFacts,
    StepTimeRepositorySnapshot,
    StepTimeSeries,
    StepTimeSourceRow,
    StepTimeStepFacts,
    StepTimeValues,
    StepTimeWindow,
)

INPUT_WAIT_KEY = "input_wait"
DATALOADER_FETCH_KEY = "dataloader_fetch"
TRACED_STEP_TIME_KEY = "traced_step_time"
_AGGREGATE_METRICS: tuple[str, ...] = (
    INPUT_WAIT_KEY,
    TRACED_STEP_TIME_KEY,
)

SELECTED_METRICS: tuple[str, ...] = (
    INPUT_WAIT_KEY,
    "h2d",
    "forward",
    "backward",
    "optimizer_step",
    TRACED_STEP_TIME_KEY,
)

DISPLAY_METRICS: tuple[str, ...] = (
    INPUT_WAIT_KEY,
    "h2d",
    "forward",
    "backward",
    "optimizer_step",
    "step_time",
    TRACED_STEP_TIME_KEY,
    "residual_proxy",
)

STATISTIC_METRICS: tuple[str, ...] = (
    *DISPLAY_METRICS,
    "compute",
    "step_time_cpu",
    "step_time_gpu",
    "traced_step_time_cpu",
    "traced_step_time_gpu",
    DATALOADER_FETCH_KEY,
)

_REQUIRED_GPU_METRICS: tuple[str, ...] = _AGGREGATE_METRICS
OCCURRENCE_METRICS: frozenset[str] = frozenset(("optimizer_step", "h2d"))
_COMPOSITION_KEYS: tuple[str, ...] = (
    INPUT_WAIT_KEY,
    "forward",
    "backward",
    "optimizer_step",
    "residual_proxy",
)

_RowIndex = Mapping[tuple[int, int], StepTimeSourceRow]


class _Cohorts(NamedTuple):
    step: tuple[int, ...]
    compute: tuple[int, ...]
    composition: tuple[int, ...]
    straggler: tuple[int, ...]


class _RankAvailability(NamedTuple):
    """Window-complete selected and aggregate signal availability."""

    selected: frozenset[str]
    cpu_aggregate: frozenset[str]
    gpu_aggregate: frozenset[str]


class StepTimeAnalyzer:
    """Build one canonical fact set from a repository snapshot.

    The implementation deliberately keeps its four analysis phases in this
    module so a contributor can follow the complete domain algorithm without
    navigating a helper package.
    """

    def analyze(
        self,
        snapshot: StepTimeRepositorySnapshot,
        *,
        window_size: int,
    ) -> StepTimeWindow:
        """Analyze a bounded source snapshot without mutating source facts."""
        declared_ranks = {int(rank) for rank in snapshot.global_ranks}
        expected_ranks = tuple(
            sorted(
                declared_ranks
                or {int(row.global_rank) for row in snapshot.rows}
            )
        )
        limit = max(1, int(window_size))

        # Phase 1: index references to flat rows and align one common suffix.
        row_index, steps_by_rank = _index_source_rows(snapshot.rows)
        steps = _common_suffix(steps_by_rank, limit)
        if not steps:
            return StepTimeWindow(
                expected_ranks=expected_ranks,
                training_strategy=snapshot.training_strategy,
                coverage=_empty_coverage(
                    window_size=limit,
                    world_size=len(expected_ranks),
                    ranks_present=len(steps_by_rank),
                ),
            )

        # Phase 2: one clock and one availability decision per rank/metric.
        observed_ranks = tuple(sorted(steps_by_rank))
        clock = _select_clock(row_index, observed_ranks, steps)
        availability = {
            rank: _RankAvailability(
                selected=_rank_availability(
                    row_index,
                    rank,
                    steps,
                    clock=clock,
                    metrics=SELECTED_METRICS,
                ),
                cpu_aggregate=_rank_availability(
                    row_index,
                    rank,
                    steps,
                    clock="cpu",
                    metrics=_AGGREGATE_METRICS,
                ),
                gpu_aggregate=_rank_availability(
                    row_index,
                    rank,
                    steps,
                    clock="gpu",
                    metrics=_AGGREGATE_METRICS,
                ),
            )
            for rank in observed_ranks
        }

        # Phase 3: typed per-step values and rank averages, derived once.
        rank_facts = tuple(
            _build_rank_facts(
                row_index,
                rank,
                steps,
                availability=availability[rank],
                clock=clock,
            )
            for rank in observed_ranks
        )

        # Phase 4: reusable coverage, statistics, cohorts, and shares.
        coverage = StepTimeCoverage(
            expected_steps=limit,
            steps_used=len(steps),
            world_size=len(expected_ranks),
            ranks_present=len(rank_facts),
        )
        metrics = _build_metrics(rank_facts, steps)
        cohorts = _build_cohorts(
            rank_facts,
            training_strategy=snapshot.training_strategy,
        )
        return StepTimeWindow(
            clock=clock,
            training_strategy=snapshot.training_strategy,
            steps=[int(step) for step in steps],
            expected_ranks=expected_ranks,
            coverage=coverage,
            rank_facts=rank_facts,
            metrics=metrics,
            step_ranks=cohorts.step,
            compute_ranks=cohorts.compute,
            straggler_ranks=cohorts.straggler,
            composition_representative_rank=(
                _composition_representative_rank(
                    rank_facts,
                    cohorts.composition,
                )
            ),
            input_wait_share=_component_share(rank_facts, INPUT_WAIT_KEY),
            h2d_share=_component_share(rank_facts, "h2d"),
            compute_share=_component_share(rank_facts, "compute"),
            residual_share=_component_share(rank_facts, "residual_proxy"),
        )


def _index_source_rows(
    rows: Sequence[StepTimeSourceRow],
) -> tuple[dict[tuple[int, int], StepTimeSourceRow], dict[int, set[int]]]:
    """Index source-row references without creating another timing payload."""
    row_index: dict[tuple[int, int], StepTimeSourceRow] = {}
    steps_by_rank: dict[int, set[int]] = {}
    for row in rows:
        rank = int(row.global_rank)
        step = int(row.step)
        key = (rank, step)
        previous = row_index.get(key)
        if previous is not None and previous.source_id >= row.source_id:
            continue
        row_index[key] = row
        steps_by_rank.setdefault(rank, set()).add(step)
    return row_index, steps_by_rank


def _common_suffix(
    steps_by_rank: Mapping[int, set[int]],
    window_size: int,
) -> tuple[int, ...]:
    """Return the latest steps observed for every rank carrying rows."""
    if not steps_by_rank:
        return ()
    common = set.intersection(*steps_by_rank.values())
    if not common:
        return ()
    return tuple(sorted(common)[-max(1, int(window_size)) :])


def _clock_value(
    row: StepTimeSourceRow,
    metric: str,
    *,
    clock: DiagnosisClock,
) -> Optional[float]:
    values = row.metrics.get(metric)
    if values is None:
        return None
    value = values.gpu_ms if clock == "gpu" else values.cpu_ms
    return float(value) if value is not None else None


def _row_supports_gpu(row: StepTimeSourceRow) -> bool:
    for metric in _REQUIRED_GPU_METRICS:
        values = row.metrics.get(metric)
        if values is None or values.gpu_ms is None:
            return False
    for metric in SELECTED_METRICS:
        if metric in _REQUIRED_GPU_METRICS:
            continue
        values = row.metrics.get(metric)
        if values is not None and values.gpu_ms is None:
            return False
    return True


def _select_clock(
    row_index: _RowIndex,
    ranks: Sequence[int],
    steps: Sequence[int],
) -> DiagnosisClock:
    for rank in ranks:
        for step in steps:
            if not _row_supports_gpu(row_index[(int(rank), int(step))]):
                return "cpu"
    return "gpu"


def _rank_availability(
    row_index: _RowIndex,
    rank: int,
    steps: Sequence[int],
    *,
    clock: DiagnosisClock,
    metrics: Sequence[str],
) -> frozenset[str]:
    """Return signals trusted across one rank's aligned window.

    H2D and optimizer events are occurrence-driven, so one measured event
    establishes the signal and absent aligned steps are measured zero. Every
    other signal must be present on every aligned step.
    """
    counts: dict[str, int] = {}
    for step in steps:
        row = row_index[(int(rank), int(step))]
        for metric in metrics:
            if _clock_value(row, metric, clock=clock) is not None:
                counts[metric] = counts.get(metric, 0) + 1

    step_count = len(steps)
    return frozenset(
        metric
        for metric, count in counts.items()
        if count == step_count
        or (metric in OCCURRENCE_METRICS and metric in metrics)
    )


def _build_rank_facts(
    row_index: _RowIndex,
    rank: int,
    steps: Sequence[int],
    *,
    availability: _RankAvailability,
    clock: DiagnosisClock,
) -> StepTimeRankFacts:
    step_facts = tuple(
        StepTimeStepFacts(
            step=int(step),
            values=_build_step_values(
                row_index[(int(rank), int(step))],
                availability=availability,
                clock=clock,
            ),
        )
        for step in steps
    )
    return StepTimeRankFacts(
        global_rank=int(rank),
        steps=step_facts,
        average=_average_values(step_facts),
    )


def _build_step_values(
    row: StepTimeSourceRow,
    *,
    availability: _RankAvailability,
    clock: DiagnosisClock,
) -> StepTimeValues:
    selected = {
        metric: (
            _clock_value(row, metric, clock=clock)
            if metric in availability.selected
            else None
        )
        for metric in SELECTED_METRICS
    }
    for metric in OCCURRENCE_METRICS:
        if metric in availability.selected and selected[metric] is None:
            selected[metric] = 0.0

    input_wait = selected[INPUT_WAIT_KEY]
    h2d = selected["h2d"]
    forward = selected["forward"]
    backward = selected["backward"]
    optimizer = selected["optimizer_step"]
    traced_step_time = selected[TRACED_STEP_TIME_KEY]
    compute = (
        float(forward + backward + optimizer)
        if None not in (forward, backward, optimizer)
        else None
    )
    residual = (
        max(0.0, float(traced_step_time - (h2d or 0.0) - compute))
        if traced_step_time is not None and compute is not None
        else None
    )
    step_time = (
        float(input_wait + traced_step_time)
        if input_wait is not None and traced_step_time is not None
        else None
    )

    cpu_input_wait, traced_step_time_cpu, step_time_cpu = (
        _clock_aggregate_values(
            row,
            clock="cpu",
            availability=availability.cpu_aggregate,
        )
    )
    _gpu_input_wait, traced_step_time_gpu, step_time_gpu = (
        _clock_aggregate_values(
            row,
            clock="gpu",
            availability=availability.gpu_aggregate,
        )
    )
    return StepTimeValues(
        input_wait_ms=input_wait,
        h2d_ms=h2d,
        forward_ms=forward,
        backward_ms=backward,
        optimizer_step_ms=optimizer,
        step_time_ms=step_time,
        traced_step_time_ms=traced_step_time,
        compute_ms=compute,
        residual_ms=residual,
        step_time_cpu_ms=step_time_cpu,
        step_time_gpu_ms=step_time_gpu,
        traced_step_time_cpu_ms=traced_step_time_cpu,
        traced_step_time_gpu_ms=traced_step_time_gpu,
        dataloader_fetch_cpu_ms=cpu_input_wait,
    )


def _clock_aggregate_values(
    row: StepTimeSourceRow,
    *,
    clock: DiagnosisClock,
    availability: frozenset[str],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return one clock's input, traced envelope, and outer Step Time."""
    input_wait = (
        _clock_value(row, INPUT_WAIT_KEY, clock=clock)
        if INPUT_WAIT_KEY in availability
        else None
    )
    traced_step_time = (
        _clock_value(row, TRACED_STEP_TIME_KEY, clock=clock)
        if TRACED_STEP_TIME_KEY in availability
        else None
    )
    step_time = (
        float(input_wait + traced_step_time)
        if input_wait is not None and traced_step_time is not None
        else None
    )
    return input_wait, traced_step_time, step_time


def _average_values(steps: Sequence[StepTimeStepFacts]) -> StepTimeValues:
    divisor = float(len(steps)) if steps else 1.0

    def average(field_name: str) -> Optional[float]:
        values = [getattr(step.values, field_name) for step in steps]
        if not values or all(value is None for value in values):
            return None
        return float(sum(float(value or 0.0) for value in values) / divisor)

    input_wait = average("input_wait_ms")
    forward = average("forward_ms")
    backward = average("backward_ms")
    optimizer = average("optimizer_step_ms")
    return StepTimeValues(
        input_wait_ms=input_wait,
        h2d_ms=average("h2d_ms"),
        forward_ms=forward,
        backward_ms=backward,
        optimizer_step_ms=optimizer,
        step_time_ms=average("step_time_ms"),
        traced_step_time_ms=average("traced_step_time_ms"),
        compute_ms=(
            float(forward + backward + optimizer)
            if None not in (forward, backward, optimizer)
            else None
        ),
        residual_ms=average("residual_ms"),
        step_time_cpu_ms=average("step_time_cpu_ms"),
        step_time_gpu_ms=average("step_time_gpu_ms"),
        traced_step_time_cpu_ms=average("traced_step_time_cpu_ms"),
        traced_step_time_gpu_ms=average("traced_step_time_gpu_ms"),
        dataloader_fetch_cpu_ms=average("dataloader_fetch_cpu_ms"),
    )


def _build_metrics(
    rank_facts: Sequence[StepTimeRankFacts],
    steps: Sequence[int],
) -> list[StepTimeMetric]:
    metrics: list[StepTimeMetric] = []
    for metric in STATISTIC_METRICS:
        values = {
            facts.global_rank: value
            for facts in rank_facts
            if (value := facts.average.value(metric)) is not None
        }
        if not values:
            continue
        measured_ranks = tuple(sorted(values))
        array = np.asarray(
            [values[rank] for rank in measured_ranks],
            dtype=np.float64,
        )
        median = float(np.median(array))
        worst_index = int(np.argmax(array))
        worst = float(array[worst_index])
        worst_rank = measured_ranks[worst_index]
        representative_rank = min(
            measured_ranks,
            key=lambda rank: (
                abs(values[rank] - median),
                values[rank],
                rank,
            ),
        )
        skew_pct = _skew(
            median,
            worst,
            population=len(measured_ranks),
        )
        metrics.append(
            StepTimeMetric(
                metric=metric,
                series=(
                    _build_series(
                        metric,
                        rank_facts,
                        measured_ranks,
                        steps,
                    )
                    if metric in DISPLAY_METRICS
                    else None
                ),
                median_total=median,
                worst_total=worst,
                worst_rank=int(worst_rank),
                skew_pct=skew_pct,
                representative_rank=int(representative_rank),
                representative_total=float(values[representative_rank]),
            )
        )
    return metrics


def _build_series(
    metric: str,
    rank_facts: Sequence[StepTimeRankFacts],
    measured_ranks: Sequence[int],
    steps: Sequence[int],
) -> StepTimeSeries:
    by_rank = {facts.global_rank: facts for facts in rank_facts}
    median_values: list[float] = []
    worst_values: list[float] = []
    for offset, _step in enumerate(steps):
        values = np.asarray(
            [
                by_rank[rank].steps[offset].values.value(metric) or 0.0
                for rank in measured_ranks
            ],
            dtype=np.float64,
        )
        median_values.append(float(np.median(values)))
        worst_values.append(float(np.max(values)))
    return StepTimeSeries(
        median=median_values,
        worst=worst_values,
    )


def _skew(
    median: float,
    worst: float,
    *,
    population: int,
) -> Optional[float]:
    if population <= 1:
        return None
    if median > 0.0:
        return (worst - median) / median
    if worst <= 0.0:
        return 0.0
    return None


def _build_cohorts(
    rank_facts: Sequence[StepTimeRankFacts],
    *,
    training_strategy: str,
) -> _Cohorts:
    def carrying(*metrics: str) -> tuple[int, ...]:
        return tuple(
            facts.global_rank
            for facts in rank_facts
            if all(
                facts.average.value(metric) is not None for metric in metrics
            )
        )

    step = carrying("step_time")
    compute = carrying(
        "step_time",
        "forward",
        "backward",
        "optimizer_step",
    )
    present_composition = tuple(
        metric
        for metric in _COMPOSITION_KEYS
        if any(facts.average.value(metric) is not None for facts in rank_facts)
    )
    composition = carrying("step_time", *present_composition)

    fsdp = str(training_strategy).strip().lower() == "fsdp"
    straggler = tuple(
        facts.global_rank
        for facts in rank_facts
        if facts.average.input_wait_ms is not None
        and (facts.average.step_time_ms or 0.0) > 0.0
        and (facts.average.backward_ms or 0.0) > 0.0
        and (not fsdp or (facts.average.forward_ms or 0.0) > 0.0)
    )
    return _Cohorts(
        step=step,
        compute=compute,
        composition=composition,
        straggler=straggler,
    )


def _composition_representative_rank(
    rank_facts: Sequence[StepTimeRankFacts],
    cohort: Sequence[int],
) -> Optional[int]:
    cohort_set = set(int(rank) for rank in cohort)
    anchors = {
        facts.global_rank: facts.average.step_time_ms
        for facts in rank_facts
        if facts.global_rank in cohort_set
    }
    clean = {
        rank: value for rank, value in anchors.items() if value is not None
    }
    if not clean:
        return None
    median = float(np.median(np.asarray(tuple(clean.values()))))
    # Preserve the existing dashboard tie-break: nearest, then lowest rank.
    return min(
        clean,
        key=lambda rank: (abs(float(clean[rank]) - median), rank),
    )


def _component_share(
    rank_facts: Sequence[StepTimeRankFacts],
    component: str,
) -> Optional[float]:
    shares: list[float] = []
    for facts in rank_facts:
        values = facts.average
        numerator = values.value(component)
        step_time = values.step_time_ms
        if numerator is None or step_time is None:
            continue
        if step_time > 0.0:
            shares.append(float(numerator) / step_time)
    if not shares:
        return None
    return float(np.median(np.asarray(shares, dtype=np.float64)))


def _empty_coverage(
    *,
    window_size: int,
    world_size: int,
    ranks_present: int,
) -> StepTimeCoverage:
    return StepTimeCoverage(
        expected_steps=max(1, int(window_size)),
        steps_used=0,
        world_size=max(0, int(world_size)),
        ranks_present=max(0, int(ranks_present)),
    )


__all__ = [
    "DATALOADER_FETCH_KEY",
    "DISPLAY_METRICS",
    "INPUT_WAIT_KEY",
    "OCCURRENCE_METRICS",
    "SELECTED_METRICS",
    "STATISTIC_METRICS",
    "StepTimeAnalyzer",
]
