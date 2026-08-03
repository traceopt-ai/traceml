"""Small typed Step Time factories for presentation-unit tests.

Production-path tests should prefer repository snapshots and
``StepTimeAnalyzer``. These factories exist for renderer tests that already
provide aggregate metric objects and only need a typed rank-fact window.
"""

from __future__ import annotations

import statistics
from typing import Any, Mapping, Optional, Sequence

from traceml_ai.diagnostics.step_time import (
    LIVE_STEP_TIME_POLICY,
    diagnose_step_time_window,
)
from traceml_ai.step_time.analysis import StepTimeAnalyzer
from traceml_ai.step_time.model import (
    DiagnosisClock,
    StepTimeCoverage,
    StepTimeLoadRequest,
    StepTimeMetric,
    StepTimeRankFacts,
    StepTimeRepositorySnapshot,
    StepTimeSourceCursor,
    StepTimeSourceRow,
    StepTimeValues,
    StepTimeWindow,
)
from traceml_ai.step_time.pipeline import (
    LiveStepTimeFreshness,
    LiveStepTimeResult,
    StepTimeAnalysis,
)
from traceml_ai.step_time.sqlite import normalize_step_time_events


def window_from_events(
    per_rank_steps: Mapping[int, Mapping[int, Any]],
    *,
    max_rows: int,
    expected_ranks: Optional[Sequence[int]] = None,
    training_strategy: str = "ddp",
) -> StepTimeWindow:
    """Analyze concise raw-event fixtures through the production analyzer.

    Raw event dictionaries belong in tests, not in the production Step Time
    API. This factory keeps event-heavy diagnosis and integration fixtures
    readable while exercising the same normalization and analysis code as a
    repository load.
    """
    expected = tuple(
        sorted(
            {int(rank) for rank in (expected_ranks or per_rank_steps.keys())}
        )
    )
    rows: list[StepTimeSourceRow] = []
    source_id = 0
    for rank, step_map in per_rank_steps.items():
        for step, events in step_map.items():
            source_id += 1
            rows.append(
                StepTimeSourceRow(
                    source_id=source_id,
                    global_rank=int(rank),
                    step=int(step),
                    metrics=normalize_step_time_events(events) or {},
                )
            )

    window = StepTimeAnalyzer().analyze(
        StepTimeRepositorySnapshot(
            rows=tuple(rows),
            global_ranks=expected,
            training_strategy=str(training_strategy),
        ),
        window_size=max_rows,
    )
    return window


def rank_average(window: StepTimeWindow, global_rank: int) -> StepTimeValues:
    """Return one fixture rank's typed average, failing clearly if absent."""
    facts = window.rank(global_rank)
    if facts is None:
        raise AssertionError(f"Step Time rank r{global_rank} is unavailable")
    return facts.average


def values_from_mapping(
    values: Mapping[str, float],
) -> StepTimeValues:
    """Translate one concise sparse fixture into canonical typed values."""

    def optional(key: str) -> Optional[float]:
        return float(values[key]) if key in values else None

    forward = optional("forward")
    backward = optional("backward")
    optimizer = optional("optimizer_step")
    compute = optional("compute")
    if compute is None and None not in (forward, backward, optimizer):
        compute = float(forward + backward + optimizer)

    input_wait = optional("input_wait")
    step_time = optional("step_time")
    total_step = optional("total_step")
    if total_step is None and input_wait is not None and step_time is not None:
        total_step = input_wait + step_time

    dataloader_cpu = optional("dataloader_fetch")
    step_time_cpu = optional("step_time_cpu")
    total_step_cpu = (
        dataloader_cpu + step_time_cpu
        if dataloader_cpu is not None and step_time_cpu is not None
        else None
    )
    return StepTimeValues(
        input_wait_ms=input_wait,
        h2d_ms=optional("h2d"),
        forward_ms=forward,
        backward_ms=backward,
        optimizer_step_ms=optimizer,
        step_time_ms=step_time,
        compute_ms=compute,
        residual_ms=optional("residual_proxy"),
        total_step_ms=total_step,
        dataloader_cpu_ms=dataloader_cpu,
        total_step_cpu_ms=total_step_cpu,
    )


def window_from_rank_averages(
    per_rank_timing: Mapping[int, Mapping[str, float]],
    *,
    metrics: Sequence[StepTimeMetric] = (),
    clock: DiagnosisClock = "cpu",
    expected_ranks: Optional[Sequence[int]] = None,
    training_strategy: str = "ddp",
    steps_used: Optional[int] = None,
) -> StepTimeWindow:
    """Build a typed aggregate-only window for a renderer unit test."""
    rank_facts = tuple(
        StepTimeRankFacts(
            global_rank=int(rank),
            average=values_from_mapping(values),
        )
        for rank, values in sorted(per_rank_timing.items())
    )
    expected = tuple(
        sorted(
            int(rank)
            for rank in (
                expected_ranks
                if expected_ranks is not None
                else per_rank_timing.keys()
            )
        )
    )

    def carrying(*names: str) -> tuple[int, ...]:
        return tuple(
            facts.global_rank
            for facts in rank_facts
            if all(facts.average.value(name) is not None for name in names)
        )

    composition_names = tuple(
        name
        for name in (
            "input_wait",
            "forward",
            "backward",
            "optimizer_step",
            "residual_proxy",
        )
        if any(facts.average.value(name) is not None for facts in rank_facts)
    )
    composition = carrying("step_time", *composition_names)
    strategy = str(training_strategy).strip().lower()
    straggler = tuple(
        facts.global_rank
        for facts in rank_facts
        if facts.average.input_wait_ms is not None
        and (facts.average.step_time_ms or 0.0) > 0.0
        and (facts.average.backward_ms or 0.0) > 0.0
        and (strategy != "fsdp" or (facts.average.forward_ms or 0.0) > 0.0)
    )
    inferred_steps = max(
        (
            len(metric.series.median)
            for metric in metrics
            if metric.series is not None
        ),
        default=1 if metrics else 0,
    )
    resolved_steps = max(
        0,
        int(inferred_steps if steps_used is None else steps_used),
    )
    composition_representative = _composition_representative(
        rank_facts,
        composition,
    )
    return StepTimeWindow(
        clock=clock,
        training_strategy=strategy,
        expected_ranks=expected,
        coverage=StepTimeCoverage(
            expected_steps=resolved_steps,
            steps_used=resolved_steps,
            world_size=len(expected),
            ranks_present=len(rank_facts),
        ),
        rank_facts=rank_facts,
        metrics=metrics if isinstance(metrics, list) else list(metrics),
        iteration_ranks=carrying("input_wait", "step_time"),
        compute_ranks=carrying(
            "input_wait",
            "step_time",
            "forward",
            "backward",
            "optimizer_step",
        ),
        straggler_ranks=straggler,
        composition_representative_rank=composition_representative,
        input_wait_share=_component_share(rank_facts, "input_wait"),
        h2d_share=_component_share(rank_facts, "h2d"),
        compute_share=_component_share(rank_facts, "compute"),
        residual_share=_component_share(rank_facts, "residual_proxy"),
    )


def live_result_from_window(
    window: StepTimeWindow,
    *,
    freshness: LiveStepTimeFreshness = "live",
) -> LiveStepTimeResult:
    """Wrap one typed fixture window in the canonical live result shape."""
    request = StepTimeLoadRequest(
        window_size=max(1, int(window.coverage.expected_steps)),
        lookback_factor=4,
    )
    cursor_value = window.steps[-1] if window.steps else None
    snapshot = StepTimeRepositorySnapshot(
        cursor=StepTimeSourceCursor(
            last_row_id=cursor_value,
            latest_step=cursor_value,
        ),
        training_strategy=window.training_strategy,
    )
    return LiveStepTimeResult(
        freshness=freshness,
        analysis=StepTimeAnalysis(
            request=request,
            snapshot=snapshot,
            window=window,
            diagnosis=diagnose_step_time_window(
                window,
                policy=LIVE_STEP_TIME_POLICY,
            ),
        ),
    )


def _composition_representative(
    rank_facts: Sequence[StepTimeRankFacts],
    cohort: Sequence[int],
) -> Optional[int]:
    ranks = set(cohort)
    anchors = {
        facts.global_rank: (
            facts.average.total_step_ms
            if facts.average.total_step_ms is not None
            else facts.average.step_time_ms
        )
        for facts in rank_facts
        if facts.global_rank in ranks
    }
    clean = {
        rank: value for rank, value in anchors.items() if value is not None
    }
    if not clean:
        return None
    middle = float(statistics.median(clean.values()))
    return min(
        clean,
        key=lambda rank: (abs(float(clean[rank]) - middle), rank),
    )


def _component_share(
    rank_facts: Sequence[StepTimeRankFacts],
    component: str,
) -> Optional[float]:
    shares = []
    for facts in rank_facts:
        values = facts.average
        numerator = values.value(component)
        if (
            numerator is None
            or values.input_wait_ms is None
            or values.step_time_ms is None
        ):
            continue
        total = values.input_wait_ms + values.step_time_ms
        if total > 0.0:
            shares.append(float(numerator) / total)
    return float(statistics.median(shares)) if shares else None


__all__ = [
    "live_result_from_window",
    "rank_average",
    "values_from_mapping",
    "window_from_events",
    "window_from_rank_averages",
]
