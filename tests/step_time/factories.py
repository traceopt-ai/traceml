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
    SUMMARY_STEP_TIME_POLICY,
    diagnose_step_time_window,
)
from traceml_ai.diagnostics.step_time.api import DEFAULT_THRESHOLDS
from traceml_ai.diagnostics.step_time.context import build_step_time_context
from traceml_ai.diagnostics.step_time.policy import StepTimeDiagnosisPolicy
from traceml_ai.step_time.analysis import StepTimeAnalyzer
from traceml_ai.step_time.model import (
    DiagnosisClock,
    StepTimeCoverage,
    StepTimeLoadRequest,
    StepTimeMetric,
    StepTimeRankFacts,
    StepTimeRepositorySnapshot,
    StepTimeSeries,
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


def time_metric(
    name: str,
    *,
    median: float,
    worst: float,
    worst_rank: int | None = 1,
    skew: float = 0.0,
    steps: int = 64,
) -> StepTimeMetric:
    """Build one repeated aggregate metric series for diagnosis tests."""
    return StepTimeMetric(
        metric=name,
        series=StepTimeSeries(
            median=[median] * steps,
            worst=[worst] * steps,
        ),
        median_total=median,
        worst_total=worst,
        worst_rank=worst_rank,
        skew_pct=skew,
    )


def timing_row(
    *,
    dataloader: float = 10.0,
    h2d: float = 0.0,
    forward: float = 20.0,
    backward: float = 30.0,
    optimizer: float = 10.0,
    residual: float = 0.0,
    step_time: float | None = None,
    total_step: float | None = None,
    input_wait_cpu: float | None = None,
    input_wait_gpu: float | None = None,
    step_time_cpu: float | None = None,
    step_time_gpu: float | None = None,
) -> dict[str, float]:
    """Build one concise per-rank aggregate timing row."""
    known_step = h2d + forward + backward + optimizer
    local_step = known_step + residual if step_time is None else step_time
    row = {
        "input_wait": dataloader,
        "h2d": h2d,
        "forward": forward,
        "backward": backward,
        "optimizer_step": optimizer,
        "step_time": local_step,
        "residual_proxy": residual,
        "total_step": (
            dataloader + local_step if total_step is None else total_step
        ),
    }
    if input_wait_cpu is not None and step_time_cpu is not None:
        row.update(
            input_wait=input_wait_cpu,
            step_time=step_time_cpu,
            total_step=input_wait_cpu + step_time_cpu,
        )
    if input_wait_gpu is not None and step_time_gpu is not None:
        row.update(
            input_wait=input_wait_gpu,
            step_time=step_time_gpu,
            total_step=input_wait_gpu + step_time_gpu,
        )
    return row


def metrics_from_rank_timings(
    per_rank_timing: Mapping[int, Mapping[str, float]],
    *,
    steps: int = 64,
) -> tuple[StepTimeMetric, ...]:
    """Derive aggregate metric fixtures from concise per-rank rows."""
    metrics: list[StepTimeMetric] = []
    for key in (
        "input_wait",
        "h2d",
        "forward",
        "backward",
        "optimizer_step",
        "step_time",
        "residual_proxy",
    ):
        values_by_rank = {
            int(rank): float(values.get(key, 0.0))
            for rank, values in per_rank_timing.items()
        }
        median = float(statistics.median(values_by_rank.values()))
        worst_rank = max(
            values_by_rank,
            key=lambda rank: (values_by_rank[rank], -rank),
        )
        worst = values_by_rank[worst_rank]
        skew = ((worst - median) / median) if median > 0.0 else 0.0
        metrics.append(
            time_metric(
                key,
                median=median,
                worst=worst,
                worst_rank=worst_rank,
                skew=skew,
                steps=steps,
            )
        )
    return tuple(metrics)


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


def time_context(
    *metrics: StepTimeMetric,
    per_rank_timing: Mapping[int, Mapping[str, float]] | None = None,
    diagnosis_clock: str = "cpu",
):
    """Build a diagnostic context from aggregate test fixtures."""
    window = window_from_rank_averages(
        per_rank_timing or {},
        metrics=metrics,
        clock="gpu" if diagnosis_clock == "gpu" else "cpu",
    )
    return build_step_time_context(
        window=window,
        thresholds=DEFAULT_THRESHOLDS,
    )


def rank_context(
    per_rank_timing: Mapping[int, Mapping[str, float]],
    *,
    training_strategy: str = "ddp",
    diagnosis_clock: str = "cpu",
):
    """Build a complete context from concise per-rank timing rows."""
    window = window_from_rank_averages(
        per_rank_timing,
        metrics=metrics_from_rank_timings(per_rank_timing),
        clock="gpu" if diagnosis_clock == "gpu" else "cpu",
        training_strategy=training_strategy,
    )
    return build_step_time_context(
        window=window,
        thresholds=DEFAULT_THRESHOLDS,
    )


def diagnose_rank_map(
    metrics: tuple[StepTimeMetric, ...],
    thresholds=DEFAULT_THRESHOLDS,
    *,
    per_rank_timing: Mapping[int, Mapping[str, float]],
    diagnosis_clock: str = "cpu",
    training_strategy: str = "ddp",
):
    """Diagnose a concise typed fixture without a production map adapter."""
    window = window_from_rank_averages(
        per_rank_timing,
        metrics=metrics,
        clock="gpu" if diagnosis_clock == "gpu" else "cpu",
        training_strategy=training_strategy,
    )
    return diagnose_step_time_window(
        window,
        policy=StepTimeDiagnosisPolicy(
            name="test",
            thresholds=thresholds,
        ),
    )


def diagnose_summary_events(
    per_rank_steps: Mapping[int, Mapping[int, Any]],
    *,
    max_rows: int,
):
    """Run the summary policy over a concise raw-event fixture."""
    window = window_from_events(per_rank_steps, max_rows=max_rows)
    return diagnose_step_time_window(window, policy=SUMMARY_STEP_TIME_POLICY)


def single_rank_step_metrics(
    *,
    step: float = 100.0,
    dataloader: float = 5.0,
    forward: float = 30.0,
    backward: float = 50.0,
    optimizer: float = 10.0,
    residual: float = 5.0,
    steps: int = 64,
) -> tuple[StepTimeMetric, ...]:
    """Build the standard complete single-rank metric cohort."""
    values = {
        "step_time": step,
        "input_wait": dataloader,
        "forward": forward,
        "backward": backward,
        "optimizer_step": optimizer,
        "residual_proxy": residual,
    }
    return tuple(
        time_metric(
            name,
            median=value,
            worst=value,
            worst_rank=0,
            steps=steps,
        )
        for name, value in values.items()
    )


def event_stats(
    cpu_ms: float | None = None,
    *,
    gpu_ms: float | None = None,
) -> dict[str, dict[str, float | bool | int | None]]:
    """Build one normalized event/device statistics mapping."""
    device = "cuda:0" if gpu_ms is not None else "cpu"
    duration = gpu_ms if gpu_ms is not None else cpu_ms
    return {
        device: {
            "is_gpu": gpu_ms is not None,
            "duration_ms": duration,
            "cpu_ms": cpu_ms,
            "gpu_ms": gpu_ms,
            "n_calls": 1,
        }
    }


def summary_step_events(
    *,
    input_wait_gpu: float | None,
    h2d: float = 0.0,
    step_time_gpu: float = 60.0,
    steps: int = 60,
) -> dict[int, dict]:
    """Build a repeated complete phase window for summary diagnosis tests."""
    clock = "gpu" if input_wait_gpu is not None else "cpu"
    values = {
        "_traceml_internal:dataloader_next": (
            input_wait_gpu if input_wait_gpu is not None else 5.0
        ),
        "_traceml_internal:h2d_time": h2d,
        "_traceml_internal:forward_time": 20.0,
        "_traceml_internal:backward_time": 30.0,
        "_traceml_internal:optimizer_step": 10.0,
        "_traceml_internal:step_time": (
            step_time_gpu if clock == "gpu" else 60.0
        ),
    }
    return {
        step: {
            event: event_stats(**{f"{clock}_ms": value})
            for event, value in values.items()
        }
        for step in range(steps)
    }


def step_time_values(
    *,
    dataloader: float = 5.0,
    h2d: float = 0.0,
    forward: float = 30.0,
    backward: float = 50.0,
    optimizer: float = 10.0,
    step_cpu: float | None = None,
) -> StepTimeValues:
    """Build canonical values for summary-card fixtures."""
    compute = forward + backward + optimizer
    known_step = h2d + compute
    effective_step = max(
        step_cpu if step_cpu is not None else known_step,
        known_step,
    )
    residual = max(0.0, effective_step - known_step)
    return StepTimeValues(
        dataloader_cpu_ms=dataloader,
        input_wait_ms=dataloader,
        step_time_ms=effective_step,
        h2d_ms=h2d,
        forward_ms=forward,
        backward_ms=backward,
        optimizer_step_ms=optimizer,
        compute_ms=compute,
        residual_ms=residual,
        total_step_ms=dataloader + effective_step,
        total_step_cpu_ms=dataloader + effective_step,
    )


def step_events_from_values(
    values: StepTimeValues,
    *,
    steps: int = 64,
) -> dict[int, dict]:
    """Expand canonical rank values into a repeated CPU event window."""
    events = {
        "_traceml_internal:dataloader_next": values.dataloader_cpu_ms,
        "_traceml_internal:h2d_time": values.h2d_ms,
        "_traceml_internal:forward_time": values.forward_ms,
        "_traceml_internal:backward_time": values.backward_ms,
        "_traceml_internal:optimizer_step": values.optimizer_step_ms,
        "_traceml_internal:step_time": values.step_time_ms,
    }
    return {
        step: {
            event: event_stats(cpu_ms=value) for event, value in events.items()
        }
        for step in range(steps)
    }


def input_bound_step_events(
    *,
    dataloader_cpu: float = 12.0,
    step_time_cpu: float = 90.0,
    input_wait_gpu: float,
    step_time_gpu: float,
    h2d_gpu: float = 0.0,
    compute_gpu: float = 0.0,
    steps: int = 64,
) -> dict[int, dict]:
    """Build a repeated dual-clock window for bound-analysis tests."""
    events = {
        "_traceml_internal:dataloader_next": event_stats(
            cpu_ms=dataloader_cpu,
            gpu_ms=input_wait_gpu,
        ),
        "_traceml_internal:h2d_time": event_stats(gpu_ms=h2d_gpu),
        "_traceml_internal:forward_time": event_stats(gpu_ms=compute_gpu),
        "_traceml_internal:step_time": event_stats(
            cpu_ms=step_time_cpu,
            gpu_ms=step_time_gpu,
        ),
    }
    return {step: events for step in range(steps)}


def alternating_residual_step_events(steps: int = 64) -> dict[int, dict]:
    """Build the alternating residual-clamp regression fixture."""
    out: dict[int, dict] = {}
    for step in range(steps):
        compute_ms, step_time_ms = (
            (50.0, 100.0) if step % 2 == 0 else (100.0, 50.0)
        )
        out[step] = {
            "_traceml_internal:dataloader_next": event_stats(cpu_ms=0.0),
            "_traceml_internal:h2d_time": event_stats(cpu_ms=0.0),
            "_traceml_internal:forward_time": event_stats(cpu_ms=compute_ms),
            "_traceml_internal:backward_time": event_stats(cpu_ms=0.0),
            "_traceml_internal:optimizer_step": event_stats(cpu_ms=0.0),
            "_traceml_internal:step_time": event_stats(cpu_ms=step_time_ms),
        }
    return out


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
    "alternating_residual_step_events",
    "diagnose_rank_map",
    "diagnose_summary_events",
    "event_stats",
    "input_bound_step_events",
    "live_result_from_window",
    "metrics_from_rank_timings",
    "rank_average",
    "rank_context",
    "single_rank_step_metrics",
    "step_events_from_values",
    "step_time_values",
    "summary_step_events",
    "time_context",
    "time_metric",
    "timing_row",
    "values_from_mapping",
    "window_from_events",
    "window_from_rank_averages",
]
