from __future__ import annotations

from traceml.diagnostics.step_memory import (
    build_step_memory_summary_diagnosis_result,
)
from traceml.diagnostics.step_memory.adapters import (
    StepMemorySummaryMetricSignals,
    StepMemorySummaryTrendSignals,
)
from traceml.diagnostics.step_memory.rules import (
    CreepConfirmedRule,
    CreepEarlyRule,
    HighPressureRule,
    ImbalanceRule,
)
from traceml.diagnostics.step_time.api import (
    DEFAULT_THRESHOLDS,
    build_step_diagnosis_result,
)
from traceml.diagnostics.step_time.context import build_step_time_context
from traceml.diagnostics.step_time.rules import (
    ComputeBoundRule,
    ComputeStragglerRule,
    InputBoundRule,
    InputStragglerRule,
    WaitHeavyRule,
)
from traceml.renderers.step_memory.schema import (
    StepMemoryCombinedCoverage,
    StepMemoryCombinedMetric,
    StepMemoryCombinedSeries,
    StepMemoryCombinedSummary,
)
from traceml.renderers.step_time.schema import (
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeSeries,
    StepCombinedTimeSummary,
)


def _time_metric(
    name: str,
    *,
    median: float,
    worst: float,
    worst_rank: int | None = 1,
    skew: float = 0.0,
    world_size: int = 2,
    steps: int = 64,
) -> StepCombinedTimeMetric:
    return StepCombinedTimeMetric(
        metric=name,
        clock="mixed",
        series=StepCombinedTimeSeries(
            steps=list(range(steps)),
            median=[median] * steps,
            worst=[worst] * steps,
            sum=[median * world_size] * steps,
        ),
        summary=StepCombinedTimeSummary(
            window_size=steps,
            steps_used=steps,
            median_total=median,
            worst_total=worst,
            worst_rank=worst_rank,
            skew_ratio=skew,
            skew_pct=skew,
        ),
        coverage=StepCombinedTimeCoverage(
            expected_steps=steps,
            steps_used=steps,
            completed_step=steps,
            world_size=world_size,
            ranks_present=world_size,
            incomplete=False,
        ),
    )


def _time_context(*metrics: StepCombinedTimeMetric):
    return build_step_time_context(
        metrics=metrics,
        thresholds=DEFAULT_THRESHOLDS,
    )


def _single_rank_step_metrics(
    *,
    step: float = 100.0,
    dataloader: float = 5.0,
    forward: float = 30.0,
    backward: float = 50.0,
    optimizer: float = 10.0,
    wait: float = 5.0,
) -> tuple[StepCombinedTimeMetric, ...]:
    return (
        _time_metric(
            "step_time",
            median=step,
            worst=step,
            worst_rank=0,
            world_size=1,
        ),
        _time_metric(
            "dataloader_fetch",
            median=dataloader,
            worst=dataloader,
            worst_rank=0,
            world_size=1,
        ),
        _time_metric(
            "forward",
            median=forward,
            worst=forward,
            worst_rank=0,
            world_size=1,
        ),
        _time_metric(
            "backward",
            median=backward,
            worst=backward,
            worst_rank=0,
            world_size=1,
        ),
        _time_metric(
            "optimizer_step",
            median=optimizer,
            worst=optimizer,
            worst_rank=0,
            world_size=1,
        ),
        _time_metric(
            "wait_proxy",
            median=wait,
            worst=wait,
            worst_rank=0,
            world_size=1,
        ),
    )


def test_step_time_rules_trigger_and_no_trigger_cases() -> None:
    input_ctx = _time_context(
        _time_metric("step_time", median=220.0, worst=250.0),
        _time_metric("dataloader_fetch", median=10.0, worst=45.0, skew=0.2),
        _time_metric("forward", median=40.0, worst=40.0),
        _time_metric("backward", median=130.0, worst=130.0),
        _time_metric("optimizer_step", median=20.0, worst=20.0),
        _time_metric("wait_proxy", median=20.0, worst=20.0),
    )
    assert InputStragglerRule().evaluate(input_ctx).kind == "INPUT_STRAGGLER"
    assert (
        InputStragglerRule().evaluate(
            _time_context(*_single_rank_step_metrics())
        )
        is None
    )

    compute_ctx = _time_context(
        _time_metric("step_time", median=240.0, worst=310.0),
        _time_metric("dataloader_fetch", median=10.0, worst=10.0),
        _time_metric("forward", median=40.0, worst=90.0, skew=0.2),
        _time_metric("backward", median=130.0, worst=160.0, skew=0.15),
        _time_metric("optimizer_step", median=20.0, worst=20.0),
        _time_metric("wait_proxy", median=40.0, worst=40.0),
    )
    assert (
        ComputeStragglerRule().evaluate(compute_ctx).kind
        == "COMPUTE_STRAGGLER"
    )
    assert (
        ComputeStragglerRule().evaluate(
            _time_context(*_single_rank_step_metrics())
        )
        is None
    )

    assert (
        InputBoundRule()
        .evaluate(
            _time_context(
                *_single_rank_step_metrics(
                    step=100.0,
                    dataloader=35.0,
                    forward=20.0,
                    backward=30.0,
                    optimizer=5.0,
                    wait=10.0,
                )
            )
        )
        .kind
        == "INPUT_BOUND"
    )
    assert (
        InputBoundRule().evaluate(
            _time_context(*_single_rank_step_metrics(dataloader=10.0))
        )
        is None
    )

    assert (
        WaitHeavyRule()
        .evaluate(_time_context(*_single_rank_step_metrics(wait=20.0)))
        .kind
        == "WAIT_HEAVY"
    )
    assert (
        WaitHeavyRule().evaluate(
            _time_context(*_single_rank_step_metrics(wait=5.0))
        )
        is None
    )

    assert (
        ComputeBoundRule()
        .evaluate(
            _time_context(*_single_rank_step_metrics(dataloader=2.0, wait=3.0))
        )
        .kind
        == "COMPUTE_BOUND"
    )
    assert (
        ComputeBoundRule().evaluate(
            _time_context(
                *_single_rank_step_metrics(dataloader=35.0, wait=3.0)
            )
        )
        is None
    )


def test_step_time_primary_selection_combines_input_and_compute_stragglers() -> (
    None
):
    result = build_step_diagnosis_result(
        [
            _time_metric("step_time", median=240.0, worst=330.0),
            _time_metric(
                "dataloader_fetch", median=10.0, worst=45.0, skew=0.2
            ),
            _time_metric("forward", median=40.0, worst=90.0, skew=0.2),
            _time_metric("backward", median=130.0, worst=160.0, skew=0.15),
            _time_metric("optimizer_step", median=20.0, worst=20.0),
            _time_metric("wait_proxy", median=40.0, worst=40.0),
        ]
    )

    assert result.primary.kind == "STRAGGLER"
    assert {issue.kind for issue in result.issues} >= {
        "INPUT_STRAGGLER",
        "COMPUTE_STRAGGLER",
        "STRAGGLER",
    }


def _trend(
    *,
    early: bool = False,
    confirmed: bool = False,
) -> StepMemorySummaryTrendSignals:
    return StepMemorySummaryTrendSignals(
        eligible=True,
        baseline_avg_bytes=100.0,
        mid_avg_bytes=120.0,
        recent_avg_bytes=160.0,
        overall_abs_delta_bytes=60.0,
        overall_worst_growth_pct=0.6,
        overall_median_growth_pct=0.4,
        early=early,
        confirmed=confirmed,
        score=2.5,
    )


def _memory_signal(
    *,
    steps_used: int = 60,
    pressure_frac: float | None = 0.5,
    skew_pct: float = 0.0,
    trend: StepMemorySummaryTrendSignals | None = None,
) -> StepMemorySummaryMetricSignals:
    return StepMemorySummaryMetricSignals(
        metric="peak_reserved",
        device="cuda:0",
        steps_used=steps_used,
        window_size=steps_used,
        completed_step=steps_used,
        ranks_seen=2,
        worst_rank=1,
        worst_peak_bytes=90.0,
        median_peak_bytes=80.0,
        skew_ratio=skew_pct,
        skew_pct=skew_pct,
        pressure_frac=pressure_frac,
        trend=trend or _trend(),
    )


def _memory_metric(
    *,
    worst_peak: float = 90.0,
    median_peak: float = 80.0,
    steps_used: int = 60,
) -> StepMemoryCombinedMetric:
    return StepMemoryCombinedMetric(
        metric="peak_reserved",
        device="cuda:0",
        series=StepMemoryCombinedSeries(
            steps=list(range(steps_used)),
            median=[median_peak] * steps_used,
            worst=[worst_peak] * steps_used,
        ),
        summary=StepMemoryCombinedSummary(
            window_size=steps_used,
            steps_used=steps_used,
            median_peak=median_peak,
            worst_peak=worst_peak,
            worst_rank=1,
            skew_ratio=0.0,
            skew_pct=0.0,
        ),
        coverage=StepMemoryCombinedCoverage(
            expected_steps=steps_used,
            steps_used=steps_used,
            completed_step=steps_used,
            world_size=2,
            ranks_present=2,
            incomplete=False,
        ),
    )


def test_step_memory_rules_trigger_and_no_trigger_cases() -> None:
    assert (
        HighPressureRule().evaluate(_memory_signal(pressure_frac=0.93)).kind
        == "HIGH_PRESSURE"
    )
    assert (
        HighPressureRule().evaluate(_memory_signal(pressure_frac=0.2)) is None
    )

    assert (
        ImbalanceRule().evaluate(_memory_signal(skew_pct=0.3)).kind
        == "IMBALANCE"
    )
    assert ImbalanceRule().evaluate(_memory_signal(skew_pct=0.01)) is None

    assert (
        CreepConfirmedRule()
        .evaluate(_memory_signal(trend=_trend(confirmed=True)))
        .kind
        == "CREEP_CONFIRMED"
    )
    assert (
        CreepConfirmedRule().evaluate(_memory_signal(trend=_trend())) is None
    )

    assert (
        CreepEarlyRule()
        .evaluate(_memory_signal(trend=_trend(early=True)))
        .kind
        == "CREEP_EARLY"
    )
    assert (
        CreepEarlyRule().evaluate(
            _memory_signal(trend=_trend(early=True, confirmed=True))
        )
        is None
    )


def test_step_memory_primary_selection_uses_sorted_summary_issues() -> None:
    result = build_step_memory_summary_diagnosis_result(
        [_memory_metric(worst_peak=96.0, median_peak=80.0)],
        gpu_total_bytes=100.0,
    )

    assert result.primary.kind == "HIGH_PRESSURE"
    assert result.issues[0].kind == "HIGH_PRESSURE"
