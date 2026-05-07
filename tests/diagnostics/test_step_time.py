from __future__ import annotations

from traceml.diagnostics.step_time.api import (
    DEFAULT_THRESHOLDS,
    build_step_diagnosis_result,
)
from traceml.diagnostics.step_time.adapters import (
    DEFAULT_SUMMARY_DIAG_CONFIG,
    RankStepSignals,
    build_summary_step_diagnosis_result,
)
from traceml.diagnostics.step_time.context import build_step_time_context
from traceml.diagnostics.step_time.policy import (
    LIVE_STEP_TIME_POLICY,
    SUMMARY_STEP_TIME_POLICY,
)
from traceml.diagnostics.step_time.rules import (
    ComputeBoundRule,
    ComputeStragglerRule,
    InputBoundRule,
    InputStragglerRule,
    WaitHeavyRule,
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

    input_bound = _time_context(
        *_single_rank_step_metrics(
            step=100.0,
            dataloader=35.0,
            forward=20.0,
            backward=30.0,
            optimizer=5.0,
            wait=10.0,
        )
    )
    assert InputBoundRule().evaluate(input_bound).kind == "INPUT_BOUND"
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


def test_step_time_live_and_summary_policies_are_explicit() -> None:
    assert LIVE_STEP_TIME_POLICY.name == "live"
    assert SUMMARY_STEP_TIME_POLICY.name == "summary"
    assert DEFAULT_THRESHOLDS == LIVE_STEP_TIME_POLICY.thresholds
    assert DEFAULT_SUMMARY_DIAG_CONFIG == SUMMARY_STEP_TIME_POLICY
    assert (
        SUMMARY_STEP_TIME_POLICY.thresholds.wait_share_warn
        > LIVE_STEP_TIME_POLICY.thresholds.wait_share_warn
    )
    assert (
        SUMMARY_STEP_TIME_POLICY.min_steps_for_diag
        > LIVE_STEP_TIME_POLICY.min_steps_for_diag
    )


def test_summary_step_time_adapter_uses_summary_policy_by_default() -> None:
    rank_signals = {
        0: RankStepSignals(
            steps_analyzed=40,
            dataloader_ms=1.0,
            forward_ms=20.0,
            backward_ms=60.0,
            optimizer_ms=10.0,
            step_cpu_ms=100.0,
        )
    }

    assert (
        build_summary_step_diagnosis_result(rank_signals, max_rows=100) is None
    )

    rank_signals[0] = RankStepSignals(
        steps_analyzed=60,
        dataloader_ms=1.0,
        forward_ms=20.0,
        backward_ms=60.0,
        optimizer_ms=10.0,
        step_cpu_ms=100.0,
    )

    result = build_summary_step_diagnosis_result(
        rank_signals,
        max_rows=100,
    )

    assert result is not None
    assert result.primary.steps_used == 60
