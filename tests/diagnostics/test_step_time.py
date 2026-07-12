# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from traceml_ai.diagnostics.step_time.api import (
    DEFAULT_THRESHOLDS,
    build_step_diagnosis_result,
)
from traceml_ai.diagnostics.step_time.adapters import (
    DEFAULT_SUMMARY_DIAG_CONFIG,
    RankStepSignals,
    build_summary_step_diagnosis_result,
)
from traceml_ai.diagnostics.step_time.context import build_step_time_context
from traceml_ai.diagnostics.step_time.policy import (
    LIVE_STEP_TIME_POLICY,
    SUMMARY_STEP_TIME_POLICY,
)
from traceml_ai.diagnostics.step_time.rules import (
    CleanStragglerRule,
    ComputeBoundRule,
    InputBoundRule,
    ResidualHeavyRule,
)
from traceml_ai.renderers.step_time.schema import (
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


def _time_context(
    *metrics: StepCombinedTimeMetric,
    per_rank_timing: dict[int, dict[str, float]] | None = None,
):
    return build_step_time_context(
        metrics=metrics,
        thresholds=DEFAULT_THRESHOLDS,
        per_rank_timing=per_rank_timing,
    )


def _median(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _timing_row(
    *,
    dataloader: float = 10.0,
    h2d: float = 0.0,
    forward: float = 20.0,
    backward: float = 30.0,
    optimizer: float = 10.0,
    residual: float = 0.0,
    step_time: float | None = None,
    total_step: float | None = None,
) -> dict[str, float]:
    known_step = h2d + forward + backward + optimizer
    local_step = known_step + residual if step_time is None else step_time
    return {
        "dataloader_fetch": dataloader,
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


def _metrics_from_per_rank_timing(
    per_rank_timing: dict[int, dict[str, float]],
    *,
    steps: int = 64,
) -> tuple[StepCombinedTimeMetric, ...]:
    metrics: list[StepCombinedTimeMetric] = []
    world_size = len(per_rank_timing)
    for key in (
        "dataloader_fetch",
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
        median = _median(list(values_by_rank.values()))
        worst_rank = max(
            values_by_rank,
            key=lambda rank: (values_by_rank[rank], -rank),
        )
        worst = values_by_rank[worst_rank]
        skew = ((worst - median) / median) if median > 0.0 else 0.0
        metrics.append(
            _time_metric(
                key,
                median=median,
                worst=worst,
                worst_rank=worst_rank,
                skew=skew,
                world_size=world_size,
                steps=steps,
            )
        )
    return tuple(metrics)


def _clean_context(
    per_rank_timing: dict[int, dict[str, float]],
):
    return _time_context(
        *_metrics_from_per_rank_timing(per_rank_timing),
        per_rank_timing=per_rank_timing,
    )


def _single_rank_step_metrics(
    *,
    step: float = 100.0,
    dataloader: float = 5.0,
    forward: float = 30.0,
    backward: float = 50.0,
    optimizer: float = 10.0,
    residual: float = 5.0,
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
            "residual_proxy",
            median=residual,
            worst=residual,
            worst_rank=0,
            world_size=1,
        ),
    )


def test_step_time_rules_trigger_and_no_trigger_cases() -> None:
    input_ctx = _clean_context(
        {
            0: _timing_row(dataloader=10.0),
            1: _timing_row(dataloader=90.0),
        }
    )
    assert CleanStragglerRule().evaluate(input_ctx).kind == "INPUT_STRAGGLER"
    assert (
        CleanStragglerRule().evaluate(
            _time_context(*_single_rank_step_metrics())
        )
        is None
    )

    compute_ctx = _clean_context(
        {
            0: _timing_row(forward=20.0, backward=30.0),
            1: _timing_row(forward=90.0, backward=30.0),
        }
    )
    assert CleanStragglerRule().evaluate(compute_ctx).kind == (
        "COMPUTE_STRAGGLER"
    )

    input_bound = _time_context(
        *_single_rank_step_metrics(
            step=100.0,
            dataloader=35.0,
            forward=20.0,
            backward=30.0,
            optimizer=5.0,
            residual=10.0,
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
        ResidualHeavyRule()
        .evaluate(_time_context(*_single_rank_step_metrics(residual=20.0)))
        .kind
        == "RESIDUAL_HEAVY"
    )
    assert (
        ResidualHeavyRule().evaluate(
            _time_context(*_single_rank_step_metrics(residual=5.0))
        )
        is None
    )

    assert (
        ComputeBoundRule()
        .evaluate(
            _time_context(
                *_single_rank_step_metrics(dataloader=2.0, residual=3.0)
            )
        )
        .kind
        == "COMPUTE_BOUND"
    )
    assert (
        ComputeBoundRule().evaluate(
            _time_context(
                *_single_rank_step_metrics(dataloader=35.0, residual=3.0)
            )
        )
        is None
    )


def test_clean_backward_discount_removes_telescoped_delay_from_peer() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=100.0,
            forward=20.0,
            backward=20.0,
            optimizer=0.0,
            total_step=140.0,
        ),
        1: _timing_row(
            dataloader=0.0,
            forward=20.0,
            backward=120.0,
            optimizer=0.0,
            total_step=140.0,
        ),
    }
    ctx = _clean_context(per_rank)
    issue = CleanStragglerRule().evaluate(ctx)

    assert ctx.clean_rank_values["clean_backward"][1] == pytest.approx(20.0)
    assert ctx.clean_rank_values["clean_compute"][0] == pytest.approx(40.0)
    assert ctx.clean_rank_values["clean_compute"][1] == pytest.approx(40.0)
    assert ctx.clean_rank_values["clean_step"][0] == pytest.approx(140.0)
    assert issue is not None
    assert issue.kind == "INPUT_STRAGGLER"
    assert issue.ranks == (0,)
    assert issue.evidence["component_excesses_ms"]["input"] == pytest.approx(
        50.0
    )


@pytest.mark.parametrize(
    ("per_rank", "expected_kind", "expected_phase"),
    [
        (
            {
                0: _timing_row(forward=20.0),
                1: _timing_row(forward=100.0),
            },
            "COMPUTE_STRAGGLER",
            "compute",
        ),
        (
            {
                0: _timing_row(h2d=0.0),
                1: _timing_row(h2d=80.0),
            },
            "H2D_STRAGGLER",
            "h2d",
        ),
        (
            {
                0: _timing_row(residual=0.0),
                1: _timing_row(residual=80.0),
            },
            "RESIDUAL_STRAGGLER",
            "residual",
        ),
        (
            {
                0: _timing_row(dataloader=10.0, forward=20.0),
                1: _timing_row(dataloader=60.0, forward=40.0),
            },
            "STRAGGLER",
            "mixed",
        ),
    ],
)
def test_clean_straggler_classifies_component_excess(
    per_rank: dict[int, dict[str, float]],
    expected_kind: str,
    expected_phase: str,
) -> None:
    result = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(per_rank),
        per_rank_timing=per_rank,
    )

    assert result.primary.kind == expected_kind
    assert result.primary.worst_rank == 1
    assert result.issues[0].phase == expected_phase
    assert result.issues[0].evidence["clean_step_slack_ms"] > 0.0


def test_step_time_primary_prefers_clean_straggler_over_residual_heavy() -> (
    None
):
    per_rank = {
        0: _timing_row(dataloader=10.0, residual=80.0),
        1: _timing_row(dataloader=110.0, residual=80.0),
    }

    result = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(per_rank),
        per_rank_timing=per_rank,
    )

    assert result.primary.kind == "INPUT_STRAGGLER"
    assert {issue.kind for issue in result.issues} >= {
        "INPUT_STRAGGLER",
        "RESIDUAL_HEAVY",
    }


def test_step_time_live_and_summary_policies_are_explicit() -> None:
    assert LIVE_STEP_TIME_POLICY.name == "live"
    assert SUMMARY_STEP_TIME_POLICY.name == "summary"
    assert DEFAULT_THRESHOLDS == LIVE_STEP_TIME_POLICY.thresholds
    assert DEFAULT_SUMMARY_DIAG_CONFIG == SUMMARY_STEP_TIME_POLICY
    assert (
        SUMMARY_STEP_TIME_POLICY.thresholds.residual_share_warn
        > LIVE_STEP_TIME_POLICY.thresholds.residual_share_warn
    )
    assert (
        SUMMARY_STEP_TIME_POLICY.thresholds.straggler_dominance_tolerance
        == pytest.approx(1.25)
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
            h2d_ms=0.0,
            forward_ms=20.0,
            backward_ms=60.0,
            optimizer_ms=10.0,
            step_cpu_ms=100.0,
        )
    }

    warmup = build_summary_step_diagnosis_result(rank_signals, max_rows=100)
    assert warmup is not None
    assert warmup.primary.kind == "WARMUP"
    assert (
        warmup.primary.reason
        == "Only 40 steps per rank available; summary diagnosis requires 50."
    )

    rank_signals[0] = RankStepSignals(
        steps_analyzed=60,
        dataloader_ms=1.0,
        h2d_ms=0.0,
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
