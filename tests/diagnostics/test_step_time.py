# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import pytest

from traceml_ai.diagnostics.step_time.api import (
    DEFAULT_THRESHOLDS,
    StepDiagnosis,
    build_step_diagnosis_result,
)
from traceml_ai.diagnostics.step_time.context import build_step_time_context
from traceml_ai.diagnostics.step_time.formatters import format_cli_diagnosis
from traceml_ai.diagnostics.step_time.policy import SUMMARY_STEP_TIME_POLICY
from traceml_ai.diagnostics.step_time.rules import (
    ComputeBoundRule,
    H2DBoundRule,
    InputBoundRule,
    RankStragglerRule,
    ResidualHeavyRule,
)
from traceml_ai.renderers.step_time.schema import (
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeSeries,
    StepCombinedTimeSummary,
)
from traceml_ai.reporting.summaries.issue_summary import (
    diagnostic_result_to_json,
)
from traceml_ai.utils.step_time_window import (
    build_step_time_window_from_events,
    diagnose_step_time_window,
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
    diagnosis_clock: str = "cpu",
):
    return build_step_time_context(
        metrics=metrics,
        thresholds=DEFAULT_THRESHOLDS,
        per_rank_timing=per_rank_timing,
        diagnosis_clock=diagnosis_clock,
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
    input_wait_cpu: float | None = None,
    input_wait_gpu: float | None = None,
    step_time_cpu: float | None = None,
    step_time_gpu: float | None = None,
) -> dict[str, float]:
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
        row["input_wait"] = input_wait_cpu
        row["step_time"] = step_time_cpu
        row["total_step"] = input_wait_cpu + step_time_cpu
    if input_wait_gpu is not None and step_time_gpu is not None:
        row["input_wait"] = input_wait_gpu
        row["step_time"] = step_time_gpu
        row["total_step"] = input_wait_gpu + step_time_gpu
    return row


def _metrics_from_per_rank_timing(
    per_rank_timing: dict[int, dict[str, float]],
    *,
    steps: int = 64,
) -> tuple[StepCombinedTimeMetric, ...]:
    metrics: list[StepCombinedTimeMetric] = []
    world_size = len(per_rank_timing)
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


def _rank_context(
    per_rank_timing: dict[int, dict[str, float]],
    *,
    training_strategy: str = "ddp",
    diagnosis_clock: str = "cpu",
):
    return build_step_time_context(
        metrics=_metrics_from_per_rank_timing(per_rank_timing),
        thresholds=DEFAULT_THRESHOLDS,
        per_rank_timing=per_rank_timing,
        training_strategy=training_strategy,
        diagnosis_clock=diagnosis_clock,
    )


def _diagnose_summary_events(
    per_rank_steps: dict[int, dict[int, dict]],
    *,
    max_rows: int,
):
    window = build_step_time_window_from_events(
        per_rank_steps,
        max_rows=max_rows,
    )
    return diagnose_step_time_window(window, policy=SUMMARY_STEP_TIME_POLICY)


def _single_rank_step_metrics(
    *,
    step: float = 100.0,
    dataloader: float = 5.0,
    forward: float = 30.0,
    backward: float = 50.0,
    optimizer: float = 10.0,
    residual: float = 5.0,
    steps: int = 64,
) -> tuple[StepCombinedTimeMetric, ...]:
    return (
        _time_metric(
            "step_time",
            median=step,
            worst=step,
            worst_rank=0,
            world_size=1,
            steps=steps,
        ),
        _time_metric(
            "input_wait",
            median=dataloader,
            worst=dataloader,
            worst_rank=0,
            world_size=1,
            steps=steps,
        ),
        _time_metric(
            "forward",
            median=forward,
            worst=forward,
            worst_rank=0,
            world_size=1,
            steps=steps,
        ),
        _time_metric(
            "backward",
            median=backward,
            worst=backward,
            worst_rank=0,
            world_size=1,
            steps=steps,
        ),
        _time_metric(
            "optimizer_step",
            median=optimizer,
            worst=optimizer,
            worst_rank=0,
            world_size=1,
            steps=steps,
        ),
        _time_metric(
            "residual_proxy",
            median=residual,
            worst=residual,
            worst_rank=0,
            world_size=1,
            steps=steps,
        ),
    )


def test_diagnosis_clock_selection_prefers_gpu_then_cpu() -> None:
    events = {
        "_traceml_internal:dataloader_next": {
            "cuda:0": {
                "duration_ms": 12.0,
                "cpu_ms": 12.0,
                "gpu_ms": 4.0,
            }
        },
        "_traceml_internal:step_time": {
            "cuda:0": {
                "duration_ms": 60.0,
                "cpu_ms": 60.0,
                "gpu_ms": 20.0,
            }
        },
        "_traceml_internal:forward_time": {
            "cuda:0": {
                "duration_ms": 30.0,
                "cpu_ms": 30.0,
                "gpu_ms": 8.0,
            }
        },
        "_traceml_internal:backward_time": {
            "cuda:0": {
                "duration_ms": 20.0,
                "cpu_ms": 20.0,
                "gpu_ms": 7.0,
            }
        },
    }

    selected = build_step_time_window_from_events(
        {0: {1: events}},
        max_rows=1,
        expected_ranks=[0],
    )

    assert selected.clock == "gpu"
    assert selected.per_rank_timing[0]["input_wait"] == pytest.approx(4.0)
    assert selected.per_rank_timing[0]["step_time"] == pytest.approx(20.0)
    assert selected.per_rank_timing[0]["residual_proxy"] == pytest.approx(5.0)
    assert selected.per_rank_step_timing[0][1]["input_wait"] == pytest.approx(
        4.0
    )
    assert selected.per_rank_step_timing[0][1]["step_time"] == pytest.approx(
        20.0
    )

    events["_traceml_internal:dataloader_next"]["cuda:0"]["gpu_ms"] = None
    selected = build_step_time_window_from_events(
        {0: {1: events}},
        max_rows=1,
        expected_ranks=[0],
    )

    assert selected.clock == "cpu"
    assert selected.per_rank_timing[0]["input_wait"] == pytest.approx(12.0)
    assert selected.per_rank_timing[0]["step_time"] == pytest.approx(60.0)
    assert selected.per_rank_timing[0]["residual_proxy"] == pytest.approx(10.0)

    duration_only_events = {
        "_traceml_internal:dataloader_next": {"cpu": {"duration_ms": 12.0}},
        "_traceml_internal:step_time": {"cpu": {"duration_ms": 60.0}},
        "_traceml_internal:h2d_time": {"cpu": {"duration_ms": 1.0}},
        "_traceml_internal:forward_time": {"cpu": {"duration_ms": 20.0}},
        "_traceml_internal:backward_time": {"cpu": {"duration_ms": 30.0}},
        "_traceml_internal:optimizer_step": {"cpu": {"duration_ms": 5.0}},
    }
    selected = build_step_time_window_from_events(
        {0: {1: duration_only_events}},
        max_rows=1,
        expected_ranks=[0],
    )

    assert selected.clock == "cpu"
    assert selected.per_rank_timing[0]["input_wait"] == pytest.approx(0.0)
    assert selected.per_rank_timing[0]["step_time"] == pytest.approx(0.0)
    assert selected.per_rank_timing[0]["residual_proxy"] == pytest.approx(0.0)


def test_input_bound_rule_uses_cpu_clock_when_gpu_is_absent() -> None:
    ctx = _time_context(
        *_single_rank_step_metrics(step=100.0, dataloader=5.0),
        per_rank_timing={
            0: _timing_row(
                dataloader=5.0,
                input_wait_cpu=35.0,
                step_time_cpu=100.0,
            )
        },
    )

    issue = InputBoundRule().evaluate(ctx)

    assert issue is not None
    assert issue.kind == "INPUT_BOUND"
    assert issue.metric == "input_wait"
    assert issue.phase == "input"
    assert issue.share_pct == pytest.approx(35.0 / 135.0)
    assert issue.score == issue.share_pct
    assert issue.evidence["diagnosis_clock"] == "cpu"
    assert issue.evidence["input_wait_ms"] == pytest.approx(35.0)
    assert issue.evidence["step_time_ms"] == pytest.approx(100.0)
    assert issue.evidence["iteration_time_ms"] == pytest.approx(135.0)


def test_input_bound_rule_ignores_duration_without_explicit_clocks() -> None:
    ctx = _time_context(
        *_single_rank_step_metrics(step=100.0, dataloader=50.0)
    )

    assert InputBoundRule().evaluate(ctx) is None


def test_input_bound_uses_median_per_rank_iteration_share() -> None:
    ctx = _rank_context(
        {
            0: _timing_row(
                dataloader=10.0,
                input_wait_gpu=10.0,
                step_time_gpu=100.0,
            ),
            1: _timing_row(
                dataloader=10.0,
                input_wait_gpu=60.0,
                step_time_gpu=100.0,
            ),
        }
    )

    expected = ((10.0 / 110.0) + (60.0 / 160.0)) / 2.0

    assert ctx.input_bound_share == pytest.approx(expected)
    assert ctx.input_bound_share != pytest.approx(35.0 / 135.0)

    issue = InputBoundRule().evaluate(ctx)
    assert issue is not None
    assert issue.severity == "crit"
    assert issue.skew_pct is not None


@pytest.mark.parametrize(
    ("input_wait", "step_time", "expected_severity"),
    [(10.0, 90.0, "warn"), (20.0, 80.0, "crit")],
)
def test_input_bound_uses_iteration_share_thresholds(
    input_wait: float,
    step_time: float,
    expected_severity: str,
) -> None:
    per_rank = {
        0: _timing_row(
            dataloader=input_wait,
            step_time=step_time,
        )
    }

    issue = InputBoundRule().evaluate(_rank_context(per_rank))

    assert issue is not None
    assert issue.share_pct == pytest.approx(input_wait / 100.0)
    assert issue.severity == expected_severity


@pytest.mark.parametrize(
    ("h2d", "expected_severity"),
    [(10.0, "warn"), (20.0, "crit")],
)
def test_h2d_bound_uses_gpu_iteration_share_thresholds(
    h2d: float,
    expected_severity: str,
) -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            h2d=h2d,
            forward=0.0,
            backward=0.0,
            optimizer=0.0,
            step_time=100.0,
        )
    }

    issue = H2DBoundRule().evaluate(
        _rank_context(per_rank, diagnosis_clock="gpu")
    )

    assert issue is not None
    assert issue.metric == "h2d"
    assert issue.phase == "h2d"
    assert issue.share_pct == pytest.approx(h2d / 100.0)
    assert issue.score == issue.share_pct
    assert issue.severity == expected_severity


def test_h2d_bound_abstains_below_warning_threshold() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            h2d=9.0,
            forward=0.0,
            backward=0.0,
            optimizer=0.0,
            step_time=100.0,
        )
    }

    assert (
        H2DBoundRule().evaluate(_rank_context(per_rank, diagnosis_clock="gpu"))
        is None
    )


def test_h2d_bound_requires_gpu_selected_timing() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            h2d=80.0,
            forward=0.0,
            backward=0.0,
            optimizer=0.0,
            step_time=100.0,
        )
    }

    assert H2DBoundRule().evaluate(_rank_context(per_rank)) is None


def test_h2d_bound_keeps_skew_as_evidence() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            h2d=10.0,
            forward=0.0,
            backward=0.0,
            optimizer=0.0,
            step_time=100.0,
        ),
        1: _timing_row(
            dataloader=0.0,
            h2d=90.0,
            forward=0.0,
            backward=0.0,
            optimizer=0.0,
            step_time=100.0,
        ),
    }

    issue = H2DBoundRule().evaluate(
        _rank_context(per_rank, diagnosis_clock="gpu")
    )

    assert issue is not None
    assert issue.skew_pct is not None
    assert issue.ranks == (1,)


@pytest.mark.parametrize(
    ("residual", "expected_severity"),
    [(10.0, "warn"), (20.0, "crit")],
)
def test_residual_heavy_uses_iteration_share_thresholds(
    residual: float,
    expected_severity: str,
) -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            forward=0.0,
            backward=0.0,
            optimizer=0.0,
            residual=residual,
            step_time=100.0,
        )
    }

    issue = ResidualHeavyRule().evaluate(_rank_context(per_rank))

    assert issue is not None
    assert issue.share_pct == pytest.approx(residual / 100.0)
    assert issue.score == issue.share_pct
    assert issue.severity == expected_severity


def test_compute_bound_is_informational_despite_compute_skew() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            forward=90.0,
            backward=0.0,
            optimizer=0.0,
            step_time=100.0,
        ),
        1: _timing_row(
            dataloader=0.0,
            forward=150.0,
            backward=0.0,
            optimizer=0.0,
            step_time=160.0,
        ),
    }

    issue = ComputeBoundRule().evaluate(_rank_context(per_rank))

    assert issue is not None
    assert issue.severity == "info"
    assert issue.score is None
    assert issue.skew_pct is not None


def test_compute_bound_requires_existing_dominance_threshold() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            h2d=20.0,
            forward=80.0,
            backward=0.0,
            optimizer=0.0,
            step_time=100.0,
        )
    }

    assert ComputeBoundRule().evaluate(_rank_context(per_rank)) is None


def test_compute_bound_abstains_for_material_h2d() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            h2d=10.0,
            forward=90.0,
            backward=0.0,
            optimizer=0.0,
            step_time=100.0,
        )
    }

    assert (
        ComputeBoundRule().evaluate(
            _rank_context(per_rank, diagnosis_clock="gpu")
        )
        is None
    )


def test_cpu_h2d_does_not_suppress_compute_bound() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            h2d=80.0,
            forward=90.0,
            backward=0.0,
            optimizer=0.0,
            step_time=100.0,
        )
    }

    issue = ComputeBoundRule().evaluate(_rank_context(per_rank))

    assert issue is not None
    assert issue.kind == "COMPUTE_BOUND"


def test_input_bound_remains_primary_when_h2d_is_also_material() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=20.0,
            h2d=20.0,
            forward=0.0,
            backward=0.0,
            optimizer=0.0,
            step_time=100.0,
        )
    }

    result = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(per_rank),
        per_rank_timing=per_rank,
        diagnosis_clock="gpu",
    )

    assert result.primary.kind == "INPUT_BOUND"
    assert {issue.kind for issue in result.issues} >= {
        "INPUT_BOUND",
        "H2D_BOUND",
    }


def test_step_time_primary_orders_by_severity_before_impact() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=20.0,
            forward=0.0,
            backward=0.0,
            optimizer=0.0,
            residual=30.0,
            step_time=100.0,
        )
    }

    result = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(per_rank),
        per_rank_timing=per_rank,
    )

    assert result.primary.kind == "RESIDUAL_HEAVY"
    assert [issue.kind for issue in result.issues[:2]] == [
        "RESIDUAL_HEAVY",
        "INPUT_BOUND",
    ]
    assert result.issues[0].severity == "crit"
    assert result.issues[1].severity == "warn"


def test_step_time_primary_orders_equal_severity_by_impact() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=15.0,
            forward=0.0,
            backward=0.0,
            optimizer=0.0,
            residual=19.0,
            step_time=100.0,
        )
    }

    result = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(per_rank),
        per_rank_timing=per_rank,
    )

    assert result.primary.kind == "RESIDUAL_HEAVY"
    assert [issue.kind for issue in result.issues[:2]] == [
        "RESIDUAL_HEAVY",
        "INPUT_BOUND",
    ]
    assert result.issues[0].severity == result.issues[1].severity == "warn"
    assert result.issues[0].score > result.issues[1].score


def test_rank_straggler_wins_only_an_exact_impact_tie() -> None:
    tied = {
        0: _timing_row(
            dataloader=20.0,
            forward=0.0,
            backward=20.0,
            optimizer=0.0,
            step_time=100.0,
        ),
        1: _timing_row(
            dataloader=20.0,
            forward=0.0,
            backward=40.0,
            optimizer=0.0,
            step_time=100.0,
        ),
    }
    higher_typical = {
        0: _timing_row(
            dataloader=20.0,
            forward=0.0,
            backward=20.0,
            optimizer=0.0,
            step_time=100.0,
        ),
        1: _timing_row(
            dataloader=20.0,
            forward=0.0,
            backward=35.0,
            optimizer=0.0,
            step_time=100.0,
        ),
    }

    tied_result = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(tied),
        per_rank_timing=tied,
    )
    typical_result = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(higher_typical),
        per_rank_timing=higher_typical,
    )

    assert tied_result.primary.kind == "STRAGGLER"
    assert [issue.kind for issue in tied_result.issues[:2]] == [
        "STRAGGLER",
        "INPUT_BOUND",
    ]
    assert tied_result.issues[0].score == tied_result.issues[1].score

    assert typical_result.primary.kind == "INPUT_BOUND"
    assert [issue.kind for issue in typical_result.issues[:2]] == [
        "INPUT_BOUND",
        "STRAGGLER",
    ]
    assert typical_result.issues[0].score > typical_result.issues[1].score


def test_step_time_primary_uses_capped_severity_before_impact() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=20.0,
            forward=0.0,
            backward=0.0,
            optimizer=0.0,
            residual=30.0,
            step_time=100.0,
        )
    }

    result = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(per_rank, steps=5),
        per_rank_timing=per_rank,
    )
    diagnosis_json, issues_json = diagnostic_result_to_json(result)

    assert result.primary.kind == "RESIDUAL_HEAVY"
    assert all(issue.severity == "warn" for issue in result.issues)
    assert diagnosis_json == issues_json[0]


@pytest.mark.parametrize(
    ("severity", "style"),
    [("warn", "bold yellow"), ("crit", "bold red")],
)
def test_h2d_bound_cli_style_matches_severity(
    severity: Literal["warn", "crit"],
    style: str,
) -> None:
    diagnosis = StepDiagnosis(
        kind="H2D_BOUND",
        status="H2D-BOUND",
        severity=severity,
        reason="H2D transfer is material.",
        action="Inspect transfers.",
        steps_used=64,
    )

    assert f"[{style}]H2D-BOUND[/{style}]" in format_cli_diagnosis(diagnosis)


def test_compute_bound_is_secondary_to_rank_straggler() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            forward=90.0,
            backward=10.0,
            optimizer=0.0,
            step_time=100.0,
        ),
        1: _timing_row(
            dataloader=0.0,
            forward=90.0,
            backward=30.0,
            optimizer=0.0,
            step_time=120.0,
        ),
    }

    result = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(per_rank),
        per_rank_timing=per_rank,
    )

    assert result.primary.kind == "STRAGGLER"
    assert {issue.kind for issue in result.issues} >= {
        "STRAGGLER",
        "COMPUTE_BOUND",
    }
    compute_issue = next(
        issue for issue in result.issues if issue.kind == "COMPUTE_BOUND"
    )
    assert compute_issue.severity == "info"


@pytest.mark.parametrize(
    ("per_rank", "expected_kind", "expected_phase"),
    [
        (
            {
                0: _timing_row(
                    dataloader=100.0,
                    backward=20.0,
                    optimizer=0.0,
                ),
                1: _timing_row(
                    dataloader=0.0,
                    backward=120.0,
                    optimizer=0.0,
                ),
            },
            "INPUT_STRAGGLER",
            "input",
        ),
        (
            {
                0: _timing_row(h2d=80.0, backward=20.0, optimizer=0.0),
                1: _timing_row(h2d=0.0, backward=120.0, optimizer=0.0),
            },
            "H2D_STRAGGLER",
            "h2d",
        ),
        (
            {
                0: _timing_row(forward=100.0, backward=20.0, optimizer=0.0),
                1: _timing_row(forward=20.0, backward=120.0, optimizer=0.0),
            },
            "COMPUTE_STRAGGLER",
            "forward",
        ),
        (
            {
                0: _timing_row(forward=20.0, backward=20.0, optimizer=0.0),
                1: _timing_row(forward=20.0, backward=120.0, optimizer=0.0),
            },
            "STRAGGLER",
            "sync",
        ),
    ],
)
def test_rank_straggler_classifies_culprit_excess(
    per_rank: dict[int, dict[str, float]],
    expected_kind: str,
    expected_phase: str,
) -> None:
    result = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(per_rank),
        per_rank_timing=per_rank,
    )

    assert result.primary.kind == expected_kind
    assert result.primary.worst_rank == 0
    assert result.issues[0].phase == expected_phase
    assert result.issues[0].evidence["culprit_rank"] == 0
    assert result.issues[0].evidence["victim_rank"] == 1
    assert result.issues[0].evidence["visible_cost_ms"] > 0.0


@pytest.mark.parametrize(
    ("per_rank", "expected_kind", "expected_phase"),
    [
        (
            {
                0: _timing_row(
                    dataloader=100.0,
                    forward=20.0,
                    backward=20.0,
                    optimizer=0.0,
                ),
                1: _timing_row(
                    dataloader=0.0,
                    forward=80.0,
                    backward=80.0,
                    optimizer=0.0,
                ),
            },
            "INPUT_STRAGGLER",
            "input",
        ),
        (
            {
                0: _timing_row(
                    h2d=100.0,
                    forward=20.0,
                    backward=20.0,
                    optimizer=0.0,
                ),
                1: _timing_row(
                    h2d=0.0,
                    forward=80.0,
                    backward=80.0,
                    optimizer=0.0,
                ),
            },
            "H2D_STRAGGLER",
            "h2d",
        ),
        (
            {
                0: _timing_row(forward=100.0, backward=20.0, optimizer=0.0),
                1: _timing_row(forward=20.0, backward=200.0, optimizer=0.0),
            },
            "STRAGGLER",
            "sync",
        ),
    ],
)
def test_fsdp_rank_straggler_uses_input_h2d_or_unattributed(
    per_rank: dict[int, dict[str, float]],
    expected_kind: str,
    expected_phase: str,
) -> None:
    result = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(per_rank),
        per_rank_timing=per_rank,
        training_strategy="fsdp",
    )

    assert result.primary.kind == expected_kind
    assert result.primary.worst_rank == 0
    assert result.issues[0].phase == expected_phase
    if expected_kind == "STRAGGLER":
        assert result.issues[0].evidence["component"] == "sync_or_unattributed"


def test_rank_straggler_not_emitted_below_visible_wait_threshold() -> None:
    per_rank = {
        0: _timing_row(backward=115.0),
        1: _timing_row(backward=120.0),
    }
    ctx = _rank_context(per_rank)

    assert ctx.rank_straggler is None
    assert RankStragglerRule().evaluate(ctx) is None


@pytest.mark.parametrize(
    ("visible_cost", "expected_severity"),
    [(9.0, None), (10.0, "warn"), (20.0, "crit")],
)
def test_rank_straggler_uses_victim_iteration_impact_thresholds(
    visible_cost: float,
    expected_severity: str | None,
) -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            backward=10.0,
            step_time=100.0,
        ),
        1: _timing_row(
            dataloader=0.0,
            backward=10.0 + visible_cost,
            step_time=100.0,
        ),
    }

    issue = RankStragglerRule().evaluate(_rank_context(per_rank))

    if expected_severity is None:
        assert issue is None
        return

    assert issue is not None
    assert issue.kind == "STRAGGLER"
    assert issue.severity == expected_severity
    assert issue.score == pytest.approx(visible_cost / 100.0)


@pytest.mark.parametrize(
    ("component", "excess", "expected_kind"),
    [
        ("input", 79.0, "STRAGGLER"),
        ("input", 80.0, "INPUT_STRAGGLER"),
        ("h2d", 79.0, "STRAGGLER"),
        ("h2d", 80.0, "H2D_STRAGGLER"),
        ("compute", 79.0, "STRAGGLER"),
        ("compute", 80.0, "COMPUTE_STRAGGLER"),
    ],
)
def test_rank_straggler_requires_component_coverage_for_attribution(
    component: str,
    excess: float,
    expected_kind: str,
) -> None:
    culprit = _timing_row(
        dataloader=0.0,
        backward=20.0,
        step_time=100.0,
    )
    victim = _timing_row(
        dataloader=0.0,
        backward=120.0,
        step_time=100.0,
    )
    if component == "input":
        culprit["input_wait"] = excess
        victim["input_wait"] = 0.0
    elif component == "h2d":
        culprit["h2d"] = excess
        victim["h2d"] = 0.0
    else:
        culprit["forward"] = 20.0 + excess
        victim["forward"] = 20.0

    issue = RankStragglerRule().evaluate(
        _rank_context({0: culprit, 1: victim})
    )

    assert issue is not None
    assert issue.kind == expected_kind
    assert issue.severity == "crit"
    assert issue.score == pytest.approx(1.0)
    assert issue.evidence["component_excesses_ms"][component] == pytest.approx(
        excess
    )
    assert issue.evidence["component_coverage"][component] == pytest.approx(
        excess / 100.0
    )


def test_rank_straggler_coverage_changes_attribution_not_severity() -> None:
    def _issue(input_excess: float):
        culprit = _timing_row(
            dataloader=input_excess,
            backward=20.0,
            step_time=100.0,
        )
        victim = _timing_row(
            dataloader=0.0,
            backward=120.0,
            step_time=100.0,
        )
        issue = RankStragglerRule().evaluate(
            _rank_context({0: culprit, 1: victim})
        )
        assert issue is not None
        return issue

    generic = _issue(79.0)
    named = _issue(80.0)

    assert generic.kind == "STRAGGLER"
    assert named.kind == "INPUT_STRAGGLER"
    assert generic.score == named.score == pytest.approx(1.0)
    assert generic.severity == named.severity == "crit"


def test_rank_straggler_component_coverage_is_bounded() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            h2d=140.0,
            backward=20.0,
            step_time=100.0,
        ),
        1: _timing_row(
            dataloader=0.0,
            h2d=0.0,
            backward=120.0,
            step_time=100.0,
        ),
    }

    issue = RankStragglerRule().evaluate(_rank_context(per_rank))

    assert issue is not None
    assert issue.kind == "H2D_STRAGGLER"
    assert issue.evidence["component_excesses_ms"]["h2d"] == 140.0
    assert issue.evidence["component_coverage"]["h2d"] == 1.0


def test_rank_straggler_keeps_confidence_and_fsdp_severity_caps() -> None:
    ddp_per_rank = {
        0: _timing_row(
            dataloader=100.0,
            backward=20.0,
            step_time=100.0,
        ),
        1: _timing_row(
            dataloader=0.0,
            backward=120.0,
            step_time=100.0,
        ),
    }
    early = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(ddp_per_rank, steps=5),
        per_rank_timing=ddp_per_rank,
    )
    confident = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(ddp_per_rank, steps=20),
        per_rank_timing=ddp_per_rank,
    )

    assert early.primary.kind == "INPUT_STRAGGLER"
    assert early.primary.severity == "warn"
    assert confident.primary.kind == "INPUT_STRAGGLER"
    assert confident.primary.severity == "crit"

    fsdp_per_rank = {
        0: _timing_row(
            dataloader=100.0,
            forward=20.0,
            backward=20.0,
            step_time=100.0,
        ),
        1: _timing_row(
            dataloader=0.0,
            forward=80.0,
            backward=80.0,
            step_time=100.0,
        ),
    }
    fsdp = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(fsdp_per_rank, steps=20),
        per_rank_timing=fsdp_per_rank,
        training_strategy="fsdp",
    )

    assert fsdp.primary.kind == "INPUT_STRAGGLER"
    assert fsdp.primary.severity == "warn"


@pytest.mark.parametrize(
    ("training_strategy", "per_rank", "expected"),
    [
        (
            "ddp",
            {
                0: _timing_row(
                    dataloader=200.0,
                    backward=0.0,
                    step_time=200.0,
                ),
                1: _timing_row(
                    dataloader=80.0,
                    backward=20.0,
                    step_time=100.0,
                ),
                2: _timing_row(
                    dataloader=0.0,
                    backward=120.0,
                    step_time=140.0,
                ),
            },
            ("INPUT_STRAGGLER", 1, 2),
        ),
        (
            "ddp",
            {
                0: _timing_row(backward=0.0, step_time=100.0),
                1: _timing_row(backward=0.0, step_time=120.0),
            },
            None,
        ),
        (
            "ddp",
            {
                0: _timing_row(
                    dataloader=200.0,
                    backward=1.0,
                    step_time=0.0,
                ),
                1: _timing_row(
                    dataloader=80.0,
                    backward=20.0,
                    step_time=100.0,
                ),
                2: _timing_row(
                    dataloader=0.0,
                    backward=120.0,
                    step_time=140.0,
                ),
            },
            ("INPUT_STRAGGLER", 1, 2),
        ),
        (
            "fsdp",
            {
                0: _timing_row(dataloader=200.0, forward=0.0, backward=1.0),
                1: _timing_row(dataloader=160.0, forward=10.0, backward=0.0),
                2: _timing_row(dataloader=100.0, forward=20.0, backward=20.0),
                3: _timing_row(dataloader=0.0, forward=80.0, backward=80.0),
            },
            ("INPUT_STRAGGLER", 2, 3),
        ),
    ],
)
def test_rank_straggler_uses_only_valid_visible_ranks(
    training_strategy: str,
    per_rank: dict[int, dict[str, float]],
    expected: tuple[str, int, int] | None,
) -> None:
    ctx = _rank_context(per_rank, training_strategy=training_strategy)
    issue = RankStragglerRule().evaluate(ctx)

    if expected is None:
        assert ctx.rank_straggler is None
        assert issue is None
        return

    expected_kind, expected_culprit, expected_victim = expected
    assert ctx.rank_straggler is not None
    assert ctx.rank_straggler.culprit_rank == expected_culprit
    assert ctx.rank_straggler.victim_rank == expected_victim
    assert issue is not None
    assert issue.kind == expected_kind


def test_ddp_missing_forward_does_not_emit_compute_straggler() -> None:
    per_rank = {
        0: _timing_row(forward=100.0, backward=20.0, optimizer=0.0),
        1: _timing_row(forward=0.0, backward=120.0, optimizer=0.0),
    }
    result = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(per_rank),
        per_rank_timing=per_rank,
    )

    assert result.primary.kind == "STRAGGLER"
    assert result.issues[0].phase == "sync"
    assert result.issues[0].evidence["component"] == "sync_or_unattributed"


def test_rank_straggler_uses_actual_upper_median_victim_rank() -> None:
    per_rank = {
        0: _timing_row(backward=10.0, forward=10.0, optimizer=0.0),
        1: _timing_row(backward=20.0, forward=10.0, optimizer=0.0),
        2: _timing_row(backward=30.0, forward=10.0, optimizer=0.0),
        3: _timing_row(backward=40.0, forward=10.0, optimizer=0.0),
    }
    ctx = _rank_context(per_rank)

    assert ctx.rank_straggler is not None
    assert ctx.rank_straggler.culprit_rank == 0
    assert ctx.rank_straggler.victim_rank == 2
    assert ctx.rank_straggler.visible_victim_ms == pytest.approx(30.0)


def test_step_time_primary_prefers_rank_straggler_over_residual_heavy() -> (
    None
):
    per_rank = {
        0: _timing_row(dataloader=110.0, backward=20.0, residual=80.0),
        1: _timing_row(dataloader=10.0, backward=120.0, residual=80.0),
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


def test_step_time_early_warning_band_caps_severity() -> None:
    per_rank = {
        0: _timing_row(dataloader=50.0, step_time=100.0),
    }

    warmup = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(per_rank, steps=1),
        per_rank_timing=per_rank,
    )
    assert warmup.primary.kind == "WARMUP"
    assert (
        warmup.primary.reason
        == "Only 1 step per rank available; diagnosis requires 2."
    )

    warning = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(per_rank, steps=5),
        per_rank_timing=per_rank,
    )
    assert warning.primary.kind == "INPUT_BOUND"
    assert warning.primary.severity == "warn"
    assert warning.issues[0].severity == "warn"

    confident = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(per_rank, steps=20),
        per_rank_timing=per_rank,
    )
    assert confident.primary.kind == "INPUT_BOUND"
    assert confident.primary.severity == "crit"
    assert confident.issues[0].severity == "crit"

    fsdp = build_step_diagnosis_result(
        _metrics_from_per_rank_timing(per_rank, steps=20),
        per_rank_timing=per_rank,
        training_strategy="fsdp",
    )
    assert fsdp.primary.kind == "INPUT_BOUND"
    assert fsdp.primary.severity == "warn"
    assert all(issue.severity == "warn" for issue in fsdp.issues)


def test_summary_step_time_window_uses_summary_policy_by_default() -> None:
    short_window = _diagnose_summary_events(
        {0: _summary_step_events(input_wait_gpu=None, steps=40)},
        max_rows=100,
    )
    assert short_window is not None
    assert short_window.primary.kind == "COMPUTE_BOUND"
    assert short_window.primary.steps_used == 40

    result = _diagnose_summary_events(
        {0: _summary_step_events(input_wait_gpu=None, steps=60)},
        max_rows=100,
    )

    assert result is not None
    assert result.primary.steps_used == 60


def _event_stats(
    *,
    cpu_ms: float | None = None,
    gpu_ms: float | None = None,
) -> dict[str, dict[str, float | bool | int | None]]:
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


def _summary_step_events(
    *,
    input_wait_gpu: float | None,
    h2d: float = 0.0,
    step_time_gpu: float = 60.0,
    steps: int = 60,
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for step in range(steps):
        events = {
            "_traceml_internal:dataloader_next": _event_stats(cpu_ms=5.0),
            "_traceml_internal:h2d_time": _event_stats(cpu_ms=h2d),
            "_traceml_internal:forward_time": _event_stats(cpu_ms=20.0),
            "_traceml_internal:backward_time": _event_stats(cpu_ms=30.0),
            "_traceml_internal:optimizer_step": _event_stats(cpu_ms=10.0),
            "_traceml_internal:step_time": _event_stats(cpu_ms=60.0),
        }
        if input_wait_gpu is not None:
            events = {
                "_traceml_internal:dataloader_next": _event_stats(
                    gpu_ms=input_wait_gpu
                ),
                "_traceml_internal:h2d_time": _event_stats(gpu_ms=h2d),
                "_traceml_internal:forward_time": _event_stats(gpu_ms=20.0),
                "_traceml_internal:backward_time": _event_stats(gpu_ms=30.0),
                "_traceml_internal:optimizer_step": _event_stats(gpu_ms=10.0),
                "_traceml_internal:step_time": _event_stats(
                    gpu_ms=step_time_gpu
                ),
            }
        out[step] = events
    return out


def test_summary_input_bound_uses_explicit_input_clocks() -> None:
    low_wait = _diagnose_summary_events(
        {0: _summary_step_events(input_wait_gpu=5.0)},
        max_rows=100,
    )
    high_wait = _diagnose_summary_events(
        {0: _summary_step_events(input_wait_gpu=25.0)},
        max_rows=100,
    )

    assert low_wait.primary.kind != "INPUT_BOUND"
    assert high_wait.primary.kind == "INPUT_BOUND"
    assert high_wait.issues[0].evidence["diagnosis_clock"] == "gpu"
    assert high_wait.issues[0].evidence["input_wait_ms"] == pytest.approx(25.0)
    assert high_wait.issues[0].evidence["iteration_time_ms"] == pytest.approx(
        85.0
    )


def test_summary_h2d_bound_uses_gpu_selected_h2d_timing() -> None:
    result = _diagnose_summary_events(
        {0: _summary_step_events(input_wait_gpu=0.0, h2d=12.0)},
        max_rows=100,
    )

    assert result.primary.kind == "H2D_BOUND"
    issue = next(issue for issue in result.issues if issue.kind == "H2D_BOUND")
    assert issue.severity == "crit"
    assert issue.evidence["diagnosis_clock"] == "gpu"
    assert issue.share_pct == pytest.approx(12.0 / 60.0)


def test_summary_h2d_bound_ignores_cpu_selected_h2d_timing() -> None:
    result = _diagnose_summary_events(
        {0: _summary_step_events(input_wait_gpu=None, h2d=80.0)},
        max_rows=100,
    )

    assert all(issue.kind != "H2D_BOUND" for issue in result.issues)


def test_summary_input_bound_trend_uses_selected_input_wait_series() -> None:
    steps = 240
    per_step: dict[int, dict] = {}
    for step in range(steps):
        input_wait = 10.0 + step * (80.0 / float(steps - 1))
        per_step[step] = {
            "_traceml_internal:dataloader_next": _event_stats(
                gpu_ms=input_wait
            ),
            "_traceml_internal:h2d_time": _event_stats(gpu_ms=0.0),
            "_traceml_internal:forward_time": _event_stats(gpu_ms=20.0),
            "_traceml_internal:backward_time": _event_stats(gpu_ms=30.0),
            "_traceml_internal:optimizer_step": _event_stats(gpu_ms=10.0),
            "_traceml_internal:step_time": _event_stats(gpu_ms=60.0),
        }

    result = _diagnose_summary_events(
        {0: per_step},
        max_rows=steps,
    )

    assert result.primary.kind == "INPUT_BOUND"
    assert result.primary.note is not None
    assert result.primary.note.startswith("Trend: input wait is ")
    assert "dataloader" not in result.primary.note
