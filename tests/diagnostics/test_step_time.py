# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import pytest

from traceml_ai.diagnostics.step_time.api import (
    DEFAULT_THRESHOLDS,
    StepDiagnosis,
    diagnose_step_time_window,
)
from traceml_ai.diagnostics.step_time.policy import (
    LIVE_STEP_TIME_POLICY,
    SUMMARY_STEP_TIME_POLICY,
)
from traceml_ai.diagnostics.step_time.rules import (
    ComputeBoundRule,
    H2DBoundRule,
    InputBoundRule,
    RankStragglerRule,
    ResidualHeavyRule,
)
from traceml_ai.diagnostics.step_time.trend import build_step_trend_note
from traceml_ai.renderers.step_time.renderer import format_cli_diagnosis
from traceml_ai.step_time.model import (
    StepTimeSeries,
    StepTimeValues,
)
from traceml_ai.reporting.summaries.issue_summary import (
    diagnostic_result_to_json,
)
from tests.step_time.factories import (
    diagnose_rank_map as _diagnose_rank_map,
    diagnose_summary_events as _diagnose_summary_events,
    event_stats as _event_stats,
    metrics_from_rank_timings as _metrics_from_per_rank_timing,
    rank_average,
    rank_context as _rank_context,
    single_rank_step_metrics as _single_rank_step_metrics,
    summary_step_events as _summary_step_events,
    time_context as _time_context,
    time_metric as _time_metric,
    timing_row as _timing_row,
    window_from_events,
    window_from_rank_averages,
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

    selected = window_from_events(
        {0: {1: events}},
        max_rows=1,
        expected_ranks=[0],
    )

    assert selected.clock == "gpu"
    average = rank_average(selected, 0)
    assert average.input_wait_ms == pytest.approx(4.0)
    assert average.traced_step_time_ms == pytest.approx(20.0)
    assert average.step_time_ms == pytest.approx(24.0)
    # optimizer_step and h2d were never measured: the residual must not
    # silently absorb them as zeros.
    assert average.residual_ms is None
    assert average.optimizer_step_ms is None
    rank_facts = selected.rank(0)
    assert rank_facts is not None
    step_values = rank_facts.steps[0].values
    assert step_values.input_wait_ms == pytest.approx(4.0)
    assert step_values.traced_step_time_ms == pytest.approx(20.0)
    assert step_values.step_time_ms == pytest.approx(24.0)

    events["_traceml_internal:dataloader_next"]["cuda:0"]["gpu_ms"] = None
    selected = window_from_events(
        {0: {1: events}},
        max_rows=1,
        expected_ranks=[0],
    )

    assert selected.clock == "cpu"
    average = rank_average(selected, 0)
    assert average.input_wait_ms == pytest.approx(12.0)
    assert average.traced_step_time_ms == pytest.approx(60.0)
    assert average.step_time_ms == pytest.approx(72.0)
    # optimizer_step was never measured, so compute and residual stay
    # underivable on the cpu clock as well.
    assert average.residual_ms is None

    duration_only_events = {
        "_traceml_internal:dataloader_next": {"cpu": {"duration_ms": 12.0}},
        "_traceml_internal:step_time": {"cpu": {"duration_ms": 60.0}},
        "_traceml_internal:h2d_time": {"cpu": {"duration_ms": 1.0}},
        "_traceml_internal:forward_time": {"cpu": {"duration_ms": 20.0}},
        "_traceml_internal:backward_time": {"cpu": {"duration_ms": 30.0}},
        "_traceml_internal:optimizer_step": {"cpu": {"duration_ms": 5.0}},
    }
    selected = window_from_events(
        {0: {1: duration_only_events}},
        max_rows=1,
        expected_ranks=[0],
    )

    assert selected.clock == "cpu"
    # duration-only stats carry no selected-clock timing at all: every
    # metric is missing, not measured-as-zero.
    assert rank_average(selected, 0) == StepTimeValues()
    assert selected.metrics == []


def test_input_bound_rule_uses_cpu_clock_when_gpu_is_absent() -> None:
    ctx = _time_context(
        *_single_rank_step_metrics(step=100.0, dataloader=5.0),
        per_rank_timing={
            0: _timing_row(
                dataloader=5.0,
                input_wait_cpu=35.0,
                traced_step_time_cpu=100.0,
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
    assert issue.evidence["traced_step_time_ms"] == pytest.approx(100.0)
    assert issue.evidence["step_time_ms"] == pytest.approx(135.0)


def test_input_bound_rule_ignores_duration_without_explicit_clocks() -> None:
    ctx = _time_context(
        *_single_rank_step_metrics(step=100.0, dataloader=50.0)
    )

    assert InputBoundRule().evaluate(ctx) is None


def test_input_bound_uses_median_per_rank_step_time_share() -> None:
    ctx = _rank_context(
        {
            0: _timing_row(
                dataloader=10.0,
                input_wait_gpu=10.0,
                traced_step_time_gpu=100.0,
            ),
            1: _timing_row(
                dataloader=10.0,
                input_wait_gpu=60.0,
                traced_step_time_gpu=100.0,
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
    ("input_wait", "traced_step_time", "expected_severity"),
    [(10.0, 90.0, "warn"), (20.0, 80.0, "crit")],
)
def test_input_bound_uses_step_time_share_thresholds(
    input_wait: float,
    traced_step_time: float,
    expected_severity: str,
) -> None:
    per_rank = {
        0: _timing_row(
            dataloader=input_wait,
            traced_step_time=traced_step_time,
        )
    }

    issue = InputBoundRule().evaluate(_rank_context(per_rank))

    assert issue is not None
    assert issue.share_pct == pytest.approx(input_wait / 100.0)
    assert issue.severity == expected_severity


@pytest.mark.parametrize(
    ("h2d_by_rank", "clock", "expected_severity", "expected_share"),
    [
        pytest.param((10.0,), "gpu", "warn", 0.10, id="warning"),
        pytest.param((20.0,), "gpu", "crit", 0.20, id="critical"),
        pytest.param((9.0,), "gpu", None, None, id="below-threshold"),
        pytest.param((80.0,), "cpu", None, None, id="cpu-clock"),
        pytest.param((10.0, 90.0), "gpu", "crit", 0.50, id="skew"),
    ],
)
def test_h2d_bound_conditions(
    h2d_by_rank: tuple[float, ...],
    clock: str,
    expected_severity: str | None,
    expected_share: float | None,
) -> None:
    per_rank = {
        rank: _timing_row(
            dataloader=0.0,
            h2d=h2d,
            forward=0.0,
            backward=0.0,
            optimizer=0.0,
            traced_step_time=100.0,
        )
        for rank, h2d in enumerate(h2d_by_rank)
    }

    issue = H2DBoundRule().evaluate(
        _rank_context(per_rank, diagnosis_clock=clock)
    )

    if expected_severity is None:
        assert issue is None
        return

    assert issue is not None
    assert issue.metric == "h2d"
    assert issue.phase == "h2d"
    assert issue.share_pct == pytest.approx(expected_share)
    assert issue.score == issue.share_pct
    assert issue.severity == expected_severity
    if len(h2d_by_rank) > 1:
        assert issue.skew_pct is not None
        assert issue.ranks == (1,)


@pytest.mark.parametrize(
    ("residual", "expected_severity"),
    [(10.0, "warn"), (20.0, "crit")],
)
def test_residual_heavy_uses_step_time_share_thresholds(
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
            traced_step_time=100.0,
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
            traced_step_time=100.0,
        ),
        1: _timing_row(
            dataloader=0.0,
            forward=150.0,
            backward=0.0,
            optimizer=0.0,
            traced_step_time=160.0,
        ),
    }

    issue = ComputeBoundRule().evaluate(_rank_context(per_rank))

    assert issue is not None
    assert issue.severity == "info"
    assert issue.score is None
    assert issue.skew_pct is not None


@pytest.mark.parametrize(
    ("compute", "expected"),
    [(89.9, False), (90.0, True)],
)
def test_compute_bound_uses_step_time_share_threshold(
    compute: float,
    expected: bool,
) -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            forward=compute,
            backward=0.0,
            optimizer=0.0,
            traced_step_time=100.0,
        )
    }

    issue = ComputeBoundRule().evaluate(_rank_context(per_rank))

    assert (issue is not None) is expected
    if issue is not None:
        assert issue.share_pct == pytest.approx(0.90)
        assert issue.score is None


def test_compute_share_is_median_of_per_rank_step_time_shares() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            forward=100.0,
            backward=0.0,
            optimizer=0.0,
            traced_step_time=100.0,
        ),
        1: _timing_row(
            dataloader=100.0,
            forward=80.0,
            backward=0.0,
            optimizer=0.0,
            traced_step_time=100.0,
        ),
    }

    context = _rank_context(per_rank)

    assert context.compute_share == pytest.approx(0.70)
    aggregate_median_ratio = 90.0 / (50.0 + 100.0)
    assert context.compute_share != pytest.approx(aggregate_median_ratio)


@pytest.mark.parametrize(
    ("overrides", "clock", "expected_share"),
    [
        pytest.param(
            {"h2d": 20.0, "forward": 80.0},
            "cpu",
            None,
            id="below-dominance-threshold",
        ),
        pytest.param(
            {"h2d": 10.0, "forward": 90.0},
            "gpu",
            0.90,
            id="material-h2d",
        ),
        pytest.param(
            {
                "dataloader": 10.0,
                "forward": 90.0,
                "traced_step_time": 90.0,
            },
            "cpu",
            0.90,
            id="material-input-wait",
        ),
        pytest.param(
            {
                "forward": 90.0,
                "residual": 10.0,
                "traced_step_time": 100.0,
            },
            "cpu",
            0.90,
            id="material-residual",
        ),
    ],
)
def test_compute_bound_abstains_for_competing_costs(
    overrides: dict[str, float],
    clock: str,
    expected_share: float | None,
) -> None:
    row = {
        "dataloader": 0.0,
        "h2d": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
        "traced_step_time": 100.0,
        **overrides,
    }
    per_rank = {0: _timing_row(**row)}
    context = _rank_context(per_rank, diagnosis_clock=clock)

    if expected_share is not None:
        assert context.compute_share == pytest.approx(expected_share)
    assert ComputeBoundRule().evaluate(context) is None


def test_cpu_h2d_does_not_suppress_compute_bound() -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            h2d=80.0,
            forward=90.0,
            backward=0.0,
            optimizer=0.0,
            traced_step_time=100.0,
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
            traced_step_time=100.0,
        )
    }

    result = _diagnose_rank_map(
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
            traced_step_time=100.0,
        )
    }

    result = _diagnose_rank_map(
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
            traced_step_time=100.0,
        )
    }

    result = _diagnose_rank_map(
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
            traced_step_time=100.0,
        ),
        1: _timing_row(
            dataloader=20.0,
            forward=0.0,
            backward=40.0,
            optimizer=0.0,
            traced_step_time=100.0,
        ),
    }
    higher_typical = {
        0: _timing_row(
            dataloader=20.0,
            forward=0.0,
            backward=20.0,
            optimizer=0.0,
            traced_step_time=100.0,
        ),
        1: _timing_row(
            dataloader=20.0,
            forward=0.0,
            backward=35.0,
            optimizer=0.0,
            traced_step_time=100.0,
        ),
    }

    tied_result = _diagnose_rank_map(
        _metrics_from_per_rank_timing(tied),
        per_rank_timing=tied,
    )
    typical_result = _diagnose_rank_map(
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
            traced_step_time=100.0,
        )
    }

    result = _diagnose_rank_map(
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
            traced_step_time=100.0,
        ),
        1: _timing_row(
            dataloader=0.0,
            forward=90.0,
            backward=30.0,
            optimizer=0.0,
            traced_step_time=120.0,
        ),
    }

    result = _diagnose_rank_map(
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
    ("per_rank", "expected_kind", "expected_phase", "expected_component"),
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
            None,
        ),
        (
            {
                0: _timing_row(h2d=80.0, backward=20.0, optimizer=0.0),
                1: _timing_row(h2d=0.0, backward=120.0, optimizer=0.0),
            },
            "H2D_STRAGGLER",
            "h2d",
            None,
        ),
        (
            {
                0: _timing_row(forward=100.0, backward=20.0, optimizer=0.0),
                1: _timing_row(forward=20.0, backward=120.0, optimizer=0.0),
            },
            "COMPUTE_STRAGGLER",
            "forward",
            None,
        ),
        (
            {
                0: _timing_row(forward=20.0, backward=20.0, optimizer=0.0),
                1: _timing_row(forward=20.0, backward=120.0, optimizer=0.0),
            },
            "STRAGGLER",
            "sync",
            "sync_or_unattributed",
        ),
        (
            {
                0: _timing_row(
                    forward=100.0,
                    backward=20.0,
                    optimizer=0.0,
                ),
                1: _timing_row(
                    forward=0.0,
                    backward=120.0,
                    optimizer=0.0,
                ),
            },
            "STRAGGLER",
            "sync",
            "sync_or_unattributed",
        ),
    ],
)
def test_rank_straggler_classifies_culprit_excess(
    per_rank: dict[int, dict[str, float]],
    expected_kind: str,
    expected_phase: str,
    expected_component: str | None,
) -> None:
    result = _diagnose_rank_map(
        _metrics_from_per_rank_timing(per_rank),
        per_rank_timing=per_rank,
    )

    assert result.primary.kind == expected_kind
    assert result.primary.worst_rank == 0
    assert result.issues[0].phase == expected_phase
    assert result.issues[0].evidence["culprit_rank"] == 0
    assert result.issues[0].evidence["victim_rank"] == 1
    assert result.issues[0].evidence["visible_cost_ms"] > 0.0
    if expected_component is not None:
        assert result.issues[0].evidence["component"] == expected_component


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
    result = _diagnose_rank_map(
        _metrics_from_per_rank_timing(per_rank),
        per_rank_timing=per_rank,
        training_strategy="fsdp",
    )

    assert result.primary.kind == expected_kind
    assert result.primary.worst_rank == 0
    assert result.issues[0].phase == expected_phase
    if expected_kind == "STRAGGLER":
        assert result.issues[0].evidence["component"] == "sync_or_unattributed"


@pytest.mark.parametrize(
    ("visible_cost", "expected_severity"),
    [(5.0, None), (9.0, None), (10.0, "warn"), (20.0, "crit")],
)
def test_rank_straggler_uses_victim_step_time_impact_thresholds(
    visible_cost: float,
    expected_severity: str | None,
) -> None:
    per_rank = {
        0: _timing_row(
            dataloader=0.0,
            backward=10.0,
            traced_step_time=100.0,
        ),
        1: _timing_row(
            dataloader=0.0,
            backward=10.0 + visible_cost,
            traced_step_time=100.0,
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
    assert issue.evidence["step_time_ms"] == pytest.approx(100.0)
    assert "iteration_time_ms" not in issue.evidence


@pytest.mark.parametrize(
    ("component", "excess", "expected_kind", "expected_coverage"),
    [
        ("input", 79.0, "STRAGGLER", 0.79),
        ("input", 80.0, "INPUT_STRAGGLER", 0.80),
        ("h2d", 79.0, "STRAGGLER", 0.79),
        ("h2d", 80.0, "H2D_STRAGGLER", 0.80),
        ("h2d", 140.0, "H2D_STRAGGLER", 1.00),
        ("compute", 79.0, "STRAGGLER", 0.79),
        ("compute", 80.0, "COMPUTE_STRAGGLER", 0.80),
    ],
)
def test_rank_straggler_requires_component_coverage_for_attribution(
    component: str,
    excess: float,
    expected_kind: str,
    expected_coverage: float,
) -> None:
    culprit = _timing_row(
        dataloader=0.0,
        backward=20.0,
        traced_step_time=100.0,
    )
    victim = _timing_row(
        dataloader=0.0,
        backward=120.0,
        traced_step_time=100.0,
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
        expected_coverage
    )
    if component == "input":
        expected_summary = (
            "r0 has excess input wait burden relative to victim r1"
            if expected_kind == "INPUT_STRAGGLER"
            else "r0 is slower than victim r1"
        )
        assert expected_summary in issue.summary
        if expected_kind == "INPUT_STRAGGLER":
            assert "~80.0% of visible wait cost" in issue.summary


def test_rank_straggler_keeps_confidence_and_fsdp_severity_caps() -> None:
    ddp_per_rank = {
        0: _timing_row(
            dataloader=100.0,
            backward=20.0,
            traced_step_time=100.0,
        ),
        1: _timing_row(
            dataloader=0.0,
            backward=120.0,
            traced_step_time=100.0,
        ),
    }
    early = _diagnose_rank_map(
        _metrics_from_per_rank_timing(ddp_per_rank, steps=5),
        per_rank_timing=ddp_per_rank,
    )
    confident = _diagnose_rank_map(
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
            traced_step_time=100.0,
        ),
        1: _timing_row(
            dataloader=0.0,
            forward=80.0,
            backward=80.0,
            traced_step_time=100.0,
        ),
    }
    fsdp = _diagnose_rank_map(
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
                    traced_step_time=200.0,
                ),
                1: _timing_row(
                    dataloader=80.0,
                    backward=20.0,
                    traced_step_time=100.0,
                ),
                2: _timing_row(
                    dataloader=0.0,
                    backward=120.0,
                    traced_step_time=140.0,
                ),
            },
            ("INPUT_STRAGGLER", 1, 2),
        ),
        (
            "ddp",
            {
                0: _timing_row(backward=0.0, traced_step_time=100.0),
                1: _timing_row(backward=0.0, traced_step_time=120.0),
            },
            None,
        ),
        (
            "ddp",
            {
                0: _timing_row(
                    dataloader=200.0,
                    backward=1.0,
                    traced_step_time=0.0,
                ),
                1: _timing_row(
                    dataloader=80.0,
                    backward=20.0,
                    traced_step_time=100.0,
                ),
                2: _timing_row(
                    dataloader=0.0,
                    backward=120.0,
                    traced_step_time=140.0,
                ),
            },
            ("INPUT_STRAGGLER", 0, 1),
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

    result = _diagnose_rank_map(
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
        0: _timing_row(dataloader=50.0, traced_step_time=100.0),
    }

    warmup = _diagnose_rank_map(
        _metrics_from_per_rank_timing(per_rank, steps=1),
        per_rank_timing=per_rank,
    )
    assert warmup.primary.kind == "WARMUP"
    assert (
        warmup.primary.reason
        == "Only 1 step per rank available; diagnosis requires 2."
    )

    warning = _diagnose_rank_map(
        _metrics_from_per_rank_timing(per_rank, steps=5),
        per_rank_timing=per_rank,
    )
    assert warning.primary.kind == "INPUT_BOUND"
    assert warning.primary.severity == "warn"
    assert warning.issues[0].severity == "warn"

    confident = _diagnose_rank_map(
        _metrics_from_per_rank_timing(per_rank, steps=20),
        per_rank_timing=per_rank,
    )
    assert confident.primary.kind == "INPUT_BOUND"
    assert confident.primary.severity == "crit"
    assert confident.issues[0].severity == "crit"

    fsdp = _diagnose_rank_map(
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


def test_builtin_live_and_summary_policies_use_identical_thresholds() -> None:
    live_thresholds = LIVE_STEP_TIME_POLICY.thresholds
    summary_thresholds = SUMMARY_STEP_TIME_POLICY.thresholds

    assert live_thresholds is summary_thresholds
    assert live_thresholds.compute_bound_share_warn == pytest.approx(0.90)

    window = window_from_events(
        {0: _summary_step_events(input_wait_gpu=None, steps=40)},
        max_rows=100,
    )
    live = diagnose_step_time_window(window, policy=LIVE_STEP_TIME_POLICY)
    summary = diagnose_step_time_window(
        window,
        policy=SUMMARY_STEP_TIME_POLICY,
    )

    assert live.primary == summary.primary
    assert live.issues == summary.issues


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
    assert high_wait.issues[0].evidence[
        "traced_step_time_ms"
    ] == pytest.approx(60.0)
    assert high_wait.issues[0].evidence["step_time_ms"] == pytest.approx(85.0)


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


def test_summary_compute_trend_uses_outer_step_time_series() -> None:
    steps = 240
    per_step: dict[int, dict] = {}
    for step in range(steps):
        input_wait = 0.0 if step < steps // 2 else 10.0
        per_step[step] = {
            "_traceml_internal:dataloader_next": _event_stats(
                gpu_ms=input_wait
            ),
            "_traceml_internal:h2d_time": _event_stats(gpu_ms=0.0),
            "_traceml_internal:forward_time": _event_stats(gpu_ms=50.0),
            "_traceml_internal:backward_time": _event_stats(gpu_ms=30.0),
            "_traceml_internal:optimizer_step": _event_stats(gpu_ms=10.0),
            "_traceml_internal:step_time": _event_stats(gpu_ms=90.0),
        }

    result = _diagnose_summary_events(
        {0: per_step},
        max_rows=steps,
    )

    assert result.primary.kind == "COMPUTE_BOUND"
    assert result.primary.note is not None
    assert result.primary.note.startswith("Trend: step time is worsening")


@pytest.mark.parametrize(
    ("kind", "residual_share", "input_share"),
    [
        ("RESIDUAL_HEAVY", None, 0.0),
        ("INPUT_BOUND", 0.0, None),
    ],
)
def test_trend_abstains_when_required_share_is_unavailable(
    kind: str,
    residual_share: float | None,
    input_share: float | None,
) -> None:
    metric = _time_metric("trend", median=10.0, worst=10.0, steps=120)
    rising = [1.0] * 60 + [10.0] * 60
    metric = replace(
        metric,
        series=StepTimeSeries(
            median=rising,
            worst=rising,
        ),
    )

    note = build_step_trend_note(
        diagnosis_kind=kind,
        steps_used=120,
        single_rank=False,
        step_metric=metric,
        residual_metric=metric,
        input_wait_metric=metric,
        residual_share=residual_share,
        input_bound_share=input_share,
        residual_warn_threshold=0.1,
        input_warn_threshold=0.1,
    )

    assert note is None


def test_largest_compute_phase_uses_one_eligible_rank_cohort() -> None:
    context = _time_context(
        _time_metric("input_wait", median=0.0, worst=0.0),
        _time_metric("forward", median=55.0, worst=100.0),
        _time_metric("backward", median=20.0, worst=20.0),
        _time_metric("optimizer_step", median=10.0, worst=10.0),
        _time_metric("step_time", median=100.0, worst=100.0),
        per_rank_timing={
            0: {
                "input_wait": 0.0,
                "forward": 10.0,
                "backward": 20.0,
                "optimizer_step": 10.0,
                "step_time": 100.0,
            },
            1: {
                "input_wait": 0.0,
                "forward": 100.0,
                "step_time": 100.0,
            },
        },
    )

    assert context.largest_compute == "Backward"
