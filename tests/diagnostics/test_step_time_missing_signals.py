# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Missing Step Time signals stay missing and gate diagnosis (issue #257).

Every scenario runs the real event pipeline where possible:
raw event payloads -> canonical window -> shared diagnosis. Each
missing-signal test has a measured-zero or complete-data control twin so
"absent" and "measured zero" stay distinguishable forever.
"""

from __future__ import annotations

from typing import Iterable

import pytest

from traceml_ai.diagnostics.common import DiagnosticResult
from traceml_ai.diagnostics.step_time.api import StepDiagnosis
from traceml_ai.diagnostics.step_time.context import build_step_time_context
from traceml_ai.diagnostics.step_time.formatters import format_cli_diagnosis
from traceml_ai.diagnostics.step_time.policy import SUMMARY_STEP_TIME_POLICY
from traceml_ai.diagnostics.step_time.rules import RankStragglerRule
from traceml_ai.renderers.step_time.schema import (
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeSummary,
)
from traceml_ai.reporting.compare.policy import step_time_status_rank
from traceml_ai.reporting.primary_diagnosis import build_primary_diagnosis
from traceml_ai.reporting.summaries.issue_summary import (
    diagnostic_result_to_json,
)
from traceml_ai.utils.step_time_window import (
    build_step_time_window_from_events,
    diagnose_step_time_window,
)

_EVENT_NAMES = {
    "input_wait": "_traceml_internal:dataloader_next",
    "h2d": "_traceml_internal:h2d_time",
    "forward": "_traceml_internal:forward_time",
    "backward": "_traceml_internal:backward_time",
    "optimizer_step": "_traceml_internal:optimizer_step",
    "step_time": "_traceml_internal:step_time",
}

_DEFAULT_MS = {
    "input_wait": 5.0,
    "h2d": 0.0,
    "forward": 20.0,
    "backward": 30.0,
    "optimizer_step": 10.0,
    "step_time": 62.0,
}


def _stats(value: float, *, gpu: bool) -> dict:
    return {
        "cpu": {
            "duration_ms": value,
            "cpu_ms": value,
            "gpu_ms": value if gpu else None,
            "n_calls": 1,
        }
    }


def _step_events(
    *,
    omit: Iterable[str] = (),
    gpu: bool = False,
    steps: int = 30,
    **values_ms: float,
) -> dict[int, dict]:
    """Build per-step event maps; metrics in ``omit`` are never emitted."""
    values = dict(_DEFAULT_MS)
    values.update(values_ms)
    omitted = set(omit)
    out: dict[int, dict] = {}
    for step in range(steps):
        out[step] = {
            _EVENT_NAMES[key]: _stats(values[key], gpu=gpu)
            for key in _EVENT_NAMES
            if key not in omitted
        }
    return out


def _diagnose(
    per_rank_steps: dict[int, dict],
    *,
    training_strategy: str = "ddp",
    max_rows: int = 30,
) -> DiagnosticResult[StepDiagnosis]:
    window = build_step_time_window_from_events(
        per_rank_steps,
        max_rows=max_rows,
    )
    return diagnose_step_time_window(
        window,
        policy=SUMMARY_STEP_TIME_POLICY,
        training_strategy=training_strategy,
    )


def _step_metric(*, world_size: int = 2) -> StepCombinedTimeMetric:
    return StepCombinedTimeMetric(
        metric="step_time",
        clock="cpu",
        series=None,
        summary=StepCombinedTimeSummary(
            window_size=30,
            steps_used=30,
            median_total=100.0,
            worst_total=100.0,
            worst_rank=1,
            skew_ratio=0.0,
            skew_pct=0.0,
        ),
        coverage=StepCombinedTimeCoverage(
            expected_steps=30,
            steps_used=30,
            completed_step=30,
            world_size=world_size,
            ranks_present=world_size,
            incomplete=False,
        ),
    )


# ---------------------------------------------------------------------------
# Missing vs measured-zero stays distinguishable in the window
# ---------------------------------------------------------------------------


def test_window_keeps_measured_zero_and_drops_missing() -> None:
    zero = build_step_time_window_from_events(
        {0: _step_events(h2d=0.0)}, max_rows=30
    )
    missing = build_step_time_window_from_events(
        {0: _step_events(omit=("h2d",))}, max_rows=30
    )

    assert zero.per_rank_timing[0]["h2d"] == pytest.approx(0.0)
    assert "h2d" not in missing.per_rank_timing[0]
    zero_metrics = {metric.metric for metric in zero.metrics}
    missing_metrics = {metric.metric for metric in missing.metrics}
    assert "h2d" in zero_metrics
    assert "h2d" not in missing_metrics


def test_grad_accum_sparse_optimizer_stays_available() -> None:
    events = _step_events(omit=("optimizer_step",), steps=24)
    for step in range(0, 24, 4):
        events[step][_EVENT_NAMES["optimizer_step"]] = _stats(8.0, gpu=False)

    window = build_step_time_window_from_events({0: events}, max_rows=24)

    # Present in 6 of 24 steps: available, averaged over the full window.
    assert window.per_rank_timing[0]["optimizer_step"] == pytest.approx(2.0)
    assert "residual_proxy" in window.per_rank_timing[0]
    result = diagnose_step_time_window(window, policy=SUMMARY_STEP_TIME_POLICY)
    assert result.primary.kind != "INCOMPLETE_DATA"


# ---------------------------------------------------------------------------
# Missing compute phases cannot produce COMPUTE_BOUND / RESIDUAL_HEAVY
# ---------------------------------------------------------------------------


def test_missing_forward_blocks_compute_bound_and_reports_incomplete() -> None:
    result = _diagnose(
        {
            0: _step_events(
                omit=("forward",),
                input_wait=2.0,
                backward=60.0,
                optimizer_step=10.0,
                step_time=90.0,
            )
        }
    )

    assert result.primary.kind == "INCOMPLETE_DATA"
    assert result.primary.status == "INCOMPLETE DATA"
    assert result.primary.severity == "info"
    assert result.issues[0].evidence["missing_signals"] == ["forward"]
    assert result.issues[0].evidence["signal_coverage"] == {"forward": "0/1"}


def test_complete_compute_dominance_still_reports_compute_bound() -> None:
    result = _diagnose(
        {
            0: _step_events(
                input_wait=2.0,
                forward=60.0,
                backward=20.0,
                optimizer_step=8.0,
                step_time=90.0,
            )
        }
    )

    assert result.primary.kind == "COMPUTE_BOUND"


def test_missing_optimizer_blocks_residual_heavy() -> None:
    result = _diagnose(
        {
            0: _step_events(
                omit=("optimizer_step",),
                input_wait=2.0,
                forward=40.0,
                backward=30.0,
                step_time=100.0,
            )
        }
    )

    assert result.primary.kind == "INCOMPLETE_DATA"
    assert result.issues[0].evidence["missing_signals"] == ["optimizer_step"]


def test_measured_zero_optimizer_still_reports_residual_heavy() -> None:
    result = _diagnose(
        {
            0: _step_events(
                input_wait=2.0,
                forward=40.0,
                backward=30.0,
                optimizer_step=0.0,
                step_time=100.0,
            )
        }
    )

    assert result.primary.kind == "RESIDUAL_HEAVY"


# ---------------------------------------------------------------------------
# Missing input wait gates input-dependent rules
# ---------------------------------------------------------------------------


def test_missing_input_wait_reports_incomplete_data() -> None:
    result = _diagnose(
        {
            0: _step_events(
                omit=("input_wait",),
                forward=30.0,
                backward=30.0,
                optimizer_step=10.0,
                step_time=80.0,
            )
        }
    )

    assert result.primary.kind == "INCOMPLETE_DATA"
    assert result.issues[0].evidence["missing_signals"] == ["input_wait"]


def test_missing_h2d_still_emits_input_bound_on_gpu_clock() -> None:
    result = _diagnose(
        {
            0: _step_events(
                omit=("h2d",),
                gpu=True,
                input_wait=20.0,
                forward=30.0,
                backward=30.0,
                optimizer_step=10.0,
                step_time=80.0,
            )
        }
    )

    # H2D and the residual are unavailable, but the input finding's own
    # inputs are complete, so it must still be emitted.
    assert result.primary.kind == "INPUT_BOUND"


# ---------------------------------------------------------------------------
# Missing synchronization anchors cannot produce a rank straggler
# ---------------------------------------------------------------------------

_BALANCED_RANK = dict(
    input_wait=4.0,
    h2d=0.0,
    forward=30.0,
    backward=30.0,
    optimizer_step=10.0,
    step_time=76.0,
)


def test_missing_backward_blocks_ddp_straggler() -> None:
    result = _diagnose(
        {
            0: _step_events(**_BALANCED_RANK),
            1: _step_events(
                omit=("backward",),
                **{
                    key: value
                    for key, value in _BALANCED_RANK.items()
                    if key != "backward"
                },
            ),
        }
    )

    assert result.primary.kind == "INCOMPLETE_DATA"
    assert "backward" in result.issues[0].evidence["missing_signals"]
    assert result.issues[0].evidence["signal_coverage"]["backward"] == "1/2"


def test_visible_backward_skew_still_reports_straggler() -> None:
    slow = dict(_BALANCED_RANK, backward=60.0, step_time=106.0)
    result = _diagnose(
        {
            0: _step_events(**_BALANCED_RANK),
            1: _step_events(**slow),
        }
    )

    assert result.primary.kind in {
        "STRAGGLER",
        "INPUT_STRAGGLER",
        "COMPUTE_STRAGGLER",
        "H2D_STRAGGLER",
    }


def test_missing_forward_blocks_fsdp_straggler() -> None:
    slow = dict(_BALANCED_RANK, backward=60.0, step_time=106.0)
    result = _diagnose(
        {
            0: _step_events(
                omit=("forward",),
                **{k: v for k, v in _BALANCED_RANK.items() if k != "forward"},
            ),
            1: _step_events(
                omit=("forward",),
                **{k: v for k, v in slow.items() if k != "forward"},
            ),
        },
        training_strategy="fsdp",
    )

    assert result.primary.kind == "INCOMPLETE_DATA"
    assert "forward" in result.issues[0].evidence["missing_signals"]


# ---------------------------------------------------------------------------
# Unavailable components are excluded from straggler cause attribution
# ---------------------------------------------------------------------------


def _straggler_issue(
    per_rank_timing: dict[int, dict[str, float]],
):
    context = build_step_time_context(
        metrics=(_step_metric(),),
        thresholds=SUMMARY_STEP_TIME_POLICY.thresholds,
        per_rank_timing=per_rank_timing,
    )
    return RankStragglerRule().evaluate(context)


def test_h2d_unmeasured_on_victim_cannot_name_h2d_straggler() -> None:
    sparse_victim = {
        0: {
            "input_wait": 5.0,
            "h2d": 45.0,
            "forward": 20.0,
            "backward": 10.0,
            "step_time": 100.0,
        },
        1: {
            "input_wait": 5.0,
            "forward": 20.0,
            "backward": 50.0,
            "step_time": 100.0,
        },
    }
    issue = _straggler_issue(sparse_victim)

    assert issue is not None
    assert issue.kind == "STRAGGLER"
    assert "h2d" not in issue.evidence["component_excesses_ms"]

    measured_victim = {
        0: dict(sparse_victim[0]),
        1: dict(sparse_victim[1], h2d=0.0),
    }
    control = _straggler_issue(measured_victim)

    assert control is not None
    assert control.kind == "H2D_STRAGGLER"


# ---------------------------------------------------------------------------
# Partial rank coverage: available findings still fire on measured ranks
# ---------------------------------------------------------------------------


def test_input_bound_fires_from_partially_covered_ranks() -> None:
    heavy_input = dict(_BALANCED_RANK, input_wait=20.0)
    result = _diagnose(
        {
            0: _step_events(**heavy_input),
            1: _step_events(
                omit=("input_wait",),
                **{
                    key: value
                    for key, value in _BALANCED_RANK.items()
                    if key != "input_wait"
                },
            ),
        }
    )

    assert result.primary.kind == "INPUT_BOUND"


# ---------------------------------------------------------------------------
# Complete-data behavior is unchanged
# ---------------------------------------------------------------------------


def test_complete_balanced_window_stays_balanced() -> None:
    result = _diagnose({0: _step_events(**_BALANCED_RANK)})

    assert result.primary.kind == "BALANCED"


# ---------------------------------------------------------------------------
# INCOMPLETE_DATA is a first-class status everywhere
# ---------------------------------------------------------------------------


def test_incomplete_data_serializes_and_formats() -> None:
    result = _diagnose(
        {
            0: _step_events(
                omit=("forward",),
                input_wait=2.0,
                backward=60.0,
                optimizer_step=10.0,
                step_time=90.0,
            )
        }
    )

    diagnosis_json, issues_json = diagnostic_result_to_json(result)
    assert diagnosis_json["kind"] == "INCOMPLETE_DATA"
    assert diagnosis_json["status"] == "INCOMPLETE DATA"
    assert diagnosis_json["evidence"]["missing_signals"] == ["forward"]
    assert issues_json[0] == diagnosis_json

    rendered = format_cli_diagnosis(result.primary)
    assert "INCOMPLETE DATA" in rendered
    assert "forward" in rendered


def test_incomplete_data_compare_rank_matches_no_data_tier() -> None:
    assert step_time_status_rank("INCOMPLETE DATA") == 0
    assert step_time_status_rank("INCOMPLETE DATA") < step_time_status_rank(
        "BALANCED"
    )


def test_incomplete_data_routes_to_insufficient_primary() -> None:
    result = _diagnose(
        {
            0: _step_events(
                omit=("forward",),
                input_wait=2.0,
                backward=60.0,
                optimizer_step=10.0,
                step_time=90.0,
            )
        }
    )
    diagnosis_json, issues_json = diagnostic_result_to_json(result)
    step_time_summary = {
        "diagnosis": diagnosis_json,
        "issues": issues_json,
        "global": {"window": {"steps_analyzed": 30}},
    }

    primary = build_primary_diagnosis(
        system_summary={
            "diagnosis": {"kind": "LOW_GPU_UTILIZATION"},
            "global": {"average": {"gpu_util_percent": 10.0}},
        },
        process_summary={},
        step_time_summary=step_time_summary,
        step_memory_summary={},
    )

    # Incomplete timing must become the insufficient-data primary, never
    # the low-GPU-utilization fallback reserved for BALANCED windows.
    assert primary["kind"] == "INSUFFICIENT_STEP_TIME_DATA"
    assert primary["evidence"]["step_time_status"] == "INCOMPLETE DATA"
    assert "missing phase signals" in primary["summary"]
