# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Cross-surface characterization tests for the Step Time pipeline.

These tests freeze semantic contracts, not implementation layout. Every case
starts from the same SQLite scenario and reaches the real CLI, dashboard, and
final-summary adapters. Incidental Rich markup, timestamps, cache internals,
and dictionary ordering intentionally remain outside the golden surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

from tests.step_time.scenarios import (
    SCENARIOS,
    StepTimeScenario,
    create_step_time_database,
)
from tests.step_time.factories import rank_average
from traceml_ai.diagnostics.step_time.policy import LIVE_STEP_TIME_POLICY
from traceml_ai.diagnostics.step_time.api import diagnose_step_time_window
from traceml_ai.renderers.model_diagnostics.renderer import (
    ModelDiagnosticsRenderer,
)
from traceml_ai.renderers.step_memory.schema import StepMemoryCombinedResult
from traceml_ai.renderers.step_time import renderer as cli_renderer_module
from traceml_ai.renderers.step_time.renderer import StepTimeRenderer
from traceml_ai.reporting.sections.step_time import (
    STEP_TIME_METRIC_NAMES,
    StepTimeSummarySection,
)
from traceml_ai.step_time.model import StepTimeLoadRequest
from traceml_ai.step_time.pipeline import LiveStepTimeSession

_WINDOW_KEYS = {
    "input_wait",
    "h2d",
    "forward",
    "backward",
    "optimizer_step",
    "traced_step_time",
    "step_time",
    "residual_proxy",
}


@dataclass(frozen=True)
class DiagnosisGolden:
    """Stable diagnosis fields shared by live and final surfaces."""

    kind: str
    status: str
    severity: str
    worst_rank: int | None
    issues: tuple[str, ...]
    score: float | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)


# Raw selected metrics already live beside each scenario. Only derived values
# need separate goldens; this avoids copying every fixture profile into the
# assertion table and keeps missing keys visible.
EXPECTED_DERIVED_TIMING: dict[str, dict[int, dict[str, float]]] = {
    "complete_gpu": {
        0: {"residual_proxy": 5.0, "step_time": 100.0},
        1: {"residual_proxy": 5.0, "step_time": 100.0},
    },
    "sparse_missing_forward": {
        0: {"residual_proxy": 5.0, "step_time": 100.0},
        1: {"step_time": 100.0},
    },
    "measured_zero_forward": {
        0: {"residual_proxy": 5.0, "step_time": 100.0},
        1: {"residual_proxy": 35.0, "step_time": 100.0},
    },
    "single_rank_cpu": {
        0: {"residual_proxy": 5.0, "step_time": 100.0},
    },
    "ddp_rank_straggler": {
        0: {"residual_proxy": 0.0, "step_time": 140.0},
        1: {"residual_proxy": 0.0, "step_time": 140.0},
    },
    "fsdp_rank_straggler": {
        0: {"residual_proxy": 0.0, "step_time": 140.0},
        1: {"residual_proxy": 0.0, "step_time": 160.0},
    },
}

EXPECTED_DIAGNOSIS: dict[str, DiagnosisGolden] = {
    "complete_gpu": DiagnosisGolden(
        kind="BALANCED",
        status="BALANCED",
        severity="info",
        worst_rank=0,
        issues=("BALANCED",),
    ),
    "sparse_missing_forward": DiagnosisGolden(
        kind="BALANCED",
        status="BALANCED",
        severity="info",
        worst_rank=0,
        issues=("BALANCED",),
    ),
    "measured_zero_forward": DiagnosisGolden(
        kind="RESIDUAL_HEAVY",
        status="RESIDUAL-HEAVY",
        severity="warn",
        worst_rank=0,
        issues=("RESIDUAL_HEAVY",),
        score=0.2,
    ),
    "single_rank_cpu": DiagnosisGolden(
        kind="BALANCED",
        status="BALANCED",
        severity="info",
        worst_rank=None,
        issues=("BALANCED",),
    ),
    "ddp_rank_straggler": DiagnosisGolden(
        kind="INPUT_STRAGGLER",
        status="INPUT STRAGGLER",
        severity="crit",
        worst_rank=0,
        issues=("INPUT_STRAGGLER", "INPUT_BOUND"),
        score=5.0 / 7.0,
        evidence={
            "culprit_rank": 0,
            "victim_rank": 1,
            "visible_metric": "backward",
        },
    ),
    "fsdp_rank_straggler": DiagnosisGolden(
        kind="INPUT_STRAGGLER",
        status="INPUT STRAGGLER",
        severity="warn",
        worst_rank=0,
        issues=("INPUT_STRAGGLER", "INPUT_BOUND"),
        score=0.75,
        evidence={
            "culprit_rank": 0,
            "victim_rank": 1,
            "visible_metric": "forward_backward",
        },
    ),
}

EXPECTED_METRIC_SUMMARIES: dict[
    str,
    dict[str, tuple[float, float, int, float | None]],
] = {
    "complete_gpu": {
        "traced_step_time": (95.0, 95.0, 0, 0.0),
        "step_time": (100.0, 100.0, 0, 0.0),
    },
    "sparse_missing_forward": {
        "forward": (30.0, 30.0, 0, None),
        "residual_proxy": (5.0, 5.0, 0, None),
        "traced_step_time": (95.0, 95.0, 0, 0.0),
        "step_time": (100.0, 100.0, 0, 0.0),
    },
    "measured_zero_forward": {
        "forward": (15.0, 30.0, 0, 1.0),
        "residual_proxy": (20.0, 35.0, 1, 0.75),
        "traced_step_time": (95.0, 95.0, 0, 0.0),
        "step_time": (100.0, 100.0, 0, 0.0),
    },
    "single_rank_cpu": {
        "traced_step_time": (95.0, 95.0, 0, None),
        "step_time": (100.0, 100.0, 0, None),
    },
    "ddp_rank_straggler": {
        "input_wait": (50.0, 100.0, 0, 1.0),
        "traced_step_time": (90.0, 140.0, 1, 5.0 / 9.0),
        "step_time": (140.0, 140.0, 0, 0.0),
    },
    "fsdp_rank_straggler": {
        "input_wait": (50.0, 100.0, 0, 1.0),
        "traced_step_time": (100.0, 160.0, 1, 0.6),
        "step_time": (150.0, 160.0, 1, 1.0 / 15.0),
    },
}

# Focused final-summary goldens. These pin public projection semantics without
# turning every formatting detail into a refactor constraint.
EXPECTED_PUBLIC_ROWS: dict[str, dict[int, dict[str, float | None]]] = {
    "complete_gpu": {
        0: {
            "total_step_ms": 200.0,
            "dataloader_ms": 10.0,
            "input_wait_ms": 5.0,
            "step_time_ms": 95.0,
            "compute_ms": 85.0,
            "residual_ms": 5.0,
        }
    },
    "sparse_missing_forward": {
        1: {
            "forward_ms": None,
            "compute_ms": None,
            "residual_ms": None,
            "step_time_ms": 95.0,
        }
    },
    "measured_zero_forward": {
        1: {
            "forward_ms": 0.0,
            "compute_ms": 55.0,
            "residual_ms": 35.0,
        }
    },
    "single_rank_cpu": {
        0: {
            "total_step_ms": 100.0,
            "dataloader_ms": 5.0,
            "input_wait_ms": 5.0,
            "step_time_ms": 95.0,
        }
    },
    "ddp_rank_straggler": {
        0: {"total_step_ms": 140.0, "input_wait_ms": 100.0},
        1: {"total_step_ms": 140.0, "compute_ms": 140.0},
    },
    "fsdp_rank_straggler": {
        0: {"total_step_ms": 140.0, "input_wait_ms": 100.0},
        1: {"total_step_ms": 160.0, "compute_ms": 160.0},
    },
}


@pytest.fixture(params=SCENARIOS, ids=lambda scenario: scenario.name)
def scenario_db(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> tuple[StepTimeScenario, Path]:
    """Create one canonical scenario database for a contract test."""
    scenario: StepTimeScenario = request.param
    db_path = tmp_path / f"{scenario.name}.db"
    create_step_time_database(db_path, scenario)
    return scenario, db_path


def _assert_optional_float(actual: Any, expected: float | None) -> None:
    """Compare a possibly absent numeric contract value."""
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)


def test_canonical_window_matches_golden(
    scenario_db: tuple[StepTimeScenario, Path],
) -> None:
    """Freeze alignment, clock, sparsity, metrics, and diagnosis semantics."""
    scenario, db_path = scenario_db
    result = LiveStepTimeSession(
        str(db_path),
        request=StepTimeLoadRequest(
            window_size=len(scenario.steps),
            lookback_factor=4,
        ),
    ).refresh()

    window = result.analysis.window
    assert result.freshness == "live"
    assert window.training_strategy == scenario.training_strategy
    assert window.clock == scenario.clock
    assert window.steps == list(scenario.steps)
    assert window.rank_universe == tuple(sorted(scenario.profiles))
    assert window.coverage.expected_steps == len(scenario.steps)
    assert window.coverage.steps_used == len(scenario.steps)
    assert window.coverage.world_size == len(scenario.profiles)
    assert window.coverage.ranks_present == len(scenario.profiles)

    expected_rows = {
        rank: {
            **scenario.profiles[rank],
            **EXPECTED_DERIVED_TIMING[scenario.name][rank],
        }
        for rank in scenario.profiles
    }
    assert {facts.global_rank for facts in window.rank_facts} == set(
        expected_rows
    )
    for rank, expected in expected_rows.items():
        values = rank_average(window, rank)
        actual = {
            key: value
            for key in _WINDOW_KEYS
            if (value := values.value(key)) is not None
        }
        assert actual == pytest.approx(expected)

    for metric in _WINDOW_KEYS:
        expected_ranks = tuple(
            rank for rank, row in expected_rows.items() if metric in row
        )
        assert window.ranks_for(metric) == expected_ranks

    metrics = {metric.metric: metric for metric in window.metrics}
    for metric_name, expected in EXPECTED_METRIC_SUMMARIES[
        scenario.name
    ].items():
        median, worst, worst_rank, skew = expected
        metric = metrics[metric_name]
        assert metric.median_total == pytest.approx(median)
        assert metric.worst_total == pytest.approx(worst)
        assert metric.worst_rank == worst_rank
        _assert_optional_float(metric.skew_pct, skew)

    diagnosis = diagnose_step_time_window(
        window,
        policy=LIVE_STEP_TIME_POLICY,
    )
    expected = EXPECTED_DIAGNOSIS[scenario.name]
    assert diagnosis.primary.kind == expected.kind
    assert diagnosis.primary.status == expected.status
    assert diagnosis.primary.severity == expected.severity
    assert diagnosis.primary.worst_rank == expected.worst_rank
    assert diagnosis.primary.steps_used == len(scenario.steps)
    assert tuple(issue.kind for issue in diagnosis.issues) == expected.issues
    _assert_optional_float(diagnosis.issues[0].score, expected.score)
    for key, value in expected.evidence.items():
        assert diagnosis.issues[0].evidence[key] == value


def test_cli_dashboard_summary_diagnosis_parity(
    scenario_db: tuple[StepTimeScenario, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove all user surfaces expose one diagnosis for the same window."""
    scenario, db_path = scenario_db
    captured: dict[str, Any] = {}

    def capture_cli(diagnosis: Any) -> str:
        captured["diagnosis"] = diagnosis
        return diagnosis.status

    monkeypatch.setattr(
        cli_renderer_module,
        "format_cli_diagnosis",
        capture_cli,
    )
    cli_renderer = StepTimeRenderer(
        LiveStepTimeSession(
            str(db_path),
            request=StepTimeLoadRequest(
                window_size=len(scenario.steps),
                lookback_factor=4,
            ),
        )
    )
    cli_renderer.get_panel_renderable()
    cli_diagnosis = captured["diagnosis"]

    dashboard_result = LiveStepTimeSession(
        str(db_path),
        request=StepTimeLoadRequest(
            window_size=len(scenario.steps),
            lookback_factor=4,
        ),
    ).refresh()
    dashboard_renderer = ModelDiagnosticsRenderer(str(db_path))
    monkeypatch.setattr(
        dashboard_renderer._step_memory,
        "compute_dashboard",
        lambda: StepMemoryCombinedResult(
            metrics=[],
            status_message="No GPU detected",
        ),
    )
    dashboard = dashboard_renderer.get_dashboard_renderable(dashboard_result)
    dashboard_diagnosis = next(
        item for item in dashboard["items"] if item["source"] == "step_time"
    )

    summary = (
        StepTimeSummarySection(max_rows=len(scenario.steps))
        .build(str(db_path))
        .payload
    )
    summary_diagnosis = summary["diagnosis"]

    assert dashboard_diagnosis["kind"] == cli_diagnosis.kind
    assert summary_diagnosis["kind"] == cli_diagnosis.kind
    assert dashboard_diagnosis["status"] == cli_diagnosis.status
    assert summary_diagnosis["status"] == cli_diagnosis.status
    assert dashboard_diagnosis["severity"] == cli_diagnosis.severity
    assert summary_diagnosis["severity"] == cli_diagnosis.severity
    assert dashboard_diagnosis["reason"] == cli_diagnosis.reason
    assert summary_diagnosis["summary"] == cli_diagnosis.reason
    assert dashboard_diagnosis["action"] == cli_diagnosis.action
    assert summary_diagnosis["action"] == cli_diagnosis.action
    assert dashboard_diagnosis["steps_used"] == len(scenario.steps)
    assert summary["global"]["window"]["steps_analyzed"] == len(scenario.steps)

    expected_ranks = (
        [] if cli_diagnosis.worst_rank is None else [cli_diagnosis.worst_rank]
    )
    assert dashboard_diagnosis["worst_rank"] == cli_diagnosis.worst_rank
    assert summary_diagnosis["ranks"] == expected_ranks


def test_final_summary_projection_matches_golden(
    scenario_db: tuple[StepTimeScenario, Path],
) -> None:
    """Freeze public summary projection, especially absence versus zero."""
    scenario, db_path = scenario_db
    payload = (
        StepTimeSummarySection(max_rows=len(scenario.steps))
        .build(str(db_path))
        .payload
    )

    window = payload["global"]["window"]
    assert window["alignment"] == "common_steps"
    assert window["steps_analyzed"] == len(scenario.steps)
    assert window["start_step"] == scenario.steps[0]
    assert window["end_step"] == scenario.steps[-1]
    assert window["completed_step"] == scenario.steps[-1]
    assert window["window_size"] == len(scenario.steps)
    assert window["diagnosis_clock"] == scenario.clock

    metadata = payload["metadata"]
    assert metadata["global_ranks_seen"] == len(scenario.profiles)
    assert metadata["global_ranks_used"] == len(scenario.profiles)
    assert metadata["section_metric_names"] == STEP_TIME_METRIC_NAMES

    rows = payload["groups"]["rows"]
    assert set(rows) == {str(rank) for rank in scenario.profiles}
    for rank, expected_metrics in EXPECTED_PUBLIC_ROWS[scenario.name].items():
        metrics = rows[str(rank)]["metrics"]
        assert list(metrics) == STEP_TIME_METRIC_NAMES
        for metric, expected in expected_metrics.items():
            _assert_optional_float(metrics[metric], expected)

    # The same diagnosis object is projected as the first ordered issue.
    assert payload["diagnosis"] == payload["issues"][0]
