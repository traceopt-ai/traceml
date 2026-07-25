# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from traceml_ai.reporting.primary_diagnosis import build_primary_diagnosis


def _system(kind: str = "NORMAL", gpu_util: float = 87.0) -> dict:
    status_by_kind = {
        "NORMAL": "NORMAL",
        "LOW_GPU_UTILIZATION": "LOW GPU UTIL",
        "MODERATE_GPU_UTILIZATION": "MODERATE GPU UTIL",
    }
    return {
        "diagnosis": {
            "kind": kind,
            "status": status_by_kind.get(kind, kind),
            "severity": "info",
            "summary": "system summary",
            "action": "system action",
        },
        "global": {
            "average": {
                "gpu_util_percent": gpu_util,
            }
        },
    }


def _step_time(
    kind: str,
    *,
    status: str | None = None,
    phase: str | None = None,
    issues: list[dict] | None = None,
) -> dict:
    diagnosis = {
        "kind": kind,
        "status": status or kind.replace("_", "-"),
        "severity": "warn",
        "summary": "step-time summary",
        "action": "step-time action",
        "metric": None,
        "phase": phase,
    }
    if issues is None:
        issues = [diagnosis]
    return {
        "diagnosis": diagnosis,
        "issues": issues,
        "global": {
            "window": {"steps_analyzed": 256, "diagnosis_clock": "gpu"},
            "average": {
                "total_step_ms": 200.0,
                "dataloader_ms": 50.0,
                "input_wait_ms": 80.0,
                "step_time_ms": 160.0,
                "h2d_ms": 10.0,
                "compute_ms": 120.0,
                "residual_ms": 20.0,
            },
            "median": {
                "dataloader_ms": {"value": 5.0, "idx": "0"},
                "input_wait_ms": {"value": 8.0, "idx": "0"},
                "h2d_ms": {"value": 10.0, "idx": "1"},
                "compute_ms": {"value": 100.0, "idx": "1"},
                "optimizer_ms": {"value": 10.0, "idx": "3"},
                "residual_ms": {"value": 20.0, "idx": "0"},
            },
            "worst": {
                "dataloader_ms": {"value": 80.0, "idx": "2"},
                "input_wait_ms": {"value": 120.0, "idx": "2"},
                "h2d_ms": {"value": 40.0, "idx": "2"},
                "compute_ms": {"value": 180.0, "idx": "2"},
                "optimizer_ms": {"value": 90.0, "idx": "2"},
                "residual_ms": {"value": 70.0, "idx": "2"},
            },
        },
    }


def _primary(step_time: dict, system: dict | None = None) -> dict:
    return build_primary_diagnosis(
        system_summary=system or _system(),
        process_summary={},
        step_time_summary=step_time,
        step_memory_summary={},
    )


def test_input_bound_uses_phase_share_evidence() -> None:
    primary = _primary(
        _step_time("INPUT_BOUND", status="INPUT-BOUND"),
        _system(gpu_util=38.0),
    )

    assert primary["kind"] == "INPUT_BOUND"
    assert primary["section"] == "step_time"
    assert primary["scope"] == "performance"
    assert primary["summary"] == (
        "Input wait was 80.0ms of 240.0ms iteration time."
    )
    assert primary["evidence"] == {
        "type": "phase_share",
        "basis": "average",
        "steps_analyzed": 256,
        "total_step_ms": 200.0,
        "step_time_ms": 160.0,
        "iteration_time_ms": 240.0,
        "diagnosis_clock": "gpu",
        "dataloader_ms": 50.0,
        "input_wait_ms": 80.0,
        "h2d_ms": 10.0,
        "compute_ms": 120.0,
        "residual_ms": 20.0,
        "shares": {
            "input_wait_pct": 33.333,
            "h2d_pct": 4.167,
            "compute_pct": 50.0,
            "residual_pct": 8.333,
        },
        "gpu_util_avg_percent": 38.0,
    }


def test_h2d_bound_uses_iteration_phase_share_evidence() -> None:
    primary = _primary(_step_time("H2D_BOUND", status="H2D-BOUND"))

    assert primary["kind"] == "H2D_BOUND"
    assert primary["summary"] == (
        "H2D transfer took 10.0ms of 240.0ms iteration time."
    )
    assert primary["evidence"]["type"] == "phase_share"
    assert primary["evidence"]["iteration_time_ms"] == 240.0
    assert primary["evidence"]["shares"] == {
        "input_wait_pct": 33.333,
        "h2d_pct": 4.167,
        "compute_pct": 50.0,
        "residual_pct": 8.333,
    }


def test_residual_heavy_uses_residual_phase_share() -> None:
    primary = _primary(_step_time("RESIDUAL_HEAVY", status="RESIDUAL-HEAVY"))

    assert primary["kind"] == "RESIDUAL_HEAVY"
    assert primary["summary"] == (
        "Residual time took 20.0ms of a 160.0ms average step."
    )
    assert primary["evidence"]["type"] == "phase_share"
    assert primary["evidence"]["shares"]["residual_pct"] == 8.333


def test_compute_bound_uses_neutral_compute_phase_share() -> None:
    primary = _primary(_step_time("COMPUTE_BOUND", status="COMPUTE-BOUND"))

    assert primary["kind"] == "COMPUTE_BOUND"
    assert primary["summary"] == (
        "Model compute took 120.0ms of a 160.0ms average step."
    )
    assert primary["evidence"]["shares"]["compute_pct"] == 50.0


@pytest.mark.parametrize(
    (
        "kind",
        "phase",
        "expected_metric",
        "expected_summary",
    ),
    [
        (
            "INPUT_STRAGGLER",
            "dataloader",
            "input_wait_ms",
            "Rank r2 input wait was 120.0ms vs median rank r0 at 8.0ms.",
        ),
        (
            "COMPUTE_STRAGGLER",
            "optimizer",
            "optimizer_ms",
            None,
        ),
        (
            "H2D_STRAGGLER",
            "h2d",
            "h2d_ms",
            "Rank r2 h2d was 40.0ms vs median rank r1 at 10.0ms.",
        ),
    ],
)
def test_rank_stragglers_use_rank_comparison_evidence(
    kind: str,
    phase: str,
    expected_metric: str,
    expected_summary: str | None,
) -> None:
    primary = _primary(
        _step_time(
            kind,
            status=kind.replace("_", " "),
            phase=phase,
        )
    )

    assert primary["kind"] == kind
    assert primary["evidence"]["type"] == "rank_comparison"
    assert primary["evidence"]["metric"] == expected_metric
    assert primary["evidence"]["phase"] == phase
    assert primary["evidence"]["worst"]["rank"] == 2
    if expected_summary is not None:
        assert primary["summary"] == expected_summary


def test_straggler_includes_input_and_compute_comparisons() -> None:
    issues = [
        {
            "kind": "STRAGGLER",
            "status": "STRAGGLER",
            "severity": "crit",
            "summary": "mixed",
            "action": "inspect",
        },
        {
            "kind": "INPUT_STRAGGLER",
            "phase": "dataloader",
        },
        {
            "kind": "COMPUTE_STRAGGLER",
            "phase": "compute",
        },
    ]
    primary = _primary(_step_time("STRAGGLER", issues=issues))

    assert primary["kind"] == "STRAGGLER"
    assert primary["evidence"]["type"] == "rank_comparison"
    comparisons = primary["evidence"]["comparisons"]
    assert [item["metric"] for item in comparisons] == [
        "input_wait_ms",
        "compute_ms",
    ]


def test_balanced_with_low_gpu_utilization_uses_utilization_fallback() -> None:
    primary = _primary(
        _step_time("BALANCED", status="BALANCED"),
        _system(kind="LOW_GPU_UTILIZATION", gpu_util=38.0),
    )

    assert primary["kind"] == "LOW_GPU_UTILIZATION_UNEXPLAINED"
    assert primary["section"] == "system"
    assert primary["evidence"] == {
        "type": "utilization_fallback",
        "gpu_util_avg_percent": 38.0,
        "step_time_status": "BALANCED",
        "steps_analyzed": 256,
    }


def test_balanced_without_low_gpu_utilization_reports_no_clear_bottleneck() -> (
    None
):
    primary = _primary(_step_time("BALANCED", status="BALANCED"))

    assert primary["kind"] == "NO_CLEAR_PERFORMANCE_BOTTLENECK"
    assert primary["section"] == "step_time"
    assert primary["evidence"]["type"] == "no_clear_bottleneck"
    assert primary["evidence"]["step_time_status"] == "BALANCED"


@pytest.mark.parametrize("kind", ["WARMUP", "NO_DATA"])
def test_warmup_and_no_data_report_insufficient_step_time_data(
    kind: str,
) -> None:
    primary = _primary(_step_time(kind, status=kind.replace("_", " ")))

    assert primary["kind"] == "INSUFFICIENT_STEP_TIME_DATA"
    assert primary["status"] == "INSUFFICIENT STEP-TIME DATA"
    assert primary["evidence"]["type"] == "insufficient_data"
    assert primary["evidence"]["step_time_status"] == kind.replace("_", " ")
