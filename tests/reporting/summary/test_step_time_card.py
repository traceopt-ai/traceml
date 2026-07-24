# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from traceml_ai.diagnostics.step_time.adapters import (
    StepTimeDiagnosisInput,
    diagnose_step_time_summary,
)
from traceml_ai.reporting.primary_diagnosis import build_primary_diagnosis
from traceml_ai.reporting.summaries.step_time import (
    RankStepSummary,
)
from traceml_ai.reporting.sections.step_time.builder import (
    build_step_time_payload,
)
from traceml_ai.reporting.sections.step_time.loader import StepTimeSectionData
from traceml_ai.reporting.sections.step_time.model import (
    rank_summaries_from_window,
)
from traceml_ai.utils.step_time_window import (
    build_step_time_window_from_events,
)


def _rank(
    *,
    steps: int = 64,
    dataloader: float = 5.0,
    h2d: float = 0.0,
    forward: float = 30.0,
    backward: float = 50.0,
    optimizer: float = 10.0,
    step_cpu: float | None = None,
) -> RankStepSummary:
    compute = forward + backward + optimizer
    known_step = h2d + compute
    effective_step = max(
        step_cpu if step_cpu is not None else known_step, known_step
    )
    residual = max(0.0, effective_step - known_step)
    return RankStepSummary(
        steps_analyzed=steps,
        avg_dataloader_ms=dataloader,
        avg_input_wait_ms=dataloader,
        avg_step_time_ms=effective_step,
        avg_h2d_ms=h2d,
        avg_forward_ms=forward,
        avg_backward_ms=backward,
        avg_optimizer_ms=optimizer,
        avg_traced_step_ms=effective_step,
        avg_compute_ms=compute,
        avg_residual_ms=residual,
        avg_total_step_ms=dataloader + effective_step,
    )


def _summary(
    per_global_rank: dict[int, RankStepSummary],
    per_rank_steps: dict[int, dict[int, dict]] | None = None,
):
    window_size = 64
    step_events = (
        per_rank_steps
        if per_rank_steps is not None
        else {
            rank: _step_events_from_rank(summary)
            for rank, summary in per_global_rank.items()
        }
    )
    window = build_step_time_window_from_events(
        step_events,
        max_rows=window_size,
        expected_ranks=sorted(per_global_rank),
    )
    selected_summary = rank_summaries_from_window(window)
    data = StepTimeSectionData(
        training_steps=100,
        latest_step_observed=99,
        step_time_window=window,
        per_global_rank_summary=selected_summary,
        identities={},
        max_rows=window_size,
        training_strategy="ddp",
    )
    diagnosis = diagnose_step_time_summary(
        StepTimeDiagnosisInput(
            window=window,
            training_strategy=data.training_strategy,
        )
    )
    return build_step_time_payload(data, diagnosis)


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


def _step_events_from_rank(
    summary: RankStepSummary,
) -> dict[int, dict]:
    return {
        step: {
            "_traceml_internal:dataloader_next": _event_stats(
                cpu_ms=summary.avg_dataloader_ms
            ),
            "_traceml_internal:h2d_time": _event_stats(
                cpu_ms=summary.avg_h2d_ms
            ),
            "_traceml_internal:forward_time": _event_stats(
                cpu_ms=summary.avg_forward_ms
            ),
            "_traceml_internal:backward_time": _event_stats(
                cpu_ms=summary.avg_backward_ms
            ),
            "_traceml_internal:optimizer_step": _event_stats(
                cpu_ms=summary.avg_optimizer_ms
            ),
            "_traceml_internal:step_time": _event_stats(
                cpu_ms=summary.avg_traced_step_ms
            ),
        }
        for step in range(int(summary.steps_analyzed))
    }


def _input_bound_step_metrics(
    *,
    dataloader_cpu: float = 12.0,
    step_time_cpu: float = 90.0,
    input_wait_gpu: float,
    step_time_gpu: float,
    h2d_gpu: float = 0.0,
    compute_gpu: float = 0.0,
    steps: int = 64,
) -> dict[int, dict]:
    return {
        step: {
            "_traceml_internal:dataloader_next": _event_stats(
                cpu_ms=dataloader_cpu, gpu_ms=input_wait_gpu
            ),
            "_traceml_internal:h2d_time": _event_stats(gpu_ms=h2d_gpu),
            "_traceml_internal:forward_time": _event_stats(gpu_ms=compute_gpu),
            "_traceml_internal:step_time": _event_stats(
                cpu_ms=step_time_cpu,
                gpu_ms=step_time_gpu,
            ),
        }
        for step in range(steps)
    }


def _alternating_residual_step_metrics(steps: int = 64) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for step in range(steps):
        compute_ms, step_time_ms = (
            (50.0, 100.0) if step % 2 == 0 else (100.0, 50.0)
        )
        out[step] = {
            "_traceml_internal:dataloader_next": _event_stats(cpu_ms=0.0),
            "_traceml_internal:h2d_time": _event_stats(cpu_ms=0.0),
            "_traceml_internal:forward_time": _event_stats(cpu_ms=compute_ms),
            "_traceml_internal:step_time": _event_stats(cpu_ms=step_time_ms),
        }
    return out


def _assert_compact_card(card: str) -> None:
    assert "- Issues:" not in card
    assert "- Note:" not in card
    assert "- Global:" not in card
    assert "- Dominant:" not in card


def _assert_public_step_metrics_keep_dataloader(payload) -> None:
    public_metric_keys = set(payload["global"]["median"])
    assert "dataloader_ms" in public_metric_keys
    assert "input_wait_ms" in public_metric_keys
    assert "step_time_ms" in public_metric_keys

    for row in payload["groups"]["rows"].values():
        row_metric_keys = set(row["metrics"])
        assert "dataloader_ms" in row_metric_keys
        assert "input_wait_ms" in row_metric_keys
        assert "step_time_ms" in row_metric_keys


def test_step_time_no_data_card_is_compact() -> None:
    payload = _summary({})

    assert payload["diagnosis"] == payload["issues"][0]
    assert payload["diagnosis"]["kind"] == "NO_DATA"
    assert "- Diagnosis: NO DATA" in payload["card"]
    assert "- Stats: n/a" in payload["card"]
    assert "- Why: Need more step-time samples." in payload["card"]
    _assert_compact_card(payload["card"])


def test_step_time_balanced_card_is_compact() -> None:
    payload = _summary(
        {
            0: _rank(
                dataloader=5.0,
                h2d=15.0,
                forward=20.0,
                backward=35.0,
                optimizer=5.0,
                step_cpu=75.0,
            ),
            1: _rank(
                dataloader=5.0,
                h2d=15.0,
                forward=21.0,
                backward=34.0,
                optimizer=5.0,
                step_cpu=75.0,
            ),
        }
    )

    assert payload["diagnosis"] == payload["issues"][0]
    assert payload["diagnosis"]["status"] == "BALANCED"
    assert "- Diagnosis: BALANCED" in payload["card"]
    assert "- Stats: median/worst |" in payload["card"]
    assert "- Residual: median/worst" in payload["card"]
    assert "- Ranks: median/worst |" in payload["card"]
    assert "- Residual ranks: median/worst" in payload["card"]
    assert "- Why: No clear timing bottleneck." in payload["card"]
    _assert_compact_card(payload["card"])


def test_step_time_compute_bound_card_uses_short_reason() -> None:
    payload = _summary(
        {
            0: _rank(
                dataloader=2.0,
                forward=20.0,
                backward=65.0,
                optimizer=5.0,
                step_cpu=95.0,
            )
        }
    )

    assert payload["diagnosis"] == payload["issues"][0]
    assert payload["diagnosis"]["status"] == "COMPUTE-BOUND"
    assert (
        "- Stats: total 97.0ms | input 2.0ms | H2D 0.0ms | compute 90.0ms"
        in payload["card"]
    )
    assert "- Residual: 5.0ms" in payload["card"]
    assert (
        "- Why: Compute dominated (90.0ms/97.0ms); backward was largest."
        in payload["card"]
    )
    _assert_compact_card(payload["card"])


def test_step_time_input_bound_card_uses_short_reason() -> None:
    payload = _summary(
        {
            0: _rank(
                dataloader=40.0,
                forward=20.0,
                backward=35.0,
                optimizer=5.0,
                step_cpu=100.0,
            )
        },
        per_rank_steps={
            0: _input_bound_step_metrics(
                input_wait_gpu=40.0,
                step_time_gpu=100.0,
                compute_gpu=90.0,
            )
        },
    )

    assert payload["diagnosis"] == payload["issues"][0]
    assert payload["diagnosis"]["status"] == "INPUT-BOUND"
    assert payload["diagnosis"]["metric"] == "input_wait"
    assert payload["diagnosis"]["phase"] == "input"
    assert payload["diagnosis"]["evidence"]["input_wait_ms"] == 40.0
    assert payload["diagnosis"]["evidence"]["step_time_ms"] == 100.0
    assert payload["diagnosis"]["evidence"]["iteration_time_ms"] == 140.0
    assert payload["diagnosis"]["evidence"]["diagnosis_clock"] == "gpu"
    assert payload["global"]["window"]["diagnosis_clock"] == "gpu"
    average = payload["global"]["average"]
    assert average["dataloader_ms"] == 12.0
    assert average["input_wait_ms"] == 40.0
    assert average["step_time_ms"] == 100.0
    assert average["total_step_ms"] == 102.0
    row_metrics = payload["groups"]["rows"]["0"]["metrics"]
    assert row_metrics["dataloader_ms"] == 12.0
    assert row_metrics["input_wait_ms"] == 40.0
    assert row_metrics["step_time_ms"] == 100.0
    assert row_metrics["total_step_ms"] == 102.0
    _assert_public_step_metrics_keep_dataloader(payload)
    assert (
        "- Why: Input wait was 40.0ms of 140.0ms gpu iteration time."
        in payload["card"]
    )
    _assert_compact_card(payload["card"])


def test_step_time_h2d_bound_card_uses_short_reason() -> None:
    payload = _summary(
        {
            0: _rank(
                dataloader=12.0,
                h2d=20.0,
                forward=20.0,
                backward=35.0,
                optimizer=5.0,
                step_cpu=100.0,
            )
        },
        per_rank_steps={
            0: _input_bound_step_metrics(
                input_wait_gpu=0.0,
                h2d_gpu=20.0,
                compute_gpu=70.0,
                step_time_cpu=100.0,
                step_time_gpu=100.0,
            )
        },
    )

    assert payload["diagnosis"] == payload["issues"][0]
    assert payload["diagnosis"]["status"] == "H2D-BOUND"
    assert payload["diagnosis"]["metric"] == "h2d"
    assert payload["diagnosis"]["phase"] == "h2d"
    assert payload["diagnosis"]["evidence"]["h2d_ms"] == 20.0
    assert payload["diagnosis"]["evidence"]["diagnosis_clock"] == "gpu"
    assert (
        "- Why: H2D transfer was high inside the total step" in payload["card"]
    )
    _assert_compact_card(payload["card"])


def test_step_time_residual_heavy_card_uses_short_reason() -> None:
    payload = _summary(
        {
            0: _rank(
                dataloader=2.0,
                forward=20.0,
                backward=45.0,
                optimizer=5.0,
                step_cpu=100.0,
            )
        }
    )

    assert payload["diagnosis"] == payload["issues"][0]
    assert payload["diagnosis"]["status"] == "RESIDUAL-HEAVY"
    assert (
        "- Why: Residual time was high inside the total step (30.0ms/102.0ms)."
        in payload["card"]
    )
    _assert_compact_card(payload["card"])


def test_step_time_residual_uses_average_of_per_step_clamps() -> None:
    payload = _summary(
        {0: _rank(steps=64)},
        per_rank_steps={0: _alternating_residual_step_metrics()},
    )

    average = payload["global"]["average"]
    row_metrics = payload["groups"]["rows"]["0"]["metrics"]
    assert average["step_time_ms"] == 75.0
    assert average["compute_ms"] == 75.0
    assert average["residual_ms"] == 25.0
    assert row_metrics["residual_ms"] == 25.0
    assert payload["diagnosis"]["status"] == "RESIDUAL-HEAVY"

    primary = build_primary_diagnosis(
        system_summary={"global": {"average": {"gpu_util_percent": 0.0}}},
        process_summary={},
        step_time_summary=payload,
        step_memory_summary={},
    )
    assert primary["kind"] == "RESIDUAL_HEAVY"
    assert primary["evidence"]["residual_ms"] == 25.0
    assert primary["evidence"]["shares"]["residual_pct"] == 33.333


def test_step_time_input_straggler_card_shows_rank_evidence() -> None:
    payload = _summary(
        {
            0: _rank(
                dataloader=10.0,
                forward=40.0,
                backward=190.0,
            ),
            1: _rank(
                dataloader=70.0,
                forward=40.0,
                backward=130.0,
            ),
        }
    )

    assert payload["diagnosis"] == payload["issues"][0]
    assert payload["diagnosis"]["status"] == "INPUT STRAGGLER"
    assert payload["diagnosis"]["metric"] == "input_wait"
    assert payload["diagnosis"]["phase"] == "input"
    _assert_public_step_metrics_keep_dataloader(payload)
    assert "- Ranks: median/worst |" in payload["card"]
    assert (
        "- Why: r1 input was slower than median global rank (70.0/40.0ms)."
        in payload["card"]
    )
    assert "issues" not in payload["groups"]["rows"]["1"]
    assert {issue["kind"] for issue in payload["issues"]} == {
        "INPUT_STRAGGLER",
        "INPUT_BOUND",
    }
    _assert_compact_card(payload["card"])
