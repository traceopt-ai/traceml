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
from traceml_ai.reporting.sections.step_time.alignment import AlignedStepWindow
from traceml_ai.reporting.summaries.step_time import (
    RankStepSummary,
)
from traceml_ai.reporting.sections.step_time.builder import (
    build_step_time_payload,
)
from traceml_ai.reporting.sections.step_time.loader import StepTimeSectionData
from traceml_ai.utils.step_time_diagnosis_clock import (
    INPUT_WAIT_CPU_MS_KEY,
    INPUT_WAIT_GPU_MS_KEY,
    STEP_TIME_CPU_MS_KEY,
    STEP_TIME_GPU_MS_KEY,
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
    return RankStepSummary(
        steps_analyzed=steps,
        avg_dataloader_ms=dataloader,
        avg_h2d_ms=h2d,
        avg_forward_ms=forward,
        avg_backward_ms=backward,
        avg_optimizer_ms=optimizer,
        avg_step_cpu_ms=effective_step,
        avg_traced_step_ms=effective_step,
        avg_gpu_compute_ms=compute,
        avg_total_step_ms=dataloader + effective_step,
    )


def _summary(
    per_global_rank: dict[int, RankStepSummary],
    per_rank_step_metrics: (
        dict[int, dict[int, dict[str, float]]] | None
    ) = None,
):
    window_size = 64
    step_metrics = (
        per_rank_step_metrics
        if per_rank_step_metrics is not None
        else {
            rank: _step_metrics_from_rank(summary)
            for rank, summary in per_global_rank.items()
        }
    )
    data = StepTimeSectionData(
        training_steps=100,
        latest_step_observed=99,
        aligned_summary=per_global_rank,
        aligned_step_metrics=step_metrics,
        aligned_window=AlignedStepWindow(
            alignment="common_steps",
            steps_analyzed=min(
                (item.steps_analyzed for item in per_global_rank.values()),
                default=0,
            ),
            start_step=None,
            end_step=None,
            window_size=window_size,
            global_ranks_used=len(per_global_rank),
            global_ranks_observed=len(per_global_rank),
        ),
        per_global_rank_summary=per_global_rank,
        per_global_rank_step_metrics=step_metrics,
        identities={},
        max_rows=window_size,
    )
    diagnosis = diagnose_step_time_summary(
        StepTimeDiagnosisInput(
            per_rank_step_metrics=step_metrics,
            max_rows=window_size,
        )
    )
    return build_step_time_payload(data, diagnosis)


def _step_metrics_from_rank(
    summary: RankStepSummary,
) -> dict[int, dict[str, float]]:
    return {
        step: {
            "dataloader_fetch": summary.avg_dataloader_ms,
            "h2d": summary.avg_h2d_ms,
            "forward": summary.avg_forward_ms,
            "backward": summary.avg_backward_ms,
            "optimizer_step": summary.avg_optimizer_ms,
            "step_time": summary.avg_traced_step_ms,
            "residual_proxy": max(
                0.0,
                summary.avg_traced_step_ms
                - summary.avg_h2d_ms
                - summary.avg_forward_ms
                - summary.avg_backward_ms
                - summary.avg_optimizer_ms,
            ),
            INPUT_WAIT_CPU_MS_KEY: summary.avg_dataloader_ms,
            "h2d_cpu_ms": summary.avg_h2d_ms,
            "forward_cpu_ms": summary.avg_forward_ms,
            "backward_cpu_ms": summary.avg_backward_ms,
            "optimizer_step_cpu_ms": summary.avg_optimizer_ms,
            STEP_TIME_CPU_MS_KEY: summary.avg_traced_step_ms,
        }
        for step in range(int(summary.steps_analyzed))
    }


def _input_bound_step_metrics(
    *,
    input_wait_gpu: float,
    step_time_gpu: float,
    steps: int = 64,
) -> dict[int, dict[str, float]]:
    return {
        step: {
            "dataloader_fetch": 40.0,
            "h2d": 0.0,
            "forward": 20.0,
            "backward": 35.0,
            "optimizer_step": 5.0,
            "step_time": 100.0,
            "residual_proxy": 40.0,
            INPUT_WAIT_GPU_MS_KEY: input_wait_gpu,
            STEP_TIME_GPU_MS_KEY: step_time_gpu,
        }
        for step in range(steps)
    }


def _assert_compact_card(card: str) -> None:
    assert "- Issues:" not in card
    assert "- Note:" not in card
    assert "- Global:" not in card
    assert "- Dominant:" not in card


def _assert_public_step_metrics_keep_dataloader(payload) -> None:
    public_metric_keys = set(payload["global"]["median"])
    assert "dataloader_ms" in public_metric_keys
    assert "input_wait_ms" not in public_metric_keys

    for row in payload["groups"]["rows"].values():
        row_metric_keys = set(row["metrics"])
        assert "dataloader_ms" in row_metric_keys
        assert "input_wait_ms" not in row_metric_keys


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
                dataloader=20.0,
                forward=20.0,
                backward=35.0,
                optimizer=5.0,
                step_cpu=70.0,
            ),
            1: _rank(
                dataloader=20.0,
                forward=21.0,
                backward=34.0,
                optimizer=5.0,
                step_cpu=70.0,
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
        per_rank_step_metrics={
            0: _input_bound_step_metrics(
                input_wait_gpu=40.0,
                step_time_gpu=100.0,
            )
        },
    )

    assert payload["diagnosis"] == payload["issues"][0]
    assert payload["diagnosis"]["status"] == "INPUT-BOUND"
    assert payload["diagnosis"]["metric"] == "input_wait"
    assert payload["diagnosis"]["phase"] == "input"
    assert payload["diagnosis"]["evidence"]["input_wait_ms"] == 40.0
    assert payload["diagnosis"]["evidence"]["step_time_ms"] == 100.0
    assert payload["diagnosis"]["evidence"]["diagnosis_clock"] == "gpu"
    _assert_public_step_metrics_keep_dataloader(payload)
    assert (
        "- Why: Input wait took 40.0ms of a 100.0ms gpu step."
        in payload["card"]
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
        "INPUT_STRAGGLER"
    }
    _assert_compact_card(payload["card"])
