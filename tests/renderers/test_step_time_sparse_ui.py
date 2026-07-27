# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Sparse Step Time coverage in the live CLI (issue #259).

An unavailable phase is omitted (never rendered as a measured zero), the
canonical INCOMPLETE_DATA diagnosis reaches the terminal, and a stale
complete view is never re-served once the computer's own window expires.
"""

from __future__ import annotations

from rich.console import Console

from traceml_ai.renderers.step_time.renderer import StepCombinedRenderer
from traceml_ai.renderers.step_time.schema import (
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeResult,
    StepCombinedTimeSummary,
)


def _metric(name: str, value: float) -> StepCombinedTimeMetric:
    return StepCombinedTimeMetric(
        metric=name,
        clock="cpu",
        series=None,
        summary=StepCombinedTimeSummary(
            window_size=30,
            steps_used=30,
            median_total=value,
            worst_total=value,
            worst_rank=0,
            skew_ratio=0.0,
            skew_pct=0.0,
        ),
        coverage=StepCombinedTimeCoverage(
            expected_steps=30,
            steps_used=30,
            completed_step=30,
            world_size=1,
            ranks_present=1,
            incomplete=False,
        ),
    )


def _render_text(renderable) -> str:
    console = Console(record=True, width=140, color_system=None)
    console.print(renderable)
    return console.export_text()


def _sparse_payload() -> StepCombinedTimeResult:
    # forward was never measured: no forward metric, no residual, and the
    # per-rank timing carries no forward key.
    return StepCombinedTimeResult(
        per_rank_timing={
            0: {
                "input_wait": 2.0,
                "h2d": 0.0,
                "backward": 60.0,
                "optimizer_step": 10.0,
                "step_time": 90.0,
                "total_step": 92.0,
            }
        },
        diagnosis_clock="cpu",
        diagnosis_metrics=[
            _metric("input_wait", 2.0),
            _metric("h2d", 0.0),
            _metric("backward", 60.0),
            _metric("optimizer_step", 10.0),
            _metric("step_time", 90.0),
        ],
    )


def test_cli_omits_missing_phase_and_shows_incomplete_data(
    monkeypatch,
) -> None:
    renderer = StepCombinedRenderer(db_path=":memory:")
    monkeypatch.setattr(renderer, "_payload", _sparse_payload)
    text = _render_text(renderer.get_panel_renderable())

    # The canonical diagnosis for the sparse window reaches the terminal.
    assert "INCOMPLETE DATA" in text
    assert "forward" in text

    # The unavailable phase columns are omitted entirely: the static
    # legend line mentions FWD twice (its own entry and the residual
    # formula) and RESIDUAL once, so any additional occurrence would be a
    # rendered column. BWD keeps its column (2 legend + 1 header).
    # Measured phases, including the measured-zero H2D, stay visible.
    assert text.count("FWD") == 2
    assert text.count("RESIDUAL") == 1
    assert text.count("BWD") == 3
    assert "0.0 ms" in text
    assert "60.0 ms" in text


def test_cli_drops_dead_view_when_computer_window_expires(
    monkeypatch,
) -> None:
    renderer = StepCombinedRenderer(db_path=":memory:")

    monkeypatch.setattr(renderer._computer, "compute_cli", _sparse_payload)
    live_text = _render_text(renderer.get_panel_renderable())
    assert "60.0 ms" in live_text

    # The computer only returns an empty result once its own stale window
    # is exhausted; the renderer must not re-serve the old table.
    expired = StepCombinedTimeResult(
        status_message="No fresh step-combined data",
        per_rank_timing={},
        diagnosis_clock="cpu",
        diagnosis_metrics=[],
    )
    monkeypatch.setattr(renderer._computer, "compute_cli", lambda: expired)
    stale_text = _render_text(renderer.get_panel_renderable())

    assert "60.0 ms" not in stale_text
    assert "NO DATA" in stale_text
