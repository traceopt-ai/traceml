# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Sparse Step Time coverage in the live CLI (issue #259).

An unavailable phase is omitted (never rendered as a measured zero), the
canonical INCOMPLETE_DATA diagnosis reaches the terminal, and a stale
complete view is never re-served once the live session expires its bridge.
"""

from __future__ import annotations

from unittest.mock import Mock

from rich.console import Console

from tests.step_time.factories import (
    live_result_from_window,
    window_from_rank_averages,
)
from traceml_ai.renderers.step_time.renderer import StepCombinedRenderer
from traceml_ai.step_time.model import (
    StepTimeMetric,
    StepTimeWindow,
)
from traceml_ai.step_time.pipeline import LiveStepTimeResult


def _metric(name: str, value: float) -> StepTimeMetric:
    return StepTimeMetric(
        metric=name,
        series=None,
        window_size=30,
        steps_used=30,
        median_total=value,
        worst_total=value,
        worst_rank=0,
        skew_ratio=0.0,
        skew_pct=0.0,
        measured_ranks=(0,),
    )


def _render_text(renderable) -> str:
    console = Console(record=True, width=140, color_system=None)
    console.print(renderable)
    return console.export_text()


def _sparse_payload() -> LiveStepTimeResult:
    # forward was never measured: no forward metric, no residual, and the
    # per-rank timing carries no forward key.
    per_rank_timing = {
        0: {
            "input_wait": 2.0,
            "h2d": 0.0,
            "backward": 60.0,
            "optimizer_step": 10.0,
            "step_time": 90.0,
            "total_step": 92.0,
        }
    }
    return live_result_from_window(
        window_from_rank_averages(
            per_rank_timing,
            expected_ranks=(0,),
            metrics=[
                _metric("input_wait", 2.0),
                _metric("h2d", 0.0),
                _metric("backward", 60.0),
                _metric("optimizer_step", 10.0),
                _metric("step_time", 90.0),
            ],
        )
    )


def test_cli_omits_missing_phase_and_shows_incomplete_data() -> None:
    renderer = StepCombinedRenderer(session=Mock())
    text = _render_text(renderer.render(_sparse_payload()))

    # The canonical diagnosis for the sparse window reaches the terminal.
    assert "INCOMPLETE DATA" in text
    assert "forward" in text

    # The unavailable phase columns are omitted from the table header;
    # measured phases, including the measured-zero H2D, stay visible.
    header = next(line for line in text.splitlines() if "Metric" in line)
    assert "FWD" not in header
    assert "RESIDUAL" not in header
    assert "BWD" in header
    assert "H2D" in header
    assert "0.0 ms" in text
    assert "60.0 ms" in text


def test_cli_drops_dead_view_when_session_window_expires() -> None:
    renderer = StepCombinedRenderer(session=Mock())

    live_payload = _sparse_payload()
    live_text = _render_text(renderer.render(live_payload))
    assert "60.0 ms" in live_text

    # The session only returns an expired result once its last-good window
    # is exhausted; the renderer must not re-serve the old table.
    expired = live_result_from_window(
        StepTimeWindow(),
        freshness="expired",
        status_message="No fresh step-combined data",
    )
    stale_text = _render_text(renderer.render(expired))

    assert "60.0 ms" not in stale_text
    assert "NO DATA" in stale_text
    assert "the last window expired" in stale_text


def test_cli_startup_shows_calm_waiting_panel() -> None:
    renderer = StepCombinedRenderer(session=Mock())

    text = _render_text(
        renderer.render(
            live_result_from_window(
                StepTimeWindow(),
                freshness="cold",
            )
        )
    )

    # Never had data: normal warm-up must not alarm with NO DATA.
    assert "Waiting for first fully completed step" in text
    assert "NO DATA" not in text


def test_cli_renders_multi_rank_sparse_table() -> None:
    def _multi_metric(name: str, value: float) -> StepTimeMetric:
        metric = _metric(name, value)
        return StepTimeMetric(
            metric=metric.metric,
            series=None,
            window_size=30,
            steps_used=30,
            median_total=value,
            worst_total=value * 2,
            worst_rank=1,
            skew_ratio=2.0,
            skew_pct=1.0,
            measured_ranks=(0, 1),
        )

    per_rank_timing = {
        0: {
            "input_wait": 2.0,
            "backward": 60.0,
            "optimizer_step": 10.0,
            "step_time": 90.0,
            "total_step": 92.0,
        },
        1: {
            "input_wait": 2.0,
            "backward": 120.0,
            "optimizer_step": 10.0,
            "step_time": 180.0,
            "total_step": 182.0,
        },
    }
    payload = live_result_from_window(
        window_from_rank_averages(
            per_rank_timing,
            expected_ranks=(0, 1),
            metrics=[
                _multi_metric("input_wait", 2.0),
                _multi_metric("backward", 60.0),
                _multi_metric("optimizer_step", 10.0),
                _multi_metric("step_time", 90.0),
            ],
        )
    )
    renderer = StepCombinedRenderer(session=Mock())
    text = _render_text(renderer.render(payload))

    header = next(line for line in text.splitlines() if "Metric" in line)
    assert "FWD" not in header
    assert "BWD" in header
    # Multi-rank rows render distinct median and worst averages.
    assert "Median avg" in text
    assert "Worst avg" in text
    assert "60.0 ms" in text
    assert "120.0 ms" in text
    assert "Worst Rank" in text
