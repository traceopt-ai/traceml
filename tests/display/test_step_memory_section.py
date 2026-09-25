# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""What the Step memory card puts on screen for a given payload.

The card's update path runs on every UI tick with the renderer's
``StepMemoryCombinedResult``. These tests drive that path with the real
payload types and the real card, without a browser.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui_sections.step_memory_section import (  # noqa: E402,E501
    build_step_memory_section,
    update_step_memory_section,
)
from traceml_ai.renderers.step_memory.schema import (  # noqa: E402
    StepMemoryCombinedCoverage,
    StepMemoryCombinedMetric,
    StepMemoryCombinedResult,
    StepMemoryCombinedSeries,
    StepMemoryCombinedSummary,
)

GB = 1e9


def _metric(
    name: str = "peak_allocated",
    *,
    worst=(2.0 * GB, 2.5 * GB, 3.0 * GB),
    median=(1.0 * GB, 1.0 * GB, 1.5 * GB),
    world_size: int = 2,
) -> StepMemoryCombinedMetric:
    steps = [10, 11, 12]
    return StepMemoryCombinedMetric(
        metric=name,
        device="cuda:0",
        series=StepMemoryCombinedSeries(
            steps=steps, median=list(median), worst=list(worst)
        ),
        summary=StepMemoryCombinedSummary(
            window_size=len(steps),
            steps_used=len(steps),
            median_peak=max(median),
            worst_peak=max(worst),
            worst_rank=1,
            skew_ratio=1.0,
            skew_pct=100.0,
        ),
        coverage=StepMemoryCombinedCoverage(
            expected_steps=len(steps),
            steps_used=len(steps),
            completed_step=steps[-1],
            world_size=world_size,
            ranks_present=world_size,
            incomplete=False,
        ),
    )


def _result(*metrics: StepMemoryCombinedMetric) -> StepMemoryCombinedResult:
    return StepMemoryCombinedResult(metrics=list(metrics), status_message="OK")


def _panel() -> dict:
    from nicegui import ui

    with ui.element("div"):
        return build_step_memory_section()


def test_a_multi_rank_run_draws_the_worst_rank_over_the_median() -> None:
    panel = _panel()
    update_step_memory_section(panel, _result(_metric(world_size=2)))

    series = panel["chart"].options["series"]
    assert panel["chart"].options["xAxis"]["data"] == ["10", "11", "12"]
    assert series[0]["name"] == "Worst"
    assert series[0]["data"] == [2.0, 2.5, 3.0]
    assert series[1]["data"] == [1.0, 1.0, 1.5]
    assert panel["win"].text == "3 aligned steps"
    # The tiles describe the worst rank, newest step first.
    assert panel["kpis"]["peak"].content.startswith("3.00")


def test_a_single_rank_run_draws_one_peak_line() -> None:
    """One rank is its own median; a second line would just overlap it."""
    panel = _panel()
    update_step_memory_section(panel, _result(_metric(world_size=1)))

    series = panel["chart"].options["series"]
    assert series[0]["name"] == "Peak"
    assert series[0]["data"] == [2.0, 2.5, 3.0]
    assert series[1]["data"] == []


def test_a_cpu_only_run_says_step_memory_is_unavailable() -> None:
    """The renderer's no-GPU status becomes words, not an empty chart."""
    panel = _panel()
    update_step_memory_section(
        panel,
        StepMemoryCombinedResult(
            metrics=[],
            status_message=(
                "No GPU detected. Step memory uses torch-based GPU memory "
                "telemetry."
            ),
        ),
    )

    assert panel["win"].text == "no GPU"
    assert panel["hint"].text == (
        "No GPU present, so step memory is unavailable."
    )
    assert panel["chart"].options["series"][0]["data"] == []


def test_the_preferred_metric_wins_over_list_order() -> None:
    """The card shows allocated peaks even when reserved arrives first."""
    panel = _panel()
    reserved = _metric(
        "peak_reserved",
        worst=(5.0 * GB, 5.0 * GB, 5.0 * GB),
        median=(4.0 * GB, 4.0 * GB, 4.0 * GB),
    )
    update_step_memory_section(panel, _result(reserved, _metric()))

    assert panel["chart"].options["series"][0]["data"] == [2.0, 2.5, 3.0]
    assert panel["kpis"]["peak"].content.startswith("3.00")
