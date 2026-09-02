# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""What the chart builders produce today, pinned before they move.

Part 6 of #403 moves these out of `theme.py`, which should hold styling
tokens and nothing else. They had no tests at all, so the move would have
been unverifiable: an ECharts option dict that silently loses a key
renders a subtly different chart and no assertion notices.

These describe the CURRENT output. They are deliberately shape-and-value
assertions rather than golden blobs, so they survive a formatting change
and fail on a behaviour change.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui_sections import (  # noqa: E402
    charting,
    theme,
)


def test_a_span_chart_carries_axes_a_tooltip_and_one_colour():
    out = charting.span_line_options("#2563eb", "%")
    assert sorted(out) == [
        "animationDuration",
        "backgroundColor",
        "color",
        "grid",
        "series",
        "tooltip",
        "xAxis",
        "yAxis",
    ]
    assert out["color"] == ["#2563eb"]
    assert out["animationDuration"] == 300

    # A span chart carries its single series already, filled with an area
    # under the line. The multi-line kind starts empty because its callers
    # add one series per rank or per device.
    assert len(out["series"]) == 1
    assert out["series"][0]["data"] == []
    assert "areaStyle" in out["series"][0]


def test_a_multi_line_chart_has_no_single_colour_and_starts_empty():
    out = charting.multi_line_options(" W")
    # No single colour: each series carries its own, so ranks and devices
    # keep a stable identity across ticks.
    assert "color" not in out
    assert out["series"] == []
    assert out["animationDuration"] == 300


def test_the_two_chart_kinds_share_a_clock_but_not_a_y_axis():
    """Pinned because the difference is deliberate and easy to erase.

    Both are drawn one above the other, so they share an x axis: a
    vertical read across the pair has to mean the same moment.

    They do NOT share a y axis. The span chart anchors at zero, which is
    right for a percentage that means something against 0 and 100. The
    multi-line chart does not, because zero-anchoring a memory trace puts
    a real drift inside one pixel, which is a defect this series already
    fixed once on the RSS chart.
    """
    span = charting.span_line_options("#2563eb", "%")
    multi = charting.multi_line_options("%")
    assert span["xAxis"] == multi["xAxis"]
    assert span["tooltip"]["trigger"] == multi["tooltip"]["trigger"]

    assert span["yAxis"]["min"] == 0
    assert "min" not in multi["yAxis"]


def test_a_line_carries_its_name_colour_and_data():
    s = charting.line_series("cpu", "#2563eb", [1.0, 2.0, None])
    assert s["name"] == "cpu"
    assert s["type"] == "line"
    assert s["data"] == [1.0, 2.0, None]
    assert s["lineStyle"]["color"] == "#2563eb"
    # No symbols: a 120-point series with a dot per point is unreadable.
    assert s["showSymbol"] is False


def test_a_gap_in_a_line_stays_a_gap():
    """None must survive into the series, or an absence draws as zero."""
    s = charting.line_series("rss", "#FF8C00", [1.0, None, 3.0])
    assert s["data"][1] is None


def test_line_width_is_adjustable_and_defaulted():
    assert charting.line_series("a", "#000", [])["lineStyle"]["width"] == 1.6
    assert (
        charting.line_series("a", "#000", [], width=2.4)["lineStyle"]["width"]
        == 2.4
    )


def test_reference_lines_carry_a_label_a_colour_and_a_position():
    out = charting.mark_lines([(70.0, "70 W limit", "#c00", "insideEndTop")])
    data = out["data"]
    assert len(data) == 1
    assert data[0]["yAxis"] == 70.0
    assert data[0]["label"]["formatter"] == "70 W limit"


def test_no_reference_lines_is_an_empty_set_not_a_missing_key():
    assert charting.mark_lines([])["data"] == []


def test_a_tile_value_and_unit_render_together():
    assert "42" in theme.kval("42", "%")
    assert "%" in theme.kval("42", "%")
    assert "n/a" in theme.kval("n/a")
