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

import json
import shutil
import subprocess

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui_sections import (  # noqa: E402
    charting,
)
from traceml_ai.aggregator.display_drivers.nicegui_sections.formatting import (  # noqa: E402
    format_elapsed,
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
    """The helper's own empty contract.

    Note what this does NOT cover: the System card never calls it with an
    empty list, it short-circuits to a bare ``{"data": []}`` literal that
    omits the silent/symbol/animation keys this returns. Two shapes for
    one empty case, improvised at the call site. Left alone here because
    converging them would change a rendered option dict, which this
    change is not allowed to do.
    """
    out = charting.mark_lines([])
    assert out["data"] == []
    assert out["silent"] is True
    assert out["symbol"] == "none"


def test_the_unit_reaches_both_the_axis_and_the_tooltip():
    """The unit is threaded into two formatters, and neither is obvious.

    A move that drops either leaves a chart whose numbers have no unit,
    which reads as a smaller change than it is.
    """
    out = charting.span_line_options("#2563eb", " W")
    assert " W" in out["yAxis"]["axisLabel"][":formatter"]
    assert " W" in str(out["tooltip"])

    multi = charting.multi_line_options("%")
    assert "%" in multi["yAxis"]["axisLabel"][":formatter"]


def test_a_reference_line_carries_its_colour_and_position():
    """Beyond the value and the label, which were already pinned."""
    out = charting.mark_lines(
        [(70.0, "70 W limit", "#dc2626", "insideEndTop")]
    )
    entry = out["data"][0]
    assert entry["lineStyle"]["color"] == "#dc2626"
    assert entry["label"]["position"] == "insideEndTop"


def test_two_reference_lines_both_survive():
    """Only the single-entry case was covered."""
    out = charting.mark_lines(
        [
            (70.0, "limit", "#dc2626", "insideEndTop"),
            (33.0, "lowest seen", "#9aa3af", "insideStartBottom"),
        ]
    )
    assert [e["yAxis"] for e in out["data"]] == [70.0, 33.0]


# --- the relative time axis (#511) ---------------------------------------
def test_a_span_axis_is_labelled_relative_to_the_newest_sample():
    """``-58s ... Now``: the card names its window once, in the header.

    A wall-clock tick on every axis repeated what the header says and
    made two stacked charts read as two different periods.
    """
    options = charting.multi_line_options("%")
    charting.apply_span_axis(options, 58.0, 1_700_000_000.0)

    axis = options["xAxis"]
    assert (axis["min"], axis["max"]) == (-58.0, 0)
    label = axis["axisLabel"]
    assert label["show"] is True
    assert "'Now'" in label[":formatter"]
    assert "\u2212" in label[":formatter"]
    assert "getHours" not in label[":formatter"], "no wall clock on ticks"


def test_relative_tick_labels_look_like_the_value_axis_labels():
    """Same colour, font and size on both axes of one chart."""
    options = charting.multi_line_options(" GB")
    charting.apply_span_axis(options, 58.0, 1_700_000_000.0)
    x_label = options["xAxis"]["axisLabel"]
    y_label = options["yAxis"]["axisLabel"]
    for key in ("color", "fontFamily", "fontSize"):
        assert x_label[key] == y_label[key], key


def test_the_hover_keeps_the_clock_beside_the_relative_reading():
    """Logs are keyed on the clock, so the hover still carries it."""
    options = charting.multi_line_options("%")
    charting.apply_span_axis(options, 58.0, 1_700_000_000.0)
    pointer = options["tooltip"]["axisPointer"]["label"][":formatter"]
    assert "getHours" in pointer
    assert "'Now'" in pointer


def test_relative_ticks_do_not_need_the_newest_epoch():
    """Without an epoch there is no clock, but the offsets still hold."""
    options = charting.multi_line_options("%")
    charting.apply_span_axis(options, 58.0)
    assert options["xAxis"]["axisLabel"]["show"] is True
    assert "'Now'" in options["xAxis"]["axisLabel"][":formatter"]


def test_the_relative_ticks_land_where_the_axis_is_labelled():
    options = charting.multi_line_options("%")
    charting.apply_span_axis(options, 58.0)
    label = options["xAxis"]["axisLabel"]
    assert label["customValues"] == [-58.0, -30.0, 0.0]
    # An ECharts without customValues falls back to the two ends.
    assert options["xAxis"]["interval"] == 58.0


@pytest.mark.parametrize(
    "span,expected",
    [
        (1.0, ["−1s", "Now"]),
        (2.0, ["−2s", "−1s", "Now"]),
        (3.0, ["−3s", "−2s", "−1s", "Now"]),
        (4.0, ["−4s", "−2s", "Now"]),
        (58.0, ["−58s", "−30s", "Now"]),
        (61.0, ["−1m 01s", "−30s", "Now"]),
        (317.0, ["−5m 17s", "−4m 00s", "−2m 00s", "Now"]),
        (
            10260.0,
            ["−2h 51m", "−2h 00m", "−1h 00m", "Now"],
        ),
    ],
)
def test_the_relative_ticks_name_the_span_then_round_steps(span, expected):
    """The leftmost matches the header; the ones between are round.

    Ticks at thirds printed "-1s -1s Now Now" on the first second of a run
    and "-57m 17s" between two hour ticks on a long one.
    """
    ticks = charting.relative_ticks(span)
    words = [
        "Now" if tick == 0 else "−" + format_elapsed(-tick) for tick in ticks
    ]
    assert words == expected


@pytest.mark.parametrize("span", [1, 2, 3, 4, 5, 6, 7, 9, 45, 61, 3601])
def test_no_relative_tick_label_repeats(span):
    ticks = charting.relative_ticks(float(span))
    words = [format_elapsed(-tick) for tick in ticks]
    assert len(set(words)) == len(words), words
    assert ticks == sorted(ticks) and len(ticks) <= 4


_NODE = shutil.which("node")


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
def test_the_js_tick_words_match_the_header_words():
    """The leftmost tick and the header must name the span alike.

    The tick formatter is JavaScript and the header is Python, so the
    two can drift apart without either one's tests noticing.
    """
    seconds = [0, 1, 9, 58, 59, 60, 61, 146, 599, 3600, 3661, 10920]
    script = (
        f"const f={charting._RELATIVE};"
        f"console.log(JSON.stringify({seconds}.map(s=>f(-s))));"
    )
    out = subprocess.run(
        [_NODE, "-e", script],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    expected = ["Now" if s == 0 else "−" + format_elapsed(s) for s in seconds]
    assert json.loads(out) == expected


@pytest.mark.parametrize(
    "value,step,expected",
    [
        (1.111, 0.005, "1.111 GB"),
        (1.52, 0.01, "1.52 GB"),
        (1.5, 0.5, "1.5 GB"),
        (45.0, 25.0, "45 GB"),
    ],
)
def test_a_value_axis_label_is_written_at_the_formatters_precision(
    value, step, expected
):
    """The Python twin of the JS formatter, used to size label room."""
    assert charting.value_axis_label(value, step, " GB") == expected
    decimals = str(len(expected.split(" ")[0].partition(".")[2]))
    assert f"toFixed({decimals})" in charting.value_axis_formatter(step, " GB")


def test_a_unit_axis_label_is_the_value_then_the_unit():
    assert charting.unit_axis_label(30.0, "%") == "30%"
    assert charting.unit_axis_formatter("%") == "v=>v+'%'"


def test_padded_tick_labels_share_one_width():
    padded = charting.pad_tick_labels("v=>v+'%'", 8)
    assert padded == "v=>(v=>v+'%')(v).padStart(8)"
