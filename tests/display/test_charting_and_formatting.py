# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The two presentation helper modules, on their own.

Neither imports NiceGUI, so these run without a display stack and a failure
points at the arithmetic rather than at a card.
"""

from __future__ import annotations

import pytest

from traceml_ai.aggregator.display_drivers.nicegui_sections import (
    charting,
    formatting,
)

GIB = float(1024**3)


# --- formatting ----------------------------------------------------------
def test_a_measured_value_never_prints_as_zero():
    """27 MB is a reading. Printed with one decimal it reads as absent."""
    value, rest = formatting.format_gb_pair(27 * 1024 * 1024, None)
    assert value == "0.03"
    assert rest == "GB"


def test_a_level_carries_its_denominator():
    value, rest = formatting.format_gb_pair(6.3 * GIB, 16.1 * GIB)
    assert (value, rest) == ("6.3", "/ 16.1 GB")


def test_absence_is_a_marker_not_a_number():
    assert formatting.format_gb_pair(None, 16.0 * GIB) == ("n/a", "")
    assert formatting.num(None) == "n/a"
    assert formatting.format_age(None) == "n/a"


def test_a_garbled_value_does_not_raise():
    assert formatting.num("not a number") == "n/a"
    assert formatting.format_age("later") == "n/a"
    assert formatting.gb("many") is None


def test_decimal_gb_conversion_preserves_the_moved_helper_contract():
    """Absence, measured zero, and failed conversion remain distinct."""
    assert formatting.gb_si(None) is None
    assert formatting.gb_si(0) == 0.0
    assert formatting.gb_si(1_000_000_000) == 1.0
    assert formatting.gb_si("not a number") is None
    assert formatting.gb_si(10**400) is None


@pytest.mark.parametrize(
    "seconds,expected",
    [(45.0, "45 s"), (300.0, "5 min"), (7200.0, "2.0 h")],
)
def test_an_age_is_written_in_the_unit_that_fits_it(seconds, expected):
    assert formatting.format_age(seconds) == expected


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, ""),
        (None, ""),
        (45.0, "last 45 s"),
        # 105 s used to read "last 2 min", overstating the window by 14%.
        (105.0, "last 105 s"),
        (600.0, "last 10 min"),
    ],
)
def test_a_span_is_one_phrase_or_nothing(seconds, expected):
    assert formatting.format_span(seconds) == expected


def test_a_real_but_small_percentage_is_not_rounded_to_zero():
    assert formatting.format_percent(0.4) == "<1"
    assert formatting.format_percent(22.0) == "22"
    assert formatting.format_percent(None) == "n/a"


# --- charting ------------------------------------------------------------
def test_a_capacity_axis_is_anchored_at_zero():
    top = charting.capacity_axis_max([12.0, 18.0])
    assert top >= 18.0
    assert top in (20.0, 30.0, 40.0, 60.0, 80.0, 100.0)


def test_a_drift_axis_fits_the_data_instead_of_the_origin():
    """The RSS case: a 20 MB movement on a 1.5 GB level.

    Zero-anchoring would put the whole movement inside one pixel, which is
    the leak this chart exists to show.
    """
    low, high, tick = charting.drift_axis_bounds([1.48, 1.49, 1.50])
    assert low > 1.0
    assert high < 2.0
    assert tick > 0


def test_a_flat_series_still_gets_a_range():
    low, high, _tick = charting.drift_axis_bounds([2.0, 2.0, 2.0])
    assert high > low
    # Mid-plot, not hugging the floor.
    assert (2.0 - low) == pytest.approx(high - 2.0)


def _labels(values, **kwargs):
    low, high, tick = charting.drift_axis_bounds(values, **kwargs)
    return [
        charting.value_axis_label(low + i * tick, tick, " GB")
        for i in range(3)
    ]


@pytest.mark.parametrize(
    "values",
    [
        # A steady process: a few kilobytes of movement. Every tick read
        # "0.586 GB" before the axis had a smallest range.
        [0.58601, 0.58605, 0.58608],
        [0.611, 0.6115, 0.6127],
        [0.06, 0.1, 0.143],
        [1.48, 1.49, 1.50],
        [2.0, 2.0, 2.0],
        [8.0, 10.0, 10.5],
    ],
)
def test_the_three_memory_ticks_differ_and_step_evenly(values):
    labels = _labels(values, min_span=0.01)
    assert len(set(labels)) == 3, labels
    numbers = [float(label.split()[0]) for label in labels]
    steps = [b - a for a, b in zip(numbers, numbers[1:])]
    assert steps[0] == pytest.approx(steps[1]), labels


def test_a_steady_process_gets_the_smallest_range_not_kilobytes():
    low, high, tick = charting.drift_axis_bounds(
        [0.58601, 0.58605, 0.58608], min_span=0.01
    )
    assert high - low >= 0.01
    assert low <= 0.58601 and high >= 0.58608
    assert _labels([0.58601, 0.58605, 0.58608], min_span=0.01) == [
        "0.580 GB",
        "0.587 GB",
        "0.594 GB",
    ]


def test_the_memory_axis_always_covers_the_data():
    for values in ([0.0, 0.004], [3.3, 3.31], [0.2, 9.7], [99.0, 101.0]):
        low, high, _tick = charting.drift_axis_bounds(values, min_span=0.01)
        assert low <= min(values) and high >= max(values), values


def test_a_non_finite_value_does_not_take_the_axis_down():
    low, high, _tick = charting.drift_axis_bounds(
        [1.0, float("inf"), float("nan"), 1.2], min_span=0.01
    )
    assert low <= 1.0 and high >= 1.2
    assert charting.drift_axis_bounds([float("nan")]) == (0.0, 1.0, 0.5)


def test_an_empty_series_does_not_raise():
    assert charting.drift_axis_bounds([]) == (0.0, 1.0, 0.5)
    assert charting.capacity_axis_max([]) == 4.0


@pytest.mark.parametrize(
    "step,decimals",
    [
        (50.0, "toFixed(0)"),
        (5.0, "toFixed(0)"),
        (0.5, "toFixed(1)"),
        (0.05, "toFixed(2)"),
        (0.008, "toFixed(3)"),
    ],
)
def test_tick_precision_comes_from_the_step_not_the_magnitude(step, decimals):
    assert decimals in charting.value_axis_formatter(step, " GB")


def test_a_sparkline_of_nothing_is_nothing():
    assert charting.sparkline_svg([], "#fff") == ""
    assert charting.sparkline_svg([None, None], "#fff") == ""


def test_a_sparkline_drops_gaps_rather_than_drawing_them_at_zero():
    svg = charting.sparkline_svg([1.0, None, 3.0], "#fff")
    assert svg.count(",") == 2
    assert "polyline" in svg


class _Trace:
    def __init__(self, stamps):
        self.timestamps = stamps


def test_the_shared_span_covers_every_trace_given():
    anchor, span = charting.shared_span(
        [_Trace((100.0, 110.0))], [_Trace((90.0, 105.0))]
    )
    assert anchor == 110.0
    assert span == pytest.approx(20.0)


def test_a_span_of_no_traces_is_none():
    assert charting.shared_span([], []) is None
    assert charting.shared_span([_Trace(())]) is None


def test_one_moment_spans_no_time():
    """The observed span; the axis keeps its own drawing floor."""
    anchor, span = charting.shared_span([_Trace((100.0,))])
    assert (anchor, span) == (100.0, 0.0)
    options = charting.multi_line_options("%")
    charting.apply_span_axis(options, span)
    assert options["xAxis"]["min"] == -1.0
