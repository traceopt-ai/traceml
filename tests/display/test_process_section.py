# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Process block formatting: what the block says, and what it refuses to."""

from __future__ import annotations

from typing import Any, Dict, List

from traceml_ai.aggregator.display_drivers.nicegui_sections.process_section import (  # noqa: E501
    SCOPE_NOTE,
    format_age,
    rank_color,
    rows_hint,
    rows_html,
    should_auto_open,
)

GB = 1024**3


def _rank(
    idx: int,
    *,
    cap: float = 2.1,
    rss: float = 1.4 * GB,
    alloc: float = 2.2 * GB,
    reserved: float = 5.6 * GB,
    total: float = 16.0 * GB,
    age: float = 1.0,
    stale: bool = False,
) -> Dict[str, Any]:
    return {
        "global_rank": idx,
        "node_rank": 0,
        "gpu_index": idx,
        "cpu_capacity": cap,
        "ram_used": rss,
        "ram_total": 200.0 * GB,
        "gpu_alloc": alloc,
        "gpu_reserved": reserved,
        "gpu_total": total,
        "age_s": age,
        "stale": stale,
    }


def test_the_block_states_its_scope() -> None:
    """The pairing that makes process CPU diagnostic needs this sentence.

    Without it the number reads as the box's CPU, and "host CPU high while
    process CPU is low" is unreadable.
    """
    assert "one process per rank" in SCOPE_NOTE
    assert "DataLoader workers not included" in SCOPE_NOTE


def test_rows_hint_states_coverage_and_never_a_verdict() -> None:
    roll = {
        "ranks_total": 4,
        "ranks_stale": 0,
        "reserved_imbalance_pct": 0.0,
    }
    assert rows_hint(roll, is_open=False) == (
        "4 ranks · reserved imbalance 0% · click to open"
    )
    assert rows_hint(roll, is_open=True).endswith("click to close")

    # A stale rank is named in the header, because the tiles no longer
    # speak for it.
    roll_stale = dict(roll, ranks_stale=1, reserved_imbalance_pct=16.0)
    assert rows_hint(roll_stale, is_open=False) == (
        "4 ranks · 1 stale, excluded · reserved imbalance 16% · "
        "click to open"
    )
    # One rank has no imbalance to speak of.
    assert rows_hint(
        {"ranks_total": 1, "ranks_stale": 0, "reserved_imbalance_pct": None},
        is_open=False,
    ) == ("1 rank · click to open")
    for text in (
        rows_hint(roll, is_open=False),
        rows_hint(roll_stale, is_open=True),
    ):
        for word in ("imbalanced", "high", "low", "bad", "healthy", "busy"):
            assert word not in text


def test_rows_dim_a_stale_rank_and_keep_it() -> None:
    """Which rank stopped is the first question of a stalled job."""
    ranks = [_rank(0), _rank(1), _rank(2), _rank(3, age=44.0, stale=True)]
    html = rows_html(ranks, [])
    assert "tml-stale" in html
    assert html.count("tml-stale") == 1
    assert "R3" in html
    assert "44 s" in html


def test_rows_show_absence_never_zero() -> None:
    """A CPU-only host has no CUDA numbers, and must not print 0.0 GB."""
    ranks = [
        _rank(0, alloc=None, reserved=None, total=None, cap=None)  # type: ignore[arg-type]
    ]
    html = rows_html(ranks, [])
    assert "n/a" in html
    assert "0.0 GB" not in html


def test_rows_carry_a_trend_per_rank() -> None:
    series = [
        {"global_rank": 0, "t": [1.0, 2.0, 3.0], "avg": [2.0, 2.4, 2.1]},
        {"global_rank": 1, "t": [1.0, 2.0, 3.0], "avg": [2.0, 2.0, 2.0]},
    ]
    html = rows_html([_rank(0), _rank(1)], series)
    assert html.count("<polyline") == 2
    # The chip and the line agree on the rank's colour.
    assert rank_color(0) in html and rank_color(1) in html


def test_no_rank_colour_is_a_verdict_red() -> None:
    import colorsys

    for idx in range(8):
        colour = rank_color(idx)
        red, green, blue = (
            int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)
        )
        hue = colorsys.rgb_to_hls(red, green, blue)[0] * 360
        assert not (hue < 15 or hue > 345), (idx, colour)


def test_the_rss_axis_fits_its_drift_instead_of_anchoring_at_zero() -> None:
    """The chart exists for the drift, so the drift must be visible.

    Real ranks sit around 1.5 GB and move by tens of MB across hours. On a
    zero-anchored axis that movement is under one pixel of a 92 px chart.
    """
    from traceml_ai.aggregator.display_drivers.nicegui_sections.process_section import (  # noqa: E501
        drift_axis_bounds,
    )

    low, high, tick = drift_axis_bounds([1.48, 1.49, 1.50, 1.52])
    assert low > 1.4 and high < 1.6
    # The 40 MB of drift occupies a real share of the axis, not a sliver.
    assert (1.52 - 1.48) / (high - low) > 0.4
    assert tick > 0

    # A flat series still gets a usable range rather than a zero-height one.
    flat_low, flat_high, _tick = drift_axis_bounds([2.0, 2.0])
    assert flat_high > flat_low
    # No data at all is not a crash.
    assert drift_axis_bounds([]) == (0.0, 1.0, 0.5)


def test_axis_ticks_get_their_precision_from_the_range() -> None:
    """Three ticks that all read "1.4 GB" tell the reader nothing."""
    from traceml_ai.aggregator.display_drivers.nicegui_sections import theme

    drift = theme.value_axis_formatter(0.03, " GB")
    assert "toFixed(3)" in drift
    assert "toFixed(1)" in theme.value_axis_formatter(5.0, " GB")
    assert "toFixed(0)" in theme.value_axis_formatter(200.0, " GB")


def test_age_reads_in_words() -> None:
    assert format_age(1.0) == "1 s"
    assert format_age(89.0) == "89 s"
    assert format_age(120.0) == "2 min"
    assert format_age(None) == "n/a"


def test_rows_open_on_the_rising_edge_only() -> None:
    assert should_auto_open(prev_over=False, over=True) is True
    assert should_auto_open(prev_over=True, over=True) is False
    assert should_auto_open(prev_over=True, over=False) is False
    assert should_auto_open(prev_over=False, over=False) is False


def _payload(
    ranks: List[Dict[str, Any]],
    *,
    run: bool = True,
    rows_over: bool = False,
    gpu: bool = True,
) -> Dict[str, Any]:
    base = 1_787_000_000.0
    stamps = [base + 30.0 * i for i in range(40)]
    series_key = "cpu_capacity_run" if run else "cpu_capacity"
    rss_key = "rss_run" if run else "rss"
    value_key = "avg" if run else "v"
    entries = [
        {
            "global_rank": rank["global_rank"],
            "t": stamps,
            value_key: [rank["cpu_capacity"] or 0.0] * len(stamps),
            "max": [rank["cpu_capacity"] or 0.0] * len(stamps),
            "span_s": stamps[-1] - stamps[0],
            "window_s": 30.0,
        }
        for rank in ranks
    ]
    rss_entries = [
        {
            "global_rank": rank["global_rank"],
            "t": stamps,
            value_key: [rank["ram_used"] or 0.0] * len(stamps),
            "max": [rank["ram_used"] or 0.0] * len(stamps),
            "span_s": stamps[-1] - stamps[0],
            "window_s": 30.0,
        }
        for rank in ranks
    ]
    live = [rank for rank in ranks if not rank["stale"]]
    return {
        "window_len": len(stamps),
        "gpu_available": gpu,
        "rollups": {
            "ranks": ranks,
            "ranks_total": len(ranks),
            "ranks_stale": len(ranks) - len(live),
            "ranks_reporting": len(live),
            "gpu_available": gpu,
            "cpu_capacity": {"p50": 2.1, "worst": 2.4, "worst_rank": 0},
            "rss": {
                "used": 1.4 * GB,
                "total": 200.0 * GB,
                "rank": live[0]["global_rank"] if live else None,
            },
            "cuda": {
                "alloc_p50": 2.2 * GB if gpu else None,
                "reserved": 5.6 * GB if gpu else None,
                "reserved_total": 16.0 * GB if gpu else None,
                "reserved_rank": 0 if gpu else None,
            },
            "reserved_imbalance_pct": 0.0 if gpu else None,
            "rows_over": rows_over,
            "tick_s": 2.0,
        },
        "series": {
            "cpu_capacity": [],
            "rss": [],
            "cpu_capacity_run": [],
            "rss_run": [],
            series_key: entries,
            rss_key: rss_entries,
        },
    }


def test_section_builds_and_updates_without_a_browser() -> None:
    from nicegui import ui

    from traceml_ai.aggregator.display_drivers.nicegui_sections.process_section import (  # noqa: E501
        build_process_section,
        update_process_section,
    )

    with ui.element("div"):
        panel = build_process_section()
    ranks = [_rank(0), _rank(1), _rank(2), _rank(3)]
    update_process_section(panel, _payload(ranks))

    # Levels carry their denominator; the rate carries its estimator.
    assert "2.1" in panel["tiles"]["cpu"].content
    assert panel["subs"]["cpu"].text == "median rank · of host capacity"
    assert "215 GB" in panel["tiles"]["rss"].content
    assert "17.2 GB" in panel["tiles"]["reserved"].content
    assert panel["subs"]["reserved"].text == "least-headroom rank · R0"
    assert panel["subs"]["alloc"].text == "median rank · live tensors"

    # Both charts are pinned to one axis, in wall-clock time.
    cpu_axis = panel["cpu_chart"].options["xAxis"]
    rss_axis = panel["rss_chart"].options["xAxis"]
    assert (cpu_axis["min"], cpu_axis["max"]) == (
        rss_axis["min"],
        rss_axis["max"],
    )
    assert "getHours" in cpu_axis["axisLabel"][":formatter"]
    assert len(panel["cpu_chart"].options["series"]) == 4

    # The label names the view and its estimator, never "whole run".
    assert panel["cpu_label"].text == (
        "process cpu · capacity per rank · last 20 min · rolling 30 s"
    )
    assert "whole run" not in panel["rss_label"].text

    # The rows open by themselves only on a rising edge ...
    assert panel["rows"].value is False
    update_process_section(panel, _payload(ranks, rows_over=True))
    assert panel["rows"].value is True
    # ... and a reader who closes them is not fought on the next tick.
    panel["rows"].value = False
    update_process_section(panel, _payload(ranks, rows_over=True))
    assert panel["rows"].value is False


def test_a_cpu_only_host_keeps_every_slot(tmp_path: Any = None) -> None:
    """One shape to learn on every host: the GPU tiles say so, in place."""
    from nicegui import ui

    from traceml_ai.aggregator.display_drivers.nicegui_sections.process_section import (  # noqa: E501
        build_process_section,
        update_process_section,
    )

    with ui.element("div"):
        panel = build_process_section()
    ranks = [
        _rank(0, alloc=None, reserved=None, total=None),  # type: ignore[arg-type]
        _rank(1, alloc=None, reserved=None, total=None),  # type: ignore[arg-type]
    ]
    update_process_section(panel, _payload(ranks, gpu=False))

    assert panel["tiles"]["reserved"].content == "n/a"
    assert panel["tiles"]["alloc"].content == "n/a"
    assert panel["subs"]["reserved"].text == "no GPU"
    # The host-side half of the block still works.
    assert "2.1" in panel["tiles"]["cpu"].content
    assert "215 GB" in panel["tiles"]["rss"].content
    assert len(panel["cpu_chart"].options["series"]) == 2
    # The per-rank rows are about ranks, not GPUs: they stay.
    assert "R0" in panel["rows_html"].content
