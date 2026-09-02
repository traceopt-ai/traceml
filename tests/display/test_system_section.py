# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""System block: four tiles, two charts, spread-triggered per-GPU rows.

Pure helpers carry the display rules so they can be checked without a
browser: levels read "used / total", rates read one window median, the
per-GPU rows open on their own when the across-GPU spread crosses the bar,
and nothing on the block carries a verdict word.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui_sections.system_section import (  # noqa: E402,E501
    cpu_axis_max,
    disclosure_text,
    format_gb_pair,
    format_span,
    format_window,
    gpu_color,
    odd_ones_out,
    power_axis_bounds,
    rows_html,
    should_auto_open,
    sparkline_svg,
)
from traceml_ai.renderers.system.dashboard_models import GpuRow  # noqa: E402

GB = 1e9


def test_span_reads_as_a_window_not_a_clock() -> None:
    assert format_span(200.0) == "last 3 min"
    assert format_span(238.0) == "last 4 min"
    assert format_span(50.0) == "last 50 s"
    assert format_span(0.0) == ""
    assert format_span(None) == ""


def test_levels_carry_their_denominator() -> None:
    assert format_gb_pair(6.31 * GB, 16.1 * GB) == ("6.3", "/ 16.1 GB")
    assert format_gb_pair(9.0 * GB, 200.0 * GB) == ("9.0", "/ 200 GB")
    assert format_gb_pair(None, 16.1 * GB) == ("n/a", "")
    # Changed deliberately: sub-gigabyte values keep useful precision.
    assert format_gb_pair(0.47 * GB, None) == ("0.47", "GB")


def test_disclosure_text_states_the_range_and_no_verdict() -> None:
    """The header reads the GPUs out, it does not judge them.

    Calling a GPU "idle" needs a threshold, and the one this layer used
    (20 %) disagreed with the diagnosis engine's low band (30 %), which is
    the surface inventing vocabulary the engine owns. The range says the
    same thing to a person without either.
    """

    def g(idx, util):
        return GpuRow(gpu_idx=idx, util_p50=util, util_now=util)

    def roll(gpus):
        """The range now arrives on the payload; the card only formats it."""
        values = [x.util_p50 for x in gpus if x.util_p50 is not None]
        return as_rollups(
            {"util_range": (min(values), max(values)) if values else None}
        )

    one_busy = [g(0, 100), g(1, 0), g(2, 0), g(3, 0)]
    all_busy = [g(i, 99) for i in range(4)]
    ramp = [g(0, 100), g(1, 60), g(2, 55), g(3, 40)]

    assert disclosure_text(one_busy, roll(one_busy), is_open=True) == (
        "4 GPUs · util 0 to 100% · click to close"
    )
    assert disclosure_text(one_busy, roll(one_busy), is_open=False) == (
        "4 GPUs · util 0 to 100% · click to open"
    )
    assert disclosure_text(ramp, roll(ramp), is_open=True) == (
        "4 GPUs · util 40 to 100% · click to close"
    )
    assert disclosure_text(all_busy, roll(all_busy), is_open=False) == (
        "4 GPUs · util 99 to 99% · click to open"
    )
    # One GPU has no range to speak of, and no rows worth naming.
    assert (
        disclosure_text([g(0, 99)], roll([g(0, 99)]), is_open=False)
        == "1 GPU · click to open"
    )
    assert disclosure_text([], as_rollups({}), is_open=False) == ""
    # No word anywhere claims a verdict the engine owns.
    for text in (
        disclosure_text(one_busy, roll(one_busy), is_open=True),
        disclosure_text(all_busy, roll(all_busy), is_open=False),
    ):
        for word in ("idle", "busy", "alike", "uneven", "low", "healthy"):
            assert word not in text


def test_rows_open_on_the_rising_edge_only() -> None:
    # Crossing the bar opens the rows once; staying above it does not
    # reopen rows the user closed; dropping below never closes them.
    assert should_auto_open(prev_over=False, over=True) is True
    assert should_auto_open(prev_over=True, over=True) is False
    assert should_auto_open(prev_over=True, over=False) is False
    assert should_auto_open(prev_over=False, over=False) is False


def test_power_axis_keeps_the_limit_in_frame() -> None:
    lo, hi, tick = power_axis_bounds([60.0, 66.0, 95.0], 70.0)
    assert lo <= 60.0 and hi >= 95.0 and hi >= 70.0
    # Round ticks, the bottom one a multiple of the tick size.
    assert tick % 10 == 0 and lo % tick == 0 and (hi - lo) % tick == 0
    # The reference shape: idle GPUs near 33 W, a busy one up to ~100 W.
    assert power_axis_bounds([33.0, 68.0, 101.0], 70.0) == (
        25.0,
        125.0,
        25.0,
    )
    # All busy: 60-100 W under a 70 W limit.
    assert power_axis_bounds([60.0, 100.0], 70.0) == (40.0, 120.0, 20.0)
    # No limit reported: bounds still cover the data.
    lo, hi, tick = power_axis_bounds([33.0, 34.0], None)
    assert lo <= 33.0 and hi >= 34.0 and lo >= 0.0
    assert power_axis_bounds([], None) == (0.0, 100.0, 50.0)


def test_cpu_axis_halves_to_whole_percents() -> None:
    assert cpu_axis_max([2.0, 2.4]) == 4.0
    assert cpu_axis_max([8.0, 9.0]) == 20.0
    assert cpu_axis_max([95.0]) == 100.0
    assert cpu_axis_max([]) == 4.0


def test_sparkline_is_inline_svg_with_gaps_dropped() -> None:
    svg = sparkline_svg([66.0, None, 68.0, 67.0], gpu_color(0))
    assert svg.startswith("<svg") and "<polyline" in svg
    assert gpu_color(0) in svg
    assert sparkline_svg([], gpu_color(1)) == ""
    assert sparkline_svg([None, None], gpu_color(1)) == ""


def as_payload(raw: dict):
    """The dict fixtures below, as the typed payload the card now takes.

    The fixtures stay mappings because they are more readable that way and
    because they describe the SHAPE the compute layer produces. The models
    own the adaptation, so this is the same path production takes.
    """
    from traceml_ai.renderers.system.dashboard_models import (
        SystemDashboardPayload,
        SystemRollups,
        SystemSeries,
    )

    return SystemDashboardPayload(
        window_len=raw.get("window_len", 0),
        gpu_available=bool(raw.get("gpu_available")),
        rollups=SystemRollups.from_dict(raw.get("rollups") or {}),
        series=SystemSeries.from_dict(raw.get("series") or {}),
    )


def _ctx(raw: dict):
    """One ctx mapping, typed."""
    from traceml_ai.renderers.system.dashboard_models import RunContext

    return RunContext(system_node=raw.get("system_node"))


def as_rollups(raw: dict):
    """One rollups mapping, typed."""
    from traceml_ai.renderers.system.dashboard_models import SystemRollups

    return SystemRollups.from_dict(raw)


def _gpus():
    return [
        {
            "gpu_idx": 0,
            "util_now": 100.0,
            "util_p50": 100.0,
            "mem_used": 6.67 * GB,
            "mem_total": 16.1 * GB,
            "temp": 54.0,
            "power": 68.0,
            "power_limit": 70.0,
            "reported": True,
        },
        {
            "gpu_idx": 1,
            "util_now": 0.0,
            "util_p50": 0.0,
            "mem_used": 0.47 * GB,
            "mem_total": 16.1 * GB,
            "temp": 41.0,
            "power": 33.0,
            "power_limit": 70.0,
            "reported": True,
        },
    ]


def test_rows_table_reads_per_gpu_and_tints_only_the_busy_row() -> None:
    series = [
        {"gpu_idx": 0, "values": [66.0, 68.0]},
        {"gpu_idx": 1, "values": [33.0, 33.0]},
    ]
    html = rows_html(
        as_rollups({"gpus": _gpus()}).gpus,
        series,
        as_rollups({"rows_over": True, "odd_gpus": [0]}),
    )
    assert "gpu0" in html and "gpu1" in html
    assert "6.67 / 16.1" in html and "0.47 / 16.1" in html
    assert "68 / 70" in html and "33 / 70" in html
    assert "<polyline" in html
    # One tinted row (the busy one) when the spread is over the bar...
    assert html.count("tml-mark") == 1
    assert (
        'class="tml-mark"><td><span style="color:#f97316">■</span> gpu0'
        in html
    )
    # ...and none when every GPU reads the same.
    calm = rows_html(
        as_rollups({"gpus": _gpus()}).gpus,
        series,
        as_rollups({"rows_over": False}),
    )
    assert "tml-mark" not in calm
    # No verdict words anywhere on the block.
    for word in ("Hot", "Warm", "HIGH", "verdict"):
        assert word not in html


def test_tint_marks_the_odd_ones_out() -> None:
    """The card reads the marked set; the rule itself lives in compute.

    These cases moved with it and are asserted unchanged against
    `_odd_gpus` in tests/display/test_system_section_characterization.py.
    What is checked here is that the card marks what it is handed and
    invents nothing.
    """
    assert odd_ones_out(as_rollups({"odd_gpus": [0]})) == {0}
    assert odd_ones_out(as_rollups({"odd_gpus": [3]})) == {3}
    assert odd_ones_out(as_rollups({"odd_gpus": [0, 2]})) == {0, 2}
    assert odd_ones_out(as_rollups({"odd_gpus": []})) == set()
    assert odd_ones_out(as_rollups({})) == set()


def test_no_gpu_colour_is_the_limit_red() -> None:
    import colorsys

    def hue(hex_colour: str) -> float:
        r, gg, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return colorsys.rgb_to_hls(r, gg, b)[0] * 360

    for i in range(8):
        h = hue(gpu_color(i))
        assert not (h < 15 or h > 345), (i, gpu_color(i))


def test_rows_table_shows_absence_not_zero() -> None:
    gpus = _gpus()
    gpus[1].update(
        {
            "util_now": None,
            "mem_used": None,
            "mem_total": None,
            "temp": None,
            "power": None,
            "power_limit": None,
        }
    )
    html = rows_html(as_rollups({"gpus": gpus}).gpus, [], as_rollups({}))
    assert "gpu1" in html
    assert "n/a" in html
    assert "0 / 0" not in html


def test_whole_run_charts_share_one_clock_axis() -> None:
    """A vertical read across the pair must land on the same moment.

    The two series reach back different distances (the CPU rolling mean
    drops its first partial window, the power buckets do not), so pinning
    each chart to its own span would offset them by a window. The run's
    length itself is not repeated in the labels: the context strip states
    it once, and the axis now carries the clock.
    """
    from nicegui import ui

    from traceml_ai.aggregator.display_drivers.nicegui_sections.system_section import (  # noqa: E501
        build_system_section,
        update_system_section,
    )

    base = 1_755_000_000.0
    cpu_t = [base + 120.0 + 60.0 * i for i in range(40)]  # starts later
    pwr_t = [base + 30.0 * i for i in range(80)]  # starts earlier
    with ui.element("div"):
        panel = build_system_section()
    payload = {
        "window_len": 2,
        "gpu_available": True,
        "rollups": {
            "cpu": {"now": 30.0, "p50": 32.0, "p95": 44.0},
            "ram": {"now": 9.0 * GB, "total": 200.0 * GB},
            "gpu_util": {"now": 99.0, "p50": 99.0, "p95": 99.0},
            "gpu_delta": {"now": 0.0, "p95": 0.0},
            "gpu_mem": {"now": 6.3 * GB, "total": 16.1 * GB},
            "temp": {"now": 48.0, "status": "OK"},
            "gpu_power": {
                "now": 68.0,
                "p50": 67.0,
                "limit": 70.0,
                "floor": 30.0,
            },
            "gpus": _gpus()[:1],
            "ctx": {"gpu_count": 1},
        },
        "series": {
            "x_time": [
                "2026-08-21T10:00:00+00:00",
                "2026-08-21T10:03:20+00:00",
            ],
            "cpu": [30.0, 31.0],
            "gpu_avg": [99.0, 99.0],
            "gpu_power": [{"gpu_idx": 0, "values": [66.0, 68.0]}],
            # Both flags are computed now, so a hand-built payload states
            # which view each chart is in rather than the card inferring it.
            "cpu_run_mode": "retained",
            "power_run_mode": "retained",
            "cpu_run": {
                "t": cpu_t,
                "avg": [30.0] * 40,
                "max": [44.0] * 40,
                "span_s": cpu_t[-1] - cpu_t[0],
                "window_s": 120.0,
            },
            "gpu_power_run": [
                {
                    "gpu_idx": 0,
                    "t": pwr_t,
                    "avg": [67.0] * 80,
                    "min": [57.0] * 80,
                    "max": [69.0] * 80,
                    "span_s": pwr_t[-1] - pwr_t[0],
                    "window_s": 120.0,
                }
            ],
        },
    }
    update_system_section(panel, as_payload(payload))
    cpu_axis = panel["cpu_chart"].options["xAxis"]
    pwr_axis = panel["power_chart"].options["xAxis"]
    assert (cpu_axis["min"], cpu_axis["max"]) == (
        pwr_axis["min"],
        pwr_axis["max"],
    )
    # The span reaches back to the earlier of the two starts.
    assert cpu_axis["min"] == -(
        max(cpu_t[-1], pwr_t[-1]) - min(cpu_t[0], pwr_t[0])
    )
    # Same clock in both formatters, so equal x means equal wall time.
    assert (
        cpu_axis["axisLabel"][":formatter"]
        == pwr_axis["axisLabel"][":formatter"]
    )
    # The duration is the strip's fact; the labels say the view, not the
    # length.
    assert panel["cpu_label"].text == (
        "host cpu util · avg across cores · whole run · rolling 2 min"
    )
    assert panel["power_label"].text == (
        "gpu power · per GPU vs 70 W limit · whole run · "
        "average and lowest every 2 min"
    )
    assert "min," not in panel["cpu_label"].text


def test_power_chart_draws_limit_and_floor_reference_lines() -> None:
    """The band the GPUs actually work in needs both edges.

    The limit alone says how much headroom is unused; the run's lowest
    draw says where 'this GPU is waiting' sits on the same axis, so a
    dip toward it reads as a stall rather than as a small number.
    """
    from nicegui import ui

    from traceml_ai.aggregator.display_drivers.nicegui_sections.system_section import (  # noqa: E501
        build_system_section,
        update_system_section,
    )

    _last_refs: dict = {}

    def refs_for(power: dict) -> list:
        with ui.element("div"):
            panel = build_system_section()
        payload = {
            "window_len": 2,
            "gpu_available": True,
            "rollups": {
                "cpu": {"now": 9.0, "p50": 8.0, "p95": 12.0},
                "ram": {"now": 9.0 * GB, "total": 200.0 * GB},
                "gpu_util": {"now": 25.0, "p50": 25.0, "p95": 25.0},
                "gpu_delta": {"now": 0.0, "p95": 0.0},
                "gpu_mem": {"now": 6.67 * GB, "total": 16.1 * GB},
                "temp": {"now": 54.0, "status": "OK"},
                "gpu_power": power,
                "gpus": _gpus(),
                "ctx": {"gpu_count": 2},
            },
            "series": {
                "x_time": [
                    "2026-08-21T10:00:00+00:00",
                    "2026-08-21T10:03:20+00:00",
                ],
                "cpu": [8.0, 9.0],
                "gpu_avg": [25.0, 25.0],
                "gpu_power": [{"gpu_idx": 0, "values": [66.0, 68.0]}],
            },
        }
        update_system_section(panel, as_payload(payload))
        mark = panel["power_chart"].options["series"][0]["markLine"]
        _last_refs.clear()
        _last_refs.update(mark)
        return [(r["yAxis"], r["label"]["formatter"]) for r in mark["data"]]

    both = refs_for({"now": 68.0, "p50": 67.0, "limit": 70.0, "floor": 33.0})
    assert both == [(70.0, "70 W limit"), (33.0, "33 W lowest seen")]
    # Opposite corners, so neither label lands on the other.
    data = _last_refs["data"]
    assert [r["label"]["position"] for r in data] == [
        "insideEndTop",
        "insideStartBottom",
    ]
    # The two lines carry different colours: the limit keeps the red.
    # A floor that sits at the limit would draw two lines on top of each
    # other and say nothing, so it is dropped.
    assert refs_for(
        {"now": 68.0, "p50": 67.0, "limit": 70.0, "floor": 69.0}
    ) == [(70.0, "70 W limit")]
    # A board that reports no limit still gets the floor.
    assert refs_for(
        {"now": 68.0, "p50": 67.0, "limit": None, "floor": 33.0}
    ) == [(33.0, "33 W lowest seen")]


def test_section_builds_and_updates_without_a_browser() -> None:
    from nicegui import ui

    from traceml_ai.aggregator.display_drivers.nicegui_sections.system_section import (  # noqa: E501
        build_system_section,
        update_system_section,
    )

    with ui.element("div"):
        panel = build_system_section()
    payload = {
        "window_len": 2,
        "gpu_available": True,
        "rollups": {
            "cpu": {"now": 9.0, "p50": 8.0, "p95": 12.0},
            "ram": {"now": 9.0 * GB, "total": 200.0 * GB},
            "gpu_util": {"now": 0.0, "p50": 25.0, "p95": 25.0},
            "gpu_delta": {"now": 0.0, "p95": 100.0},
            # The computer decides both of these now, so a hand-built
            # payload has to carry its answers rather than let the card
            # re-derive them from the rows.
            "rows_over": True,
            "util_range": (0.0, 100.0),
            "gpu_mem": {"now": 6.67 * GB, "total": 16.1 * GB},
            "temp": {"now": 54.0, "status": "OK"},
            "gpu_power": {"now": 68.0, "p50": 67.0, "limit": 70.0},
            "gpus": _gpus(),
            "ctx": {"gpu_count": 4},
        },
        "series": {
            "x_time": [
                "2026-08-21T10:00:00+00:00",
                "2026-08-21T10:03:20+00:00",
            ],
            "cpu": [8.0, 9.0],
            "gpu_avg": [25.0, 25.0],
            "gpu_power": [
                {"gpu_idx": 0, "values": [66.0, 68.0]},
                {"gpu_idx": 1, "values": [33.0, 33.0]},
            ],
        },
    }
    update_system_section(panel, as_payload(payload))
    # The util tile is the window median, not the raw last tick (0).
    assert "25" in panel["tiles"]["util"].content
    assert "0<" not in panel["tiles"]["util"].content
    assert "6.7" in panel["tiles"]["mem"].content
    assert "16.1" in panel["tiles"]["mem"].content
    assert "54" in panel["tiles"]["temp"].content
    assert "200" in panel["tiles"]["ram"].content
    assert panel["cpu_value"].text == "8%"
    assert panel["rows"].value is True  # spread 100 crossed the bar
    assert (
        panel["rows_hint"].text == "2 GPUs · util 0 to 100% · click to close"
    )
    # The window is named once in each chart's label, never on the axis.
    assert (
        panel["cpu_label"].text
        == "host cpu util · avg across cores · last 3 min"
    )
    assert panel["power_label"].text.endswith("70 W limit · last 3 min")
    assert panel["cpu_chart"].options["xAxis"]["axisLabel"]["show"] is True
    assert (
        "getHours"
        in panel["cpu_chart"].options["xAxis"]["axisLabel"][":formatter"]
    )
    assert ":formatter" in (
        panel["cpu_chart"].options["tooltip"]["axisPointer"]["label"]
    )
    # One reference line, the board limit, since this payload reports no
    # run floor; each line carries its own colour and label.
    refs = panel["power_chart"].options["series"][0]["markLine"]["data"]
    assert [(r["yAxis"], r["label"]["formatter"]) for r in refs] == [
        (70.0, "70 W limit")
    ]

    # The user closes the rows; the next identical tick must neither
    # reopen them nor describe them as expanded, and must not re-send
    # the charts.
    panel["rows"].value = False
    sent = []
    panel["cpu_chart"].update = lambda: sent.append("cpu")
    panel["power_chart"].update = lambda: sent.append("power")
    update_system_section(panel, as_payload(payload))
    assert panel["rows"].value is False
    assert panel["rows_hint"].text == "2 GPUs · util 0 to 100% · click to open"
    assert sent == []

    # The limit disappears on a later tick: the mark line is cleared
    # explicitly (ECharts merges options, omitting the key keeps it).
    later = dict(payload)
    later["rollups"] = dict(payload["rollups"])
    later["rollups"]["gpu_power"] = {"now": 68.0, "p50": 67.0, "limit": None}
    later["series"] = dict(payload["series"])
    later["series"]["x_time"] = payload["series"]["x_time"][:1] + [
        "2026-08-21T10:03:22+00:00"
    ]
    update_system_section(panel, as_payload(later))
    # (option mutations also notify the element, so count kinds not calls)
    assert set(sent) == {"cpu", "power"}
    assert panel["power_chart"].options["series"][0]["markLine"] == {
        "data": []
    }
    assert panel["power_label"].text == "gpu power · per GPU · last 3 min"

    # The GPUs may still report memory and temperature while no current
    # utilisation reading is available. The window median must not turn
    # that missing latest reading into a measured 0%.
    no_util = dict(later)
    no_util["rollups"] = dict(later["rollups"])
    no_util["rollups"]["util_gpu_count"] = 0
    no_util["rollups"]["gpus"] = [
        dict(row, util_now=None) for row in later["rollups"]["gpus"]
    ]
    update_system_section(panel, as_payload(no_util))
    assert panel["tiles"]["util"].content == "n/a"
    assert panel["tiles"]["mem"].content != "n/a"

    # Every GPU unreported on the newest tick (represented here by legacy
    # all-zero rows): level tiles say so instead of printing 0 °C / 0.0 GB.
    zeroed = dict(later)
    zeroed["rollups"] = dict(later["rollups"])
    zeroed["rollups"]["gpus"] = [
        {
            "gpu_idx": i,
            "util_now": None,
            "util_p50": 100.0,
            "mem_used": None,
            "mem_total": None,
            "temp": None,
            "power": None,
            "power_limit": None,
            "reported": False,
        }
        for i in range(2)
    ]
    update_system_section(panel, as_payload(zeroed))
    assert panel["tiles"]["util"].content == "n/a"
    assert panel["subs"]["util"].text == "GPU sample unreported"
    assert panel["tiles"]["mem"].content == "n/a"
    assert panel["subs"]["temp"].text == "GPU sample unreported"

    # A CPU-only box hides the GPU tiles, the power chart and the rows.
    update_system_section(
        panel,
        as_payload(
            {
                "window_len": 1,
                "gpu_available": False,
                "rollups": {
                    "cpu": {"now": 3.0, "p50": 3.0},
                    "ram": {"now": 1.0 * GB, "total": 16.0 * GB},
                    "gpu_power": {"now": None, "p50": None, "limit": None},
                    "gpus": [],
                },
                "series": {
                    "x_time": ["2026-08-21T10:00:00+00:00"],
                    "cpu": [3.0],
                    "gpu_avg": [],
                    "gpu_power": [],
                },
            }
        ),
    )
    assert panel["gpu_visible"] is False
    # The four tiles stay in place; the GPU ones read a dash and say why.
    for key in ("util", "mem", "temp"):
        assert panel["tiles"][key].content == "n/a"
        assert panel["subs"][key].text == "no GPU"
    assert "16.0" in panel["tiles"]["ram"].content  # the RAM tile still works
    # The chart and rows slots keep their place with a one-line state.
    assert panel["power_label"].text == "gpu power"
    assert panel["power_placeholder"].text == "no GPU reported"
    assert panel["rows_placeholder"].text == "per-GPU rows · no GPU"


def test_header_names_the_node_when_others_were_dropped() -> None:
    from traceml_ai.aggregator.display_drivers.nicegui_sections.system_section import (  # noqa: E501
        node_scope_text,
    )

    assert node_scope_text(as_rollups({}).ctx) == ""
    one = {"hostname": "a", "node_rank": 0, "nodes_in_window": 1}
    two = {"hostname": "a", "node_rank": 0, "nodes_in_window": 2}
    assert node_scope_text(_ctx({"system_node": one})) == ""
    assert node_scope_text(_ctx({"system_node": two})) == "node 0 of 2"
    # Only the System payload's own facts are read: a strip-style
    # node_count on the dict changes nothing.
    assert node_scope_text(_ctx({"system_node": one, "node_count": 2})) == ""


def test_every_tile_keeps_a_qualifier_line() -> None:
    """A blank qualifier made one tile a line shorter than its neighbours.

    On a single-GPU host the temperature tile had nothing to say under it
    ("max GPU" is meaningless with one GPU), so the box lost its third
    line and the row of four read as uneven.
    """
    from nicegui import ui

    from traceml_ai.aggregator.display_drivers.nicegui_sections.system_section import (  # noqa: E501
        build_system_section,
        update_system_section,
    )

    def payload(n_gpus: int) -> dict:
        return {
            "window_len": 2,
            "gpu_available": True,
            "rollups": {
                "cpu": {"now": 9.0, "p50": 8.0},
                "ram": {"now": 9.0 * GB, "total": 200.0 * GB},
                "gpu_util": {"now": 99.0, "p50": 99.0},
                # The four-GPU case deliberately has one missing reading so
                # the qualifier's newest-tick scope remains explicit.
                "util_gpu_count": 1 if n_gpus == 1 else 3,
                "gpu_delta": {"now": 0.0, "p95": 0.0},
                "gpu_mem": {"now": 6.3 * GB, "total": 16.1 * GB},
                "temp": {"now": 48.0},
                "gpu_power": {"now": 66.0, "p50": 66.0, "limit": 70.0},
                "gpus": [
                    {
                        "gpu_idx": i,
                        "util_now": 99.0,
                        "util_p50": 99.0,
                        "mem_used": 6.3 * GB,
                        "mem_total": 16.1 * GB,
                        "temp": 48.0,
                        "power": 66.0,
                        "power_limit": 70.0,
                        "reported": True,
                    }
                    for i in range(n_gpus)
                ],
            },
            "series": {
                "x_time": [
                    "2026-08-21T10:00:00+00:00",
                    "2026-08-21T10:03:20+00:00",
                ],
                "cpu": [8.0, 9.0],
                "gpu_avg": [99.0, 99.0],
                "gpu_power": [
                    {"gpu_idx": i, "values": [66.0, 66.0]}
                    for i in range(n_gpus)
                ],
            },
        }

    for n in (1, 4):
        with ui.element("div"):
            panel = build_system_section()
        update_system_section(panel, as_payload(payload(n)))
        subs = {
            k: panel["subs"][k].text for k in ("util", "mem", "temp", "ram")
        }
        assert all(subs.values()), (n, subs)
        if n == 1:
            assert subs["util"] == "1 GPU" and subs["temp"] == "1 GPU"
            assert subs["mem"] == "used / total"
        else:
            assert subs["util"] == "latest: 3/4 reporting"
            assert subs["mem"] == "max GPU" and subs["temp"] == "max GPU"


def test_rolling_window_is_spelled_out_not_named() -> None:
    """A whole-run point smooths a stretch of the run; say how long it is.

    "rolling 2 min" needs no glossary; "per slice" made a reader ask what a
    slice was, and disjoint slices did not smooth a fast oscillation anyway.
    """
    from traceml_ai.renderers.shared.run_series import (
        DEFAULT_RUN_SERIES_POLICY,
    )

    window_for = DEFAULT_RUN_SERIES_POLICY.window_for

    assert format_window(window_for(96 * 60)) == "2 min"  # 96-min run
    assert format_window(window_for(178 * 60)) == "5 min"  # 3-hour
    assert format_window(window_for(23 * 60)) == "30 s"
    assert format_window(window_for(4 * 60)) == "30 s"  # the floor
    assert format_window(window_for(48 * 3600)) == "5 min"  # the cap
    assert format_window(0.0) == ""


def test_a_small_real_reading_is_not_printed_as_zero() -> None:
    """Sub-gigabyte levels retain useful precision, matching Process."""
    from traceml_ai.aggregator.display_drivers.nicegui_sections import (
        system_section,
    )

    assert system_section.format_gb_pair(27e6, 16.1e9)[0] == "0.03"
    assert system_section.format_gb_pair(500e6, 16.1e9)[0] == "0.50"
    # At and above a gigabyte the extra digit is noise, so one decimal.
    assert system_section.format_gb_pair(6.3e9, 16.1e9)[0] == "6.3"
    # Absence remains distinct from every measured value.
    assert system_section.format_gb_pair(None, 16.1e9)[0] == "n/a"


def _tile_number(content: str) -> str:
    """A tile's value with its unit markup stripped off.

    ``theme.kval`` wraps the unit in its own span, and a unit can carry a
    digit (the GPU rows do). The value is what must not be a number when
    nothing was measured.
    """
    return re.sub(r"<span class='kunit'>.*?</span>", "", content)


def test_tiles_abstain_when_the_payload_carries_no_measurement() -> None:
    """Every tile says n/a when its field is absent.

    A freeze test, and deliberately one: this path is already correct and
    the change that accompanies it is in the compute layer, which used to
    send 0.0 where it now sends None. Pinning the card here is what makes
    that change safe to state as a fix rather than a swap of one wrong
    value for another.

    The devices are reported: they carry a memory total and a power limit.
    That is the case the per-device guard does not cover, because the
    guard asks whether the DEVICE said anything, not whether the metric
    did.
    """
    from nicegui import ui

    from traceml_ai.aggregator.display_drivers.nicegui_sections.system_section import (  # noqa: E501
        build_system_section,
        update_system_section,
    )

    with ui.element("div"):
        panel = build_system_section()
    payload = {
        "window_len": 20,
        "gpu_available": True,
        "rollups": {
            "cpu": {"now": None, "p50": None, "p95": None},
            "ram": {"now": None, "p95": None, "total": None, "headroom": None},
            "gpu_util": {"now": None, "p50": None, "p95": None},
            "gpu_delta": {"now": None, "p95": None},
            "gpu_mem": {
                "now": None,
                "p95": None,
                "headroom": None,
                "total": None,
            },
            "temp": {"now": None, "p95": None, "status": None},
            "gpu_power": {
                "now": None,
                "p50": None,
                "limit": None,
                "floor": None,
            },
            "gpus": [
                {
                    "gpu_idx": i,
                    "mem_total": 16.1 * GB,
                    "power_limit": 70.0,
                    "reported": True,
                }
                for i in range(4)
            ],
            "ctx": {"gpu_count": 4},
            "util_gpu_count": 0,
        },
        "series": {"x_time": [], "cpu": [], "gpu_avg": []},
    }

    update_system_section(panel, as_payload(payload))

    # The invariant is that no tile shows a number, not the exact string:
    # the temperature tile keeps its unit beside the marker ("n/a °C")
    # while the byte tiles drop theirs, which is a pre-existing difference
    # in this card and not what this test is pinning.
    for key in ("ram", "util", "mem", "temp"):
        content = panel["tiles"][key].content
        assert "n/a" in content, key
        assert not re.search(r"\d", _tile_number(content)), key
    # Not "0%", which is what a measured idle host reads.
    assert panel["cpu_value"].text == ""


def test_a_widened_bucket_reads_as_hours_not_a_big_minute_count() -> None:
    """Long runs now produce widths past an hour, so the label must too.

    Before the whole-run power chart was capped, its width could not
    exceed the 300 s window ceiling, so this branch was unreachable and
    minutes were always enough. Capping widens the bucket instead of
    growing the count, which puts multi-hour widths on the label: a seven
    day run buckets at 5040 s, and "84 min" is not how a person reads it.
    """
    assert format_window(120.0) == "2 min"
    assert format_window(719.0) == "12 min"
    assert format_window(1440.0) == "24 min"
    assert format_window(3600.0) == "1 h"
    assert format_window(5040.0) == "1 h 24 min"
    assert format_window(7200.0) == "2 h"
    # Unchanged below a minute, and still empty for nothing.
    assert format_window(30.0) == "30 s"
    assert format_window(0.0) == ""
