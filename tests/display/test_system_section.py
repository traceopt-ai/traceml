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

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui_sections.system_section import (  # noqa: E402,E501
    SPREAD_EXPAND_PTS,
    cpu_axis_max,
    disclosure_text,
    format_gb_pair,
    format_span,
    gpu_color,
    odd_ones_out,
    power_axis_bounds,
    rows_html,
    should_auto_open,
    sparkline_svg,
)

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
    assert format_gb_pair(0.47 * GB, None) == ("0.5", "GB")


def test_disclosure_text_speaks_in_gpu_words() -> None:
    def g(i, u):
        return {"gpu_idx": i, "util_p50": u}

    one_busy = [g(0, 100), g(1, 0), g(2, 0), g(3, 0)]
    all_busy = [g(i, 100) for i in range(4)]
    ramp = [g(0, 98), g(1, 100), g(2, 100), g(3, 100)]
    # The trigger fired and GPUs sit idle: say which.
    assert disclosure_text(one_busy, over=True, is_open=True) == (
        "1 of 4 GPUs busy, 3 idle · click to close"
    )
    # A user closed the rows while the spread stays over the bar: the
    # words keep the fact, the tail follows the real state.
    assert disclosure_text(one_busy, over=True, is_open=False) == (
        "1 of 4 GPUs busy, 3 idle · click to open"
    )
    # The trigger fired on a ramp (nobody idle): honest, no mechanism.
    assert disclosure_text(ramp, over=True, is_open=True) == (
        "uneven load across GPUs · click to close"
    )
    assert disclosure_text(all_busy, over=False, is_open=False) == (
        "all 4 GPUs alike · click to open"
    )
    assert disclosure_text(all_busy, over=False, is_open=True) == (
        "all 4 GPUs alike · click to close"
    )
    assert disclosure_text([g(0, 99)], over=False, is_open=False) == (
        "1 GPU · click to open"
    )
    assert disclosure_text([], over=False, is_open=False) == ""
    # No mechanism words anywhere in the header.
    for text in (
        disclosure_text(one_busy, over=True, is_open=True),
        disclosure_text(all_busy, over=False, is_open=False),
    ):
        for banned in ("spread", "auto", ">", "20"):
            assert banned not in text
    assert SPREAD_EXPAND_PTS == 20.0


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
        },
    ]


def test_rows_table_reads_per_gpu_and_tints_only_the_busy_row() -> None:
    series = [
        {"gpu_idx": 0, "values": [66.0, 68.0]},
        {"gpu_idx": 1, "values": [33.0, 33.0]},
    ]
    html = rows_html(_gpus(), series, spread=100.0)
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
    calm = rows_html(_gpus(), series, spread=0.0)
    assert "tml-mark" not in calm
    # No verdict words anywhere on the block.
    for word in ("Hot", "Warm", "HIGH", "verdict"):
        assert word not in html


def test_tint_marks_the_odd_ones_out() -> None:
    def g(i, u):
        return {"gpu_idx": i, "util_p50": u, "util_now": u}

    # 1 busy of 4: the busy GPU is the anomaly.
    assert odd_ones_out([g(0, 100), g(1, 0), g(2, 0), g(3, 0)]) == {0}
    # 3 busy of 4 (a starved or dead rank): the idle GPU is the anomaly,
    # not the first busy row.
    assert odd_ones_out([g(0, 100), g(1, 100), g(2, 100), g(3, 0)]) == {3}
    # 2 and 2: ties go to the busy side.
    assert odd_ones_out([g(0, 100), g(1, 0), g(2, 100), g(3, 0)]) == {0, 2}
    # Nothing to mark when every GPU reads the same, or only one reports.
    assert odd_ones_out([g(0, 100), g(1, 100)]) == set()
    assert odd_ones_out([g(0, 100), {"gpu_idx": 1}]) == set()


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
    html = rows_html(gpus, [], spread=None)
    assert "gpu1" in html
    assert "n/a" in html
    assert "0 / 0" not in html


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
            "gpu_available": True,
            "cpu": {"now": 9.0, "p50": 8.0, "p95": 12.0},
            "ram": {"now": 9.0 * GB, "total": 200.0 * GB},
            "gpu_util": {"now": 0.0, "p50": 25.0, "p95": 25.0},
            "gpu_delta": {"now": 0.0, "p95": 100.0},
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
    update_system_section(panel, payload)
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
        panel["rows_hint"].text == "1 of 2 GPUs busy, 1 idle · click to close"
    )
    # The window is named once in each chart's label, never on the axis.
    assert panel["cpu_label"].text == "host cpu % · last 3 min"
    assert panel["power_label"].text.endswith("70 W limit · last 3 min")
    assert panel["cpu_chart"].options["xAxis"]["axisLabel"]["show"] is False
    assert ":formatter" in (
        panel["cpu_chart"].options["tooltip"]["axisPointer"]["label"]
    )
    assert panel["power_chart"].options["series"][0]["markLine"]["data"] == [
        {"yAxis": 70.0}
    ]

    # The user closes the rows; the next identical tick must neither
    # reopen them nor describe them as expanded, and must not re-send
    # the charts.
    panel["rows"].value = False
    sent = []
    panel["cpu_chart"].update = lambda: sent.append("cpu")
    panel["power_chart"].update = lambda: sent.append("power")
    update_system_section(panel, payload)
    assert panel["rows"].value is False
    assert (
        panel["rows_hint"].text == "1 of 2 GPUs busy, 1 idle · click to open"
    )
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
    update_system_section(panel, later)
    # (option mutations also notify the element, so count kinds not calls)
    assert set(sent) == {"cpu", "power"}
    assert panel["power_chart"].options["series"][0]["markLine"] == {
        "data": []
    }
    assert panel["power_label"].text == "gpu power · per GPU · last 3 min"

    # Every GPU unreported on the newest tick (the sampler's all-zero
    # fallback): level tiles say so instead of printing 0 °C / 0.0 GB.
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
        }
        for i in range(2)
    ]
    update_system_section(panel, zeroed)
    assert panel["tiles"]["mem"].content == "n/a"
    assert panel["subs"]["temp"].text == "GPU sample unreported"

    # A CPU-only box hides the GPU tiles, the power chart and the rows.
    update_system_section(
        panel,
        {
            "window_len": 1,
            "gpu_available": False,
            "rollups": {
                "gpu_available": False,
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
        },
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

    assert node_scope_text({}) == ""
    one = {"hostname": "a", "node_rank": 0, "nodes_in_window": 1}
    two = {"hostname": "a", "node_rank": 0, "nodes_in_window": 2}
    assert node_scope_text({"system_node": one}) == ""
    assert node_scope_text({"system_node": two}) == "node 0 of 2"
    # Only the System payload's own facts are read: a strip-style
    # node_count on the dict changes nothing.
    assert node_scope_text({"system_node": one, "node_count": 2}) == ""


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
                "gpu_available": True,
                "cpu": {"now": 9.0, "p50": 8.0},
                "ram": {"now": 9.0 * GB, "total": 200.0 * GB},
                "gpu_util": {"now": 99.0, "p50": 99.0},
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
        update_system_section(panel, payload(n))
        subs = {
            k: panel["subs"][k].text for k in ("util", "mem", "temp", "ram")
        }
        assert all(subs.values()), (n, subs)
        if n == 1:
            assert subs["util"] == "1 GPU" and subs["temp"] == "1 GPU"
            assert subs["mem"] == "used / total"
        else:
            assert subs["util"] == "avg of 4 GPUs"
            assert subs["mem"] == "max GPU" and subs["temp"] == "max GPU"
