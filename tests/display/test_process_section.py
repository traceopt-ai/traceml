# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""What the Process card puts on screen for a given payload.

The card this file describes is the one this PR builds: four tiles, two
per-rank charts and a per-rank table. The previous card's assertions are
NOT carried over unchanged, because the card deliberately says different
things now. The two differences worth naming:

* The single ``GPU MEM`` tile becomes the ``cuda allocated`` and ``cuda
  reserved`` pair. One tile could not say which of the two it held, and the
  two numbers answer different questions: allocated is what the tensors
  need, reserved is what the process is holding from the device.
* The ``CPU`` and ``RAM`` tiles become ``cpu`` and ``rss``. CPU keeps its
  meaning in the value: a raw 700% process reading is not comparable between
  hosts, while 87.5% of the host is.

``test_the_card_renders_the_same_from_a_real_database`` closes the loop
from database through compute to screen, and states both new meanings as
explicit numbers so the change is visible rather than implied.

#511 rewrote the words, not the numbers. The card is titled for the
trainer process, one rank reads as usage rather than as a comparison,
many ranks name the rank each tile selected, a CPU-only host loses its
CUDA tiles once data has arrived, and each graph highlights the rank its
tile selected with the others muted. The window medians are unchanged.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui_sections import (  # noqa: E402
    charting,
    process_section,
    theme,
)
from traceml_ai.renderers.process.dashboard_compute import (  # noqa: E402
    RECENT_WINDOW_S,
    ProcessDashboardComputer,
)
from traceml_ai.renderers.process.dashboard_models import (  # noqa: E402
    CudaHeadroomSample,
    MetricRollup,
    ProcessDashboardPayload,
    RankChart,
    RankCoverage,
    RankSnapshot,
    RankTrace,
)

GB = 1_000_000_000.0
GIB = float(1024**3)


class _El:
    """Records what the card asks of an element: its style and classes."""

    def __init__(self) -> None:
        self.styles: list = []
        self.class_names: set = set()

    def style(self, text: str) -> "_El":
        self.styles.append(text)
        return self

    def classes(self, add=None, *, remove=None) -> "_El":
        for name in (remove or "").split():
            self.class_names.discard(name)
        for name in (add or "").split():
            self.class_names.add(name)
        return self

    @property
    def hidden(self) -> bool:
        for text in reversed(self.styles):
            match = re.search(r"display:\s*([\w-]+)", text)
            if match:
                return match.group(1) == "none"
        return False


class _Html(_El):
    def __init__(self) -> None:
        super().__init__()
        self.content = ""


class _Text(_El):
    def __init__(self) -> None:
        super().__init__()
        self.text = ""


class _Expansion(_El):
    def __init__(self) -> None:
        super().__init__()
        self.value = False


class _Chart:
    """The real option dict the card builds, without the element."""

    def __init__(self, unit: str) -> None:
        self.options = charting.multi_line_options(unit)
        self.updates = 0

    def update(self) -> None:
        self.updates += 1


_TILES = ("cpu", "rss", "reserved", "alloc")


def _panel() -> dict:
    return {
        "title": _Text(),
        "context": _Text(),
        "duration": _Text(),
        "tilerow": _El(),
        "tile_els": {k: _El() for k in _TILES},
        "tile_labels": {k: _Text() for k in _TILES},
        "tiles": {k: _Html() for k in _TILES},
        "subs": {k: _Text() for k in _TILES},
        "tile_tips": {k: _Text() for k in ("cpu", "rss")},
        "cpu_chart": _Chart("%"),
        "rss_chart": _Chart(" GB"),
        "cpu_label": _Text(),
        "rss_label": _Text(),
        "cpu_sub": _Text(),
        "rss_sub": _Text(),
        "cpu_tip": _Text(),
        "rss_tip": _Text(),
        "rows": _Expansion(),
        "rows_title": _Text(),
        "rows_hint": _Text(),
        "rows_html": _Html(),
        "_was_open": False,
        "_signature": None,
    }


def _shown(panel: dict) -> str:
    """Every string the card puts on screen, hidden elements left out."""
    parts = [
        panel[key].text
        for key in (
            "title",
            "context",
            "duration",
            "cpu_label",
            "rss_label",
            "cpu_sub",
            "rss_sub",
            "cpu_tip",
            "rss_tip",
        )
    ]
    for key, tile in panel["tile_els"].items():
        if tile.hidden:
            continue
        parts += [
            panel["tile_labels"][key].text,
            panel["tiles"][key].content,
            panel["subs"][key].text,
        ]
        if key in panel["tile_tips"]:
            parts.append(panel["tile_tips"][key].text)
    if not panel["rows"].hidden:
        parts += [
            panel["rows_title"].text,
            panel["rows_hint"].text,
            panel["rows_html"].content,
        ]
    for chart in ("cpu_chart", "rss_chart"):
        parts += [s["name"] for s in panel[chart].options["series"]]
    return "\n".join(parts)


def _primary(panel: dict) -> str:
    """The card's own words, without its tooltips."""
    parts = [
        panel[key].text
        for key in ("title", "context", "duration", "cpu_label", "rss_label")
    ]
    parts += [panel["cpu_sub"].text, panel["rss_sub"].text]
    for key in _TILES:
        parts += [
            panel["tile_labels"][key].text,
            panel["tiles"][key].content,
            panel["subs"][key].text,
        ]
    parts += [panel["rows_title"].text, panel["rows_hint"].text]
    parts.append(panel["rows_html"].content)
    return "\n".join(parts)


def _rank(
    index: int,
    *,
    capacity: float = 25.0,
    rss: float = 2.0 * GIB,
    reserved: float = 6.0 * GIB,
    allocated: float = 4.0 * GIB,
    freshness: str = "fresh",
    age_s: float = 2.0,
) -> RankSnapshot:
    return RankSnapshot(
        global_rank=index,
        node_rank=0,
        gpu_index=index,
        cpu_capacity_percent=capacity,
        ram_used_bytes=rss,
        ram_used_p50_bytes=rss,
        ram_total_bytes=64.0 * GIB,
        gpu_reserved_p50_bytes=reserved,
        cuda_least_headroom_sample=CudaHeadroomSample(
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            total_bytes=40.0 * GIB,
        ),
        gpu_total_bytes=40.0 * GIB,
        age_s=age_s,
        freshness=freshness,
    )


def _chart(*ranks: int, mode: str = "recent") -> RankChart:
    stamps = (1_700_000_000.0, 1_700_000_002.0, 1_700_000_004.0)
    return RankChart(
        mode=mode,
        window_s=120.0 if mode == "retained" else None,
        traces=tuple(
            RankTrace(
                global_rank=index,
                timestamps=stamps,
                values=(20.0 + index, 21.0 + index, 22.0 + index),
            )
            for index in ranks
        ),
    )


def _payload(
    *,
    ranks=(0, 1),
    gpu: bool = True,
    imbalance=None,
    rows_open: bool = False,
    stale: int = 0,
    unknown: int = 0,
) -> ProcessDashboardPayload:
    snapshots = tuple(
        _rank(
            i,
            reserved=(6.0 * GIB if gpu else None),
            allocated=(4.0 * GIB if gpu else None),
        )
        for i in ranks
    )
    if not gpu:
        snapshots = tuple(
            RankSnapshot(
                global_rank=r.global_rank,
                node_rank=r.node_rank,
                cpu_capacity_percent=r.cpu_capacity_percent,
                ram_used_bytes=r.ram_used_bytes,
                ram_used_p50_bytes=r.ram_used_p50_bytes,
                ram_total_bytes=r.ram_total_bytes,
                age_s=r.age_s,
                freshness=r.freshness,
            )
            for r in snapshots
        )
    return ProcessDashboardPayload(
        window_len=3,
        ranks=snapshots,
        coverage=RankCoverage(
            total=len(snapshots),
            live=len(snapshots) - stale - unknown,
            stale=stale,
            unknown=unknown,
        ),
        cpu_capacity=MetricRollup(
            now=87.5, p95=87.5, p50=25.0, worst_rank=ranks[-1]
        ),
        rss_worst=MetricRollup(
            now=2.5 * GIB,
            p95=2.5 * GIB,
            total=64.0 * GIB,
            worst_rank=ranks[0],
        ),
        gpu_reserved=(
            MetricRollup(now=7.0 * GIB, p95=7.0 * GIB, worst_rank=1)
            if gpu
            else None
        ),
        gpu=(MetricRollup(now=6.0 * GIB, p95=6.0 * GIB) if gpu else None),
        gpu_allocated=(
            MetricRollup(now=6.0 * GIB, worst_rank=1) if gpu else None
        ),
        reserved_imbalance_percent=imbalance,
        rows_open=rows_open,
        cpu_capacity_chart=_chart(*ranks),
        rss_chart=_chart(*ranks),
    )


def _retained(payload: ProcessDashboardPayload) -> ProcessDashboardPayload:
    """The same payload with both graphs drawing whole-run history."""
    ranks = tuple(r.global_rank for r in payload.ranks)
    return replace(
        payload,
        cpu_capacity_chart=_chart(*ranks, mode="retained"),
        rss_chart=_chart(*ranks, mode="retained"),
    )


# --- the header ----------------------------------------------------------
def test_one_rank_is_titled_as_one_trainer_process():
    panel = _panel()
    process_section.update_process_section(panel, _payload(ranks=(0,)))
    assert panel["title"].text == "Trainer process"
    assert panel["context"].text == "Rank 0 · Trainer process only"
    assert panel["duration"].text == "Last 4s"


def test_many_ranks_are_titled_as_trainer_processes():
    panel = _panel()
    process_section.update_process_section(panel, _payload(ranks=(0, 1, 2)))
    assert panel["title"].text == "Trainer processes"
    assert panel["context"].text == (
        "3 ranks reporting · one trainer process per rank"
    )
    assert panel["duration"].text == "Last 4s"


def test_a_stale_rank_is_not_counted_as_reporting():
    """The tiles exclude it, so the header must not claim it reports."""
    panel = _panel()
    process_section.update_process_section(
        panel, _payload(ranks=(0, 1), stale=1)
    )
    assert panel["context"].text == (
        "1 of 2 ranks reporting · one trainer process per rank"
    )


def test_the_window_is_named_once_in_the_header_not_on_the_graphs():
    panel = _panel()
    process_section.update_process_section(panel, _payload())
    assert "recent 60s" not in panel["title"].text
    for key in ("cpu_label", "rss_label", "cpu_sub", "rss_sub"):
        assert "4s" not in panel[key].text, key
        assert "last" not in panel[key].text.lower(), key


def test_a_whole_run_history_says_its_points_are_rolling_averages():
    """The old per-graph "rolling 2 min" moves to the header, not away."""
    panel = _panel()
    process_section.update_process_section(panel, _retained(_payload()))
    assert panel["duration"].text == "Last 4s · rolling 2 min averages"


def test_before_any_data_the_header_waits():
    """Nothing is known yet, so no rank and no window are named."""
    panel = _panel()
    process_section.update_process_section(panel, ProcessDashboardPayload())
    assert panel["title"].text == "Trainer process"
    assert panel["context"].text == "waiting for data"
    assert panel["duration"].text == ""


# --- the summary tiles ---------------------------------------------------
def test_one_rank_reads_as_usage_not_as_a_comparison():
    panel = _panel()
    process_section.update_process_section(panel, _payload(ranks=(0,)))
    assert panel["tile_labels"]["cpu"].text == "CPU usage"
    assert "87.5" in panel["tiles"]["cpu"].content
    assert "%" in panel["tiles"]["cpu"].content
    assert panel["subs"]["cpu"].text == "of host CPU"
    assert panel["tile_labels"]["rss"].text == "Process memory"
    assert "2.5" in panel["tiles"]["rss"].content
    assert "GB" in panel["tiles"]["rss"].content
    assert panel["subs"]["rss"].text == "of 64.0 GB host RAM"


def test_many_ranks_name_the_rank_each_tile_selected():
    """Same window medians as before; the words name the rank."""
    panel = _panel()
    process_section.update_process_section(panel, _payload())
    assert panel["tile_labels"]["cpu"].text == "Busiest rank CPU"
    assert "87.5" in panel["tiles"]["cpu"].content
    assert panel["subs"]["cpu"].text == "of host CPU · Rank 1"
    assert panel["tile_labels"]["rss"].text == "Highest process memory"
    assert "2.5" in panel["tiles"]["rss"].content
    assert panel["subs"]["rss"].text == "Rank 0"


def test_the_median_is_defined_in_a_tooltip_not_on_the_card():
    many, one = _panel(), _panel()
    process_section.update_process_section(many, _payload())
    process_section.update_process_section(one, _payload(ranks=(0,)))
    for key in ("cpu", "rss"):
        assert many["tile_tips"][key].text == (
            "The displayed rank has the highest median value during the "
            "shown period."
        )
        assert one["tile_tips"][key].text == "Median over the shown period."
    for panel in (many, one):
        assert "median" not in _primary(panel).lower()


def test_on_a_whole_run_graph_the_tooltip_names_the_tile_window():
    """The graphs span the run; the tile medians still cover one minute.

    "The shown period" would then describe the graphs, not the tiles, so
    the tooltip names the tiles' own window instead.
    """
    assert RECENT_WINDOW_S == 60.0, "the tooltip wording says one minute"
    many, one = _panel(), _panel()
    process_section.update_process_section(many, _retained(_payload()))
    process_section.update_process_section(
        one, _retained(_payload(ranks=(0,)))
    )
    assert many["tile_tips"]["cpu"].text == (
        "The displayed rank has the highest median value during the most "
        "recent minute."
    )
    assert one["tile_tips"]["rss"].text == (
        "Median over the most recent minute."
    )


def test_a_gpu_host_shows_all_four_tiles():
    panel = _panel()
    process_section.update_process_section(panel, _payload())
    assert not any(tile.hidden for tile in panel["tile_els"].values())


def test_allocated_and_reserved_are_two_tiles_not_one():
    """The split that #399 settled the wording for.

    A single tile could not say which of the two numbers it held. They are
    different quantities: 6 GiB of live tensors inside 7 GiB the process is
    holding from the device.
    """
    panel = _panel()
    process_section.update_process_section(panel, _payload())
    assert "7.0" in panel["tiles"]["reserved"].content
    assert "6.0" in panel["tiles"]["alloc"].content
    # Read from the per-rank rollup, not the aggregated step history: the
    # history's newest step has no GPU snapshot once a run tears down,
    # which left this tile "n/a" above rows listing each rank's bytes.
    assert panel["tiles"]["alloc"].content != "n/a"
    # Both GPU tiles describe ONE device, so they can be read together.
    assert panel["subs"]["reserved"].text == "least headroom · Rank 1"
    assert panel["subs"]["alloc"].text == "least headroom · Rank 1"


def test_one_rank_on_a_gpu_describes_cuda_without_comparing():
    """With one rank there is no rank to have the least headroom."""
    panel = _panel()
    process_section.update_process_section(panel, _payload(ranks=(0,)))
    assert "7.0" in panel["tiles"]["reserved"].content
    assert panel["subs"]["reserved"].text == "held by the process"
    assert panel["subs"]["alloc"].text == "live tensors"


# --- a CPU-only host -----------------------------------------------------
def test_an_empty_first_tick_keeps_the_four_tile_row():
    """Before any rank reports, nothing is known about the GPU either way.

    The empty payload answers ``gpu_available`` with False only because it
    has no ranks to ask. Hiding the CUDA tiles there would make every GPU
    host's card jump when its first sample arrives.
    """
    panel = _panel()
    process_section.update_process_section(panel, ProcessDashboardPayload())
    assert not any(tile.hidden for tile in panel["tile_els"].values())
    for key in ("reserved", "alloc"):
        assert panel["tiles"][key].content == "n/a"
        assert panel["subs"][key].text == ""


def test_a_cpu_only_run_gives_the_two_summaries_the_whole_row():
    panel = _panel()
    process_section.update_process_section(panel, _payload(gpu=False))
    assert panel["tile_els"]["reserved"].hidden
    assert panel["tile_els"]["alloc"].hidden
    assert not panel["tile_els"]["cpu"].hidden
    assert not panel["tile_els"]["rss"].hidden


@pytest.mark.parametrize("ranks", [(0,), (0, 1)])
def test_a_cpu_only_run_says_nothing_about_a_gpu(ranks):
    """No n/a and no "no GPU", in the tiles or in the rank details."""
    panel = _panel()
    process_section.update_process_section(
        panel, _payload(ranks=ranks, gpu=False)
    )
    shown = _shown(panel)
    assert "n/a" not in shown
    assert "no GPU" not in shown
    assert "cuda" not in shown.lower()


def test_the_tiles_sit_two_to_a_row():
    """Two to a row on every host, so a CPU-only card is the top half.

    Four to a row gave each tile a quarter of a half-width card, where
    "Highest process memory" and "of host CPU · Rank 7" wrap and leave a
    rank number alone on its own line.
    """
    panel = process_section.build_process_section()
    assert "tml-tiles-2" in panel["tilerow"].classes
    assert ".tilerow.tml-tiles-2" in theme.head_html()


def test_a_host_whose_gpu_reports_late_gets_its_tiles_back():
    panel = _panel()
    process_section.update_process_section(panel, _payload(gpu=False))
    process_section.update_process_section(panel, _payload())
    assert not any(tile.hidden for tile in panel["tile_els"].values())


# --- the words the card must not use -------------------------------------
@pytest.mark.parametrize(
    "ranks,gpu", [((0,), True), ((0,), False), ((0, 1), True), ((0, 1), False)]
)
def test_the_card_uses_no_implementation_terms(ranks, gpu):
    panel = _panel()
    process_section.update_process_section(
        panel, _payload(ranks=ranks, gpu=gpu)
    )
    primary = _primary(panel)
    for term in ("RSS", "rss", "highest median", "capacity per rank"):
        assert term not in primary, term
    assert "Total" not in _shown(panel)


@pytest.mark.parametrize("gpu", [True, False])
def test_one_rank_carries_no_comparative_wording(gpu):
    """Tooltips included: there is nothing to compare one rank with."""
    panel = _panel()
    process_section.update_process_section(
        panel, _retained(_payload(ranks=(0,), gpu=gpu))
    )
    process_section.update_process_section(
        panel, _payload(ranks=(0,), gpu=gpu)
    )
    shown = _shown(panel).lower()
    for word in ("busiest", "highest", "per rank", "by rank", "least"):
        assert word not in shown, word


def test_a_payload_of_the_wrong_type_is_ignored():
    panel = _panel()
    process_section.update_process_section(panel, None)
    process_section.update_process_section(panel, {"ranks": []})
    assert panel["tiles"]["cpu"].content == ""


# --- the two charts ------------------------------------------------------
def test_each_rank_is_its_own_line_on_both_charts():
    panel = _panel()
    process_section.update_process_section(panel, _payload(ranks=(0, 1, 2)))
    assert len(panel["cpu_chart"].options["series"]) == 3
    assert len(panel["rss_chart"].options["series"]) == 3


def test_both_charts_share_one_time_axis():
    """A vertical read across the pair has to land on the same moment."""
    panel = _panel()
    process_section.update_process_section(panel, _payload())
    cpu_axis = panel["cpu_chart"].options["xAxis"]
    rss_axis = panel["rss_chart"].options["xAxis"]
    assert (cpu_axis["min"], cpu_axis["max"]) == (
        rss_axis["min"],
        rss_axis["max"],
    )


def test_the_capacity_chart_is_zero_anchored_and_rss_is_not():
    """Different signals, so deliberately different y ranges.

    CPU capacity is a share, so the distance from zero is the reading. RSS
    is a level that drifts, and zero-anchoring it puts the drift inside one
    pixel.
    """
    panel = _panel()
    process_section.update_process_section(panel, _payload())
    assert "min" not in panel["cpu_chart"].options["yAxis"]
    assert panel["rss_chart"].options["yAxis"]["min"] > 0


def _line(panel: dict, chart: str, name: str) -> dict:
    (series,) = [
        s for s in panel[chart].options["series"] if s["name"] == name
    ]
    return series


def test_each_graph_highlights_the_rank_its_tile_selected():
    """CPU follows the busiest rank, memory the highest; the rest mute."""
    panel = _panel()
    process_section.update_process_section(panel, _payload(ranks=(0, 1, 2)))

    assert panel["cpu_label"].text == "CPU usage by rank"
    assert panel["cpu_sub"].text == "Rank 2 highlighted"
    assert panel["rss_label"].text == "Process memory by rank"
    assert panel["rss_sub"].text == "Rank 0 highlighted"

    lead = _line(panel, "cpu_chart", "Rank 2")
    assert lead["lineStyle"]["color"] == theme.C_CPU
    for name in ("Rank 0", "Rank 1"):
        other = _line(panel, "cpu_chart", name)
        assert other["lineStyle"]["color"] == theme.MUTED
        assert other["lineStyle"]["width"] < lead["lineStyle"]["width"]

    lead = _line(panel, "rss_chart", "Rank 0")
    assert lead["lineStyle"]["color"] == theme.C_MEM
    for name in ("Rank 1", "Rank 2"):
        other = _line(panel, "rss_chart", name)
        assert other["lineStyle"]["color"] == theme.MUTED


def test_one_rank_draws_in_the_accent_and_says_what_it_shows():
    panel = _panel()
    process_section.update_process_section(panel, _payload(ranks=(0,)))
    assert panel["cpu_label"].text == "CPU usage"
    assert panel["cpu_sub"].text == "Rank 0 · share of host CPU"
    assert panel["rss_label"].text == "Process memory"
    assert panel["rss_sub"].text == "Rank 0 · physical RAM used"
    assert _line(panel, "cpu_chart", "Rank 0")["lineStyle"]["color"] == (
        theme.C_CPU
    )
    assert _line(panel, "rss_chart", "Rank 0")["lineStyle"]["color"] == (
        theme.C_MEM
    )


def test_cpu_and_memory_are_different_colours():
    assert theme.C_CPU != theme.C_MEM


def test_the_graphs_repeat_no_tile_value():
    """The upper-right value restated the tile, so it is gone."""
    panel = _panel()
    process_section.update_process_section(panel, _payload())
    for key in ("cpu_label", "cpu_sub"):
        assert "87.5" not in panel[key].text
    for key in ("rss_label", "rss_sub"):
        assert "2.5" not in panel[key].text


def test_the_axes_carry_units_and_one_relative_clock():
    panel = _panel()
    process_section.update_process_section(panel, _payload())
    cpu, rss = panel["cpu_chart"].options, panel["rss_chart"].options
    assert "%" in cpu["yAxis"]["axisLabel"][":formatter"]
    assert "GB" in rss["yAxis"]["axisLabel"][":formatter"]
    assert "'Now'" in cpu["xAxis"]["axisLabel"][":formatter"]
    assert (
        cpu["xAxis"]["axisLabel"][":formatter"]
        == rss["xAxis"]["axisLabel"][":formatter"]
    )
    # The header says "Last 4s"; the leftmost tick is that same moment.
    assert cpu["xAxis"]["min"] == -4.0


def test_a_fractional_window_rounds_up_so_no_sample_falls_off():
    """Header and leftmost tick share one whole number of seconds."""
    chart = RankChart(
        traces=(
            RankTrace(
                global_rank=0, timestamps=(10.0, 68.4), values=(1.0, 2.0)
            ),
        ),
    )
    payload = replace(
        _payload(ranks=(0,)), cpu_capacity_chart=chart, rss_chart=chart
    )
    panel = _panel()
    process_section.update_process_section(panel, payload)
    assert panel["duration"].text == "Last 59s"
    assert panel["cpu_chart"].options["xAxis"]["min"] == -59.0


def test_the_hover_reads_at_the_tiles_precision():
    """A CPU hover rounded to whole percent disagreed with a 22.9% tile."""
    panel = _panel()
    process_section.update_process_section(panel, _payload())
    cpu = panel["cpu_chart"].options["tooltip"][":valueFormatter"]
    rss = panel["rss_chart"].options["tooltip"][":valueFormatter"]
    assert "toFixed(1)" in cpu and "%" in cpu
    assert "GB" in rss and "Math.round" not in rss


def test_the_aggregate_total_line_is_not_drawn():
    """One line per rank; the sum across ranks is not a rank."""
    chart = RankChart(
        mode="recent",
        traces=tuple(
            RankTrace(
                global_rank=rank,
                timestamps=(1.0, 2.0, 3.0),
                values=(10.0, 11.0, 12.0),
            )
            for rank in (0, 1)
        ),
        total=RankTrace(
            global_rank=-1,
            timestamps=(1.0, 2.0, 3.0),
            values=(20.0, 22.0, 24.0),
        ),
    )
    payload = replace(_payload(), cpu_capacity_chart=chart, rss_chart=chart)
    panel = _panel()
    process_section.update_process_section(panel, payload)
    for key in ("cpu_chart", "rss_chart"):
        names = [s["name"] for s in panel[key].options["series"]]
        assert names == ["Rank 0", "Rank 1"], key


def test_an_unchanged_tick_does_not_resend_the_charts():
    """The UI timer outpaces telemetry; a repeat tick sends no chart.

    A new sample on any rank is what earns a redraw.
    """
    panel = _panel()
    payload = _payload()
    process_section.update_process_section(panel, payload)
    process_section.update_process_section(panel, payload)
    assert panel["cpu_chart"].updates == 1
    assert panel["rss_chart"].updates == 1

    chart = payload.cpu_capacity_chart
    newer = RankChart(
        mode=chart.mode,
        window_s=chart.window_s,
        traces=tuple(
            RankTrace(
                global_rank=trace.global_rank,
                timestamps=trace.timestamps + (trace.timestamps[-1] + 2.0,),
                values=trace.values + (trace.values[-1],),
            )
            for trace in chart.traces
        ),
    )
    process_section.update_process_section(
        panel,
        ProcessDashboardPayload(
            **{**payload.__dict__, "cpu_capacity_chart": newer}
        ),
    )
    assert panel["cpu_chart"].updates == 2
    assert panel["rss_chart"].updates == 2


def test_an_empty_chart_leaves_the_label_bare():
    panel = _panel()
    process_section.update_process_section(panel, ProcessDashboardPayload())
    assert panel["cpu_label"].text == "CPU usage"
    assert panel["cpu_sub"].text == ""
    assert panel["cpu_chart"].options["series"] == []


# --- the rank details ----------------------------------------------------
def test_one_rank_hides_the_rank_details():
    panel = _panel()
    process_section.update_process_section(panel, _payload(ranks=(0,)))
    assert panel["rows"].hidden


def test_no_rank_yet_hides_the_rank_details():
    panel = _panel()
    process_section.update_process_section(panel, ProcessDashboardPayload())
    assert panel["rows"].hidden


def test_many_ranks_name_the_rank_details_and_what_they_hold():
    panel = _panel()
    process_section.update_process_section(panel, _payload(ranks=(0, 1)))
    assert not panel["rows"].hidden
    assert panel["rows_title"].text == "Rank details (2)"
    assert panel["rows_hint"].text == (
        "CPU, process memory, node and sample age"
    )


def test_the_rows_use_the_cards_own_words():
    panel = _panel()
    process_section.update_process_section(panel, _payload(ranks=(0, 1)))
    html = panel["rows_html"].content
    assert "<th>cpu usage</th>" in html
    assert "<th>process memory</th>" in html
    assert "cpu cap" not in html
    assert "<th>rss</th>" not in html
    # Written the way the tile writes it.
    assert "25.0%" in html


def test_the_rows_use_only_colours_the_graphs_use():
    """Ranks are no longer told apart by colour, so neither are rows."""
    panel = _panel()
    process_section.update_process_section(panel, _payload(ranks=(0, 1, 2)))
    html = panel["rows_html"].content
    for colour in charting.RANK_COLORS:
        assert colour not in html, colour
    used = set(re.findall(r"#[0-9a-fA-F]{6}", html))
    assert used <= {theme.C_CPU, theme.C_MEM, theme.MUTED}
    # The ranks the two graphs highlight are marked in the same colours.
    assert theme.C_CPU in html and theme.C_MEM in html


def test_every_rank_is_a_row_with_its_identity_and_its_memory():
    panel = _panel()
    process_section.update_process_section(panel, _payload(ranks=(0, 1)))
    html = panel["rows_html"].content
    assert "R0" in html and "R1" in html
    assert "G0" in html and "N0" in html
    assert "cuda allocated" in html and "cuda reserved" in html


def test_each_row_pairs_cuda_values_from_its_least_headroom_sample():
    """The row and selected CUDA tiles share one per-rank sample basis."""
    rank = RankSnapshot(
        global_rank=0,
        gpu_reserved_p50_bytes=5.0 * GIB,
        cuda_least_headroom_sample=CudaHeadroomSample(
            allocated_bytes=22.0 * GIB,
            reserved_bytes=30.0 * GIB,
            total_bytes=40.0 * GIB,
        ),
    )

    html = process_section.rows_html((rank,), None)

    assert "22.0 / 40.0 GB" in html
    assert "30.0 / 40.0 GB" in html


def test_a_stale_rank_is_dimmed_and_kept():
    """Dropping it hides the one fact worth having when a job stalls."""
    panel = _panel()
    payload = _payload()
    payload = ProcessDashboardPayload(
        **{
            **payload.__dict__,
            "ranks": (
                _rank(0),
                _rank(1, freshness="stale", age_s=900.0),
            ),
        }
    )
    process_section.update_process_section(panel, payload)
    html = panel["rows_html"].content
    assert 'class="tml-stale"' in html
    assert "R1" in html
    assert "15 min" in html


def test_the_hint_states_coverage_without_classifying_it():
    panel = _panel()
    process_section.update_process_section(
        panel, _payload(ranks=(0, 1), stale=1, imbalance=22.0)
    )
    hint = panel["rows_hint"].text
    assert panel["rows_title"].text == "Rank details (2)"
    assert hint.startswith("CPU, process memory, node and sample age · ")
    assert "1 stale, excluded" in hint
    assert "reserved imbalance 22%" in hint
    for verdict in ("bad", "high", "warning", "critical", "unhealthy"):
        assert verdict not in hint.lower()


def test_the_hint_says_when_no_rank_is_reporting():
    """Every rank stale: nothing was excluded, so the hint must not say so.

    With no live rank the aggregates are the last numbers every rank sent,
    and "excluded" would tell the reader they were computed without them.
    """
    panel = _panel()
    process_section.update_process_section(
        panel, _payload(ranks=(0, 1), stale=2)
    )
    hint = panel["rows_hint"].text
    assert hint.endswith(" · none reporting")
    assert "excluded" not in hint


def test_a_rank_without_a_clock_is_named_in_the_hint():
    panel = _panel()
    process_section.update_process_section(
        panel, _payload(ranks=(0, 1), unknown=1)
    )
    assert "1 without a clock" in panel["rows_hint"].text


def test_a_small_spread_is_not_rounded_away_to_zero():
    """0.4% is a real reading; printing "0%" would say balanced."""
    panel = _panel()
    process_section.update_process_section(panel, _payload(imbalance=0.4))
    assert "reserved imbalance <1%" in panel["rows_hint"].text


# --- the auto-open trigger ----------------------------------------------
def test_the_rows_open_when_the_engine_says_so():
    panel = _panel()
    process_section.update_process_section(panel, _payload(rows_open=True))
    assert panel["rows"].value is True


def test_the_rows_do_not_reopen_after_the_reader_closes_them():
    """Rising edge only, or a reader is fought on every tick."""
    panel = _panel()
    process_section.update_process_section(panel, _payload(rows_open=True))
    panel["rows"].value = False
    process_section.update_process_section(panel, _payload(rows_open=True))
    assert panel["rows"].value is False


def test_the_rows_stay_shut_when_the_engine_is_silent():
    panel = _panel()
    process_section.update_process_section(panel, _payload(rows_open=False))
    assert panel["rows"].value is False


def test_the_card_never_decides_the_threshold_itself():
    """The view holds no number it compares an imbalance against.

    A severity call in the view is the one thing this layer may not do, so
    a large spread with the engine silent must leave the rows shut.
    """
    panel = _panel()
    process_section.update_process_section(
        panel, _payload(imbalance=99.0, rows_open=False)
    )
    assert panel["rows"].value is False


# --- the whole path ------------------------------------------------------
def test_the_card_renders_the_same_from_a_real_database(tmp_path):
    """Database to screen, through the real compute layer.

    The unit tests above hand the card a constructed payload; this one
    proves the layer that builds it agrees, so a boundary change cannot
    pass both halves while breaking the join between them.

    The numbers assert the two deliberate meaning changes. The row holds
    6 GiB allocated inside 7 GiB reserved, and those are now two tiles
    reading 6.0 and 7.0 rather than one tile reading 6.0. CPU is 200% of a
    process on an 8-core host, which is 25% of the host's capacity, and the
    tile shows the capacity share rather than the raw number.
    """
    import sqlite3

    from traceml_ai.aggregator.sqlite_writers.process import init_schema

    path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(path)
    init_schema(conn)
    for seq in (1, 2):
        conn.execute(
            "INSERT INTO process_samples (recv_ts_ns, rank, global_rank, "
            "seq, sample_ts_s, cpu_percent, cpu_logical_core_count, "
            "ram_used_bytes, ram_total_bytes, gpu_available, "
            "gpu_mem_used_bytes, gpu_mem_reserved_bytes, "
            "gpu_mem_total_bytes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                int((1_700_000_000 + seq) * 1e9),
                0,
                0,
                seq,
                1_700_000_000.0 + seq,
                200.0,
                8,
                4.0 * GIB,
                16.0 * GIB,
                1,
                6.0 * GIB,
                7.0 * GIB,
                16.0 * GIB,
            ),
        )
    conn.commit()
    conn.close()

    payload = ProcessDashboardComputer(db_path=str(path)).compute()
    panel = _panel()
    process_section.update_process_section(panel, payload)

    assert "25.0" in panel["tiles"]["cpu"].content
    assert "4.0" in panel["tiles"]["rss"].content
    assert "7.0" in panel["tiles"]["reserved"].content
    assert "6.0" in panel["tiles"]["alloc"].content
    assert "R0" in panel["rows_html"].content
    assert panel["title"].text == "Trainer process"
    assert panel["context"].text == "Rank 0 · Trainer process only"
    assert panel["subs"]["rss"].text == "of 16.0 GB host RAM"


def test_every_class_the_card_emits_has_a_rule_behind_it():
    """The stale marker shipped once with no CSS rule to render it.

    The row carried `class="tml-stale"`, the stylesheet defined nothing for
    it, and a dead rank drew identically to a live one. Nothing in the
    markup or the payload could catch that, so the check is here: any class
    this module puts on screen must exist in the stylesheet.
    """
    from traceml_ai.aggregator.display_drivers.nicegui_sections import theme

    css = theme.head_html()
    ranks = (_rank(0), _rank(1, freshness="stale", age_s=900.0))
    html = process_section.rows_html(ranks, _chart(0, 1))

    emitted = set(re.findall(r'class="([^"]+)"', html))
    for group in emitted:
        for name in group.split():
            assert f".{name}" in css, f"{name} has no CSS rule"


def test_a_teardown_step_does_not_turn_a_gpu_run_cpu_only():
    """The last samples of every run land after torch releases the device.

    Asking the newest committed step whether there is a GPU therefore
    answers "no" on every finished run, which blanked both CUDA tiles and
    printed "no GPU" on a card that was simultaneously showing a reserved
    spread derived from CUDA bytes.
    """
    from traceml_ai.renderers.process.dashboard_models import (
        ProcessHistoryEntry,
    )

    payload = ProcessDashboardPayload(
        history=(
            ProcessHistoryEntry(
                seq=1,
                ts=1_700_000_001.0,
                cpu_percent_max=10.0,
                ram_used_bytes_max=1.0 * GIB,
                ram_total_bytes=64.0 * GIB,
                gpu=None,
            ),
        ),
        ranks=(_rank(0), _rank(1)),
        gpu_reserved=MetricRollup(now=7.0 * GIB, p95=7.0 * GIB, worst_rank=1),
        gpu=MetricRollup(now=6.0 * GIB, p95=6.0 * GIB),
        coverage=RankCoverage(total=2, live=2),
        cpu_capacity_chart=_chart(0, 1),
        rss_chart=_chart(0, 1),
    )
    assert payload.gpu_available is True

    panel = _panel()
    process_section.update_process_section(panel, payload)
    assert panel["tiles"]["reserved"].content != "n/a"
    assert panel["subs"]["reserved"].text != "no GPU"


def test_the_axis_fits_the_line_that_is_drawn_not_the_peaks():
    """Peaks are rolling maxima and nothing plots them.

    Fitting the axis to peaks put its floor above the drawn line, so the
    early samples were clipped and the drift the chart exists to show was
    understated.
    """
    chart = RankChart(
        mode="retained",
        window_s=120.0,
        traces=(
            RankTrace(
                global_rank=0,
                timestamps=(1.0, 2.0, 3.0),
                values=(8.0 * GIB, 10.0 * GIB, 10.5 * GIB),
                peaks=(10.6 * GIB, 10.7 * GIB, 10.8 * GIB),
            ),
        ),
    )
    payload = ProcessDashboardPayload(
        window_len=3,
        ranks=(_rank(0),),
        coverage=RankCoverage(total=1, live=1),
        rss_worst=MetricRollup(now=10.5 * GIB, p95=10.5 * GIB, worst_rank=0),
        cpu_capacity=MetricRollup(now=25.0, p95=25.0, worst_rank=0),
        rss_chart=chart,
        cpu_capacity_chart=chart,
    )
    panel = _panel()
    process_section.update_process_section(panel, payload)

    axis = panel["rss_chart"].options["yAxis"]
    drawn = [v for _t, v in panel["rss_chart"].options["series"][0]["data"]]
    assert axis["min"] <= min(drawn), "the floor clips the drawn line"
    assert axis["max"] >= max(drawn)


def test_a_single_sample_is_visible_rather_than_an_empty_plot():
    """A line through one point draws nothing, which is every run's start."""
    one = RankChart(
        mode="recent",
        traces=(RankTrace(global_rank=0, timestamps=(1.0,), values=(1.7,)),),
    )
    payload = ProcessDashboardPayload(
        window_len=1,
        ranks=(_rank(0),),
        coverage=RankCoverage(total=1, live=1),
        cpu_capacity=MetricRollup(now=1.7, p95=1.7, worst_rank=0),
        cpu_capacity_chart=one,
        rss_chart=one,
    )
    panel = _panel()
    process_section.update_process_section(panel, payload)
    series = panel["cpu_chart"].options["series"][0]
    assert series["data"], "the point must reach the chart"
    assert series["showSymbol"] is True, "one point needs a marker"


def test_the_allocated_tile_survives_a_teardown_step():
    """It reads the ranks, so it does not blank when history loses the GPU.

    The tile said "median rank · live tensors" while being fed the newest
    committed step's aggregate, which is None after teardown. It read
    "n/a" directly above rows listing each rank's allocated bytes.
    """
    from traceml_ai.renderers.process.dashboard_models import (
        ProcessHistoryEntry,
    )

    payload = ProcessDashboardPayload(
        history=(
            ProcessHistoryEntry(
                seq=1,
                ts=1_700_000_001.0,
                cpu_percent_max=10.0,
                ram_used_bytes_max=1.0 * GIB,
                ram_total_bytes=64.0 * GIB,
                gpu=None,
            ),
        ),
        ranks=(_rank(0), _rank(1)),
        coverage=RankCoverage(total=2, live=2),
        gpu=None,
        gpu_allocated=MetricRollup(now=4.0 * GIB),
        gpu_reserved=MetricRollup(now=6.0 * GIB, worst_rank=0),
        cpu_capacity_chart=_chart(0, 1),
        rss_chart=_chart(0, 1),
    )
    panel = _panel()
    process_section.update_process_section(panel, payload)
    assert "4.0" in panel["tiles"]["alloc"].content


def test_the_axis_fits_the_rank_lines_not_the_undrawn_total():
    """The total is no longer drawn, so it must not stretch the axis.

    Fitting to a line nobody sees would squash every drawn line into the
    lower part of the chart for a ceiling with nothing near it.
    """
    chart = RankChart(
        mode="recent",
        traces=(
            RankTrace(
                global_rank=0,
                timestamps=(1.0, 2.0, 3.0),
                values=(10.0, 11.0, 12.0),
            ),
            RankTrace(
                global_rank=1,
                timestamps=(1.0, 2.0, 3.0),
                values=(10.0, 11.0, 12.0),
            ),
        ),
        total=RankTrace(
            global_rank=-1,
            timestamps=(1.0, 2.0, 3.0),
            values=(20.0, 22.0, 24.0),
        ),
    )
    payload = ProcessDashboardPayload(
        window_len=3,
        ranks=(_rank(0), _rank(1)),
        coverage=RankCoverage(total=2, live=2),
        cpu_capacity=MetricRollup(now=12.0, worst_rank=0),
        cpu_capacity_chart=chart,
        rss_chart=chart,
    )
    panel = _panel()
    process_section.update_process_section(panel, payload)

    series = panel["cpu_chart"].options["series"]
    assert len(series) == 2, "two ranks and no total"

    drawn = [v for s in series for _t, v in s["data"]]
    top = panel["cpu_chart"].options["yAxis"]["max"]
    assert top >= max(drawn), "a drawn line must not be clipped"
    assert top < max(chart.total.values), "the total must not fit it"


def test_the_rss_row_shows_the_same_basis_as_the_rss_tile():
    """A reader who checks the tile against the rows must find it there.

    The tile names a rank at its median. The row used to show that rank's
    newest sample, so on a finished run the tile said 2.1 GB and the row
    it pointed at said 1.0 GB.
    """
    ranks = (
        RankSnapshot(
            global_rank=0,
            ram_used_bytes=1.0 * GIB,
            ram_used_p50_bytes=2.1 * GIB,
            ram_total_bytes=187.0 * GIB,
            freshness="fresh",
        ),
    )
    html = process_section.rows_html(ranks, None)
    assert "2.1" in html
    assert "1.0 /" not in html


# --- the real elements ---------------------------------------------------
@pytest.mark.parametrize(
    "make",
    [
        ProcessDashboardPayload,
        lambda: _payload(ranks=(0,)),
        lambda: _payload(ranks=(0,), gpu=False),
        lambda: _payload(ranks=(0, 1, 2)),
        lambda: _payload(ranks=(0, 1), gpu=False),
        lambda: _retained(_payload()),
    ],
)
def test_the_built_card_accepts_every_update(make):
    """The fakes above stand in for NiceGUI; this runs the real elements.

    A fake that accepts a call the element does not have would let every
    test here pass while the live card raised on its first tick.
    """
    panel = process_section.build_process_section()
    payload = make()
    process_section.update_process_section(panel, payload)
    process_section.update_process_section(panel, payload)
    # The repeated value beside each graph is gone, not just emptied.
    assert "cpu_value" not in panel and "rss_value" not in panel


def test_both_graphs_give_their_tick_labels_the_same_room():
    """A "30%" axis and a "1.111 GB" axis otherwise start at different x.

    The two plots then disagree about where any moment is, which is the
    one thing two graphs sharing a time range must agree on.
    """
    panel = _panel()
    process_section.update_process_section(panel, _payload())
    widths = {
        key: re.findall(
            r"padStart\((\d+)\)",
            panel[key].options["yAxis"]["axisLabel"][":formatter"],
        )
        for key in ("cpu_chart", "rss_chart")
    }
    assert widths["cpu_chart"] == widths["rss_chart"]
    assert len(widths["cpu_chart"]) == 1
    yaxis = panel["rss_chart"].options["yAxis"]
    widest = charting.value_axis_label(
        yaxis["max"], yaxis["max"] - yaxis["min"], " GB"
    )
    assert int(widths["cpu_chart"][0]) == len(widest)


def test_a_redraw_does_not_pad_an_already_padded_formatter():
    panel = _panel()
    payload = _payload()
    process_section.update_process_section(panel, payload)
    newer = replace(payload, window_len=payload.window_len + 1)
    process_section.update_process_section(panel, newer)
    for key in ("cpu_chart", "rss_chart"):
        formatter = panel[key].options["yAxis"]["axisLabel"][":formatter"]
        assert formatter.count("padStart") == 1, key
