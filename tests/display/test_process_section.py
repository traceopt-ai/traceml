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
* The ``CPU`` and ``RAM`` tiles become ``cpu capacity`` and ``rss``, which
  are the per-rank, denominator-carrying versions of the same quantities.
  A raw 700% CPU reading is not comparable between hosts; 87.5% of the
  host's capacity is.

``test_the_card_renders_the_same_from_a_real_database`` closes the loop
from database through compute to screen, and states both new meanings as
explicit numbers so the change is visible rather than implied.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui_sections import (  # noqa: E402
    process_section,
)
from traceml_ai.renderers.process.dashboard_compute import (  # noqa: E402
    ProcessDashboardComputer,
)
from traceml_ai.renderers.process.dashboard_models import (  # noqa: E402
    MetricRollup,
    ProcessDashboardPayload,
    RankChart,
    RankCoverage,
    RankSnapshot,
    RankTrace,
)

GB = 1_000_000_000.0
GIB = float(1024**3)


class _Html:
    def __init__(self) -> None:
        self.content = ""


class _Text:
    def __init__(self) -> None:
        self.text = ""
        self.tooltips: list = []

    def tooltip(self, text: str) -> None:
        self.tooltips.append(text)


class _Expansion:
    def __init__(self) -> None:
        self.value = False


class _Chart:
    def __init__(self) -> None:
        self.options = {
            "series": [],
            "xAxis": {"axisLabel": {}},
            "yAxis": {"axisLabel": {}},
            "tooltip": {"axisPointer": {"label": {}}},
        }
        self.updates = 0

    def update(self) -> None:
        self.updates += 1


def _panel() -> dict:
    keys = ("cpu", "rss", "reserved", "alloc")
    return {
        "tiles": {k: _Html() for k in keys},
        "subs": {k: _Text() for k in keys},
        "note": _Text(),
        "cpu_chart": _Chart(),
        "rss_chart": _Chart(),
        "cpu_label": _Text(),
        "rss_label": _Text(),
        "cpu_value": _Text(),
        "rss_value": _Text(),
        "rows": _Expansion(),
        "rows_hint": _Text(),
        "rows_html": _Html(),
        "_was_open": False,
        "_signature": None,
    }


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
        gpu_allocated_p50_bytes=allocated,
        gpu_reserved_bytes=reserved,
        gpu_reserved_p50_bytes=reserved,
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
        cpu_capacity=MetricRollup(now=87.5, p95=87.5, p50=25.0, worst_rank=1),
        rss_worst=MetricRollup(now=2.5 * GIB, p95=2.5 * GIB, worst_rank=0),
        gpu_reserved=(
            MetricRollup(now=7.0 * GIB, p95=7.0 * GIB, worst_rank=1)
            if gpu
            else None
        ),
        gpu=(MetricRollup(now=6.0 * GIB, p95=6.0 * GIB) if gpu else None),
        gpu_allocated=(MetricRollup(now=6.0 * GIB) if gpu else None),
        reserved_imbalance_percent=imbalance,
        rows_open=rows_open,
        cpu_capacity_chart=_chart(*ranks),
        rss_chart=_chart(*ranks),
    )


# --- the four tiles ------------------------------------------------------
def test_the_cpu_tile_leads_with_the_worst_rank_and_names_it():
    panel = _panel()
    process_section.update_process_section(panel, _payload())
    assert "87.5" in panel["tiles"]["cpu"].content
    assert panel["subs"]["cpu"].text == ("worst rank · R1 · of host capacity")


def test_the_rss_tile_names_its_worst_rank():
    panel = _panel()
    process_section.update_process_section(panel, _payload())
    assert "2.5" in panel["tiles"]["rss"].content
    assert panel["subs"]["rss"].text == "worst rank · R0"


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
    assert panel["subs"]["reserved"].text == "least-headroom rank · R1"
    assert panel["subs"]["alloc"].text == "median rank · live tensors"


def test_a_cpu_only_run_marks_both_gpu_tiles_absent():
    panel = _panel()
    process_section.update_process_section(panel, _payload(gpu=False))
    assert panel["tiles"]["reserved"].content == "n/a"
    assert panel["tiles"]["alloc"].content == "n/a"
    assert panel["subs"]["reserved"].text == "no GPU"


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


def test_a_retained_chart_says_it_is_rolling():
    panel = _panel()
    payload = _payload()
    payload = ProcessDashboardPayload(
        **{
            **payload.__dict__,
            "cpu_capacity_chart": _chart(0, 1, mode="retained"),
        }
    )
    process_section.update_process_section(panel, payload)
    assert "rolling 2 min" in panel["cpu_label"].text


def test_an_empty_chart_leaves_the_label_bare():
    panel = _panel()
    process_section.update_process_section(panel, ProcessDashboardPayload())
    assert panel["cpu_label"].text == "process cpu · capacity per rank"
    assert panel["cpu_chart"].options["series"] == []


# --- the per-rank rows ---------------------------------------------------
def test_every_rank_is_a_row_with_its_identity_and_its_memory():
    panel = _panel()
    process_section.update_process_section(panel, _payload(ranks=(0, 1)))
    html = panel["rows_html"].content
    assert "R0" in html and "R1" in html
    assert "G0" in html and "N0" in html
    assert "cuda allocated" in html and "cuda reserved" in html


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
    assert "2 ranks" in hint
    assert "1 stale, excluded" in hint
    assert "reserved imbalance 22%" in hint
    for verdict in ("bad", "high", "warning", "critical", "unhealthy"):
        assert verdict not in hint.lower()


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
