# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""What the Process card puts on screen for a given payload.

Characterization tests, written against version_0.3.7 rendering. The
percentile and rollup arithmetic they pin currently lives in the section;
it moves to the compute layer without any of these expectations changing,
which is the point of pinning it here first.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui_sections import (  # noqa: E402
    process_section,
)

GB = 1_000_000_000.0


class _Html:
    def __init__(self) -> None:
        self.content = ""


class _Text:
    def __init__(self) -> None:
        self.text = ""


class _Chart:
    def __init__(self) -> None:
        self.options = {
            "series": [{"data": []}, {"data": []}],
            "yAxis": [{}, {}],
        }
        self.updates = 0

    def update(self) -> None:
        self.updates += 1


def _panel() -> dict:
    return {
        "chart": _Chart(),
        "win": _Text(),
        "kpis": {k: _Html() for k in ("cpu", "ram", "gmem", "gimb")},
    }


def _entry(seq: int, **kw) -> dict:
    entry = {
        "seq": seq,
        "ts": 1_700_000_000.0 + seq,
        "cpu_max": kw.pop("cpu_max", 10.0),
        "ram_used_max": kw.pop("ram_used_max", 2.0 * GB),
        "ram_total": kw.pop("ram_total", 16.0 * GB),
    }
    entry.update(kw)
    return entry


def _payload(history, imbalance=None):
    from datetime import datetime, timezone

    return {
        "history": history,
        "gpu_used_imbalance": imbalance,
        "series": {
            "x_time": [
                datetime.fromtimestamp(e["ts"], tz=timezone.utc).isoformat()
                for e in history
            ],
            "cpu_max": [e["cpu_max"] for e in history],
            "ram_used_max": [e["ram_used_max"] for e in history],
            "gpu_used": [
                e["gpu_used"] for e in history if e.get("gpu_used") is not None
            ],
        },
    }


# --- the arithmetic the card reports ------------------------------------
@pytest.mark.parametrize(
    "values, p, expected",
    [
        ([], 50, 0.0),
        ([5.0], 95, 5.0),
        ([1.0, 2.0, 3.0, 4.0], 50, 2.5),
        ([1.0, 2.0, 3.0, 4.0], 0, 1.0),
        ([1.0, 2.0, 3.0, 4.0], 100, 4.0),
        ([3.0, 1.0, 2.0], 50, 2.0),
    ],
)
def test_percentile(values, p, expected):
    assert process_section._percentile(values, p) == pytest.approx(expected)


def test_percentile_ignores_missing_values():
    assert process_section._percentile([1.0, None, 3.0], 50) == pytest.approx(
        2.0
    )


def test_rollups_report_the_newest_cpu_and_its_window_percentiles():
    window = [_entry(i, cpu_max=float(i * 10)) for i in range(1, 5)]
    roll = process_section._compute_rollups(window)
    assert roll["cpu"]["now"] == pytest.approx(40.0)
    assert roll["cpu"]["p50"] == pytest.approx(25.0)
    assert roll["cpu"]["p95"] == pytest.approx(38.5)


def test_rollups_carry_the_ram_denominator():
    window = [_entry(1, ram_used_max=3.0 * GB, ram_total=64.0 * GB)]
    roll = process_section._compute_rollups(window)
    assert roll["ram"]["now"] == pytest.approx(3.0 * GB)
    assert roll["ram"]["total"] == pytest.approx(64.0 * GB)


def test_gpu_availability_follows_the_newest_entry():
    with_gpu = [_entry(1, gpu_used=2.0 * GB)]
    assert process_section._compute_rollups(with_gpu)["gpu_available"] is True
    assert (
        process_section._compute_rollups([_entry(1)])["gpu_available"] is False
    )


def test_gpu_rollup_is_zero_when_no_entry_carries_a_gpu():
    roll = process_section._compute_rollups([_entry(1)])
    assert roll["gpu"]["now"] == pytest.approx(0.0)
    assert roll["gpu"]["p95"] == pytest.approx(0.0)


# --- what lands on screen ------------------------------------------------
def test_a_cpu_only_payload_renders_cpu_and_ram_and_marks_gpu_absent():
    panel = _panel()
    history = [_entry(i, cpu_max=50.0, ram_used_max=4.0 * GB) for i in (1, 2)]
    process_section.update_process_section(panel, _payload(history))
    assert "50" in panel["kpis"]["cpu"].content
    assert "4.00" in panel["kpis"]["ram"].content
    assert panel["kpis"]["gmem"].content == "N/A"
    assert panel["kpis"]["gimb"].content == "—"
    assert panel["win"].text == "last 2 samples"


def test_a_gpu_payload_renders_memory_and_imbalance():
    panel = _panel()
    history = [
        _entry(i, gpu_used=6.0 * GB, gpu_total=16.0 * GB) for i in (1, 2)
    ]
    process_section.update_process_section(
        panel, _payload(history, imbalance=1.5 * GB)
    )
    assert "6.00" in panel["kpis"]["gmem"].content
    assert "1.50" in panel["kpis"]["gimb"].content


def test_an_unavailable_imbalance_renders_a_dash_not_a_zero():
    panel = _panel()
    history = [_entry(1, gpu_used=6.0 * GB, gpu_total=16.0 * GB)]
    process_section.update_process_section(
        panel, _payload(history, imbalance=None)
    )
    assert "—" in panel["kpis"]["gimb"].content


def test_the_chart_plots_ram_as_a_percentage_of_its_total():
    panel = _panel()
    history = [_entry(1, ram_used_max=4.0 * GB, ram_total=16.0 * GB)]
    process_section.update_process_section(panel, _payload(history))
    ram_series = panel["chart"].options["series"][0]["data"]
    assert len(ram_series) == 1
    assert ram_series[0][1] == pytest.approx(25.0)


def test_the_chart_plots_gpu_memory_as_a_percentage_of_its_total():
    panel = _panel()
    history = [_entry(1, gpu_used=4.0 * GB, gpu_total=16.0 * GB)]
    process_section.update_process_section(panel, _payload(history))
    gpu_series = panel["chart"].options["series"][1]["data"]
    assert gpu_series[0][1] == pytest.approx(25.0)


def test_both_axes_share_one_ceiling():
    panel = _panel()
    history = [_entry(1, ram_used_max=8.0 * GB, ram_total=16.0 * GB)]
    process_section.update_process_section(panel, _payload(history))
    options = panel["chart"].options
    assert options["yAxis"][0]["max"] == options["yAxis"][1]["max"]


def test_the_window_is_bounded_to_the_last_hundred_samples():
    panel = _panel()
    history = [_entry(i) for i in range(1, 151)]
    process_section.update_process_section(panel, _payload(history))
    assert panel["win"].text == "last 100 samples"


def test_an_empty_history_leaves_the_card_untouched():
    panel = _panel()
    process_section.update_process_section(panel, _payload([]))
    assert panel["win"].text == ""
    assert panel["chart"].updates == 0


def test_a_non_dict_payload_is_ignored():
    panel = _panel()
    process_section.update_process_section(panel, None)
    assert panel["win"].text == ""
