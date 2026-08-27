# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""What the Process card puts on screen for a given payload.

Every on-screen assertion here is the one that was pinned against
version_0.3.7 rendering; only the payload handed to the card changed shape.
The percentile and rollup tests that used to live in this file moved to
``tests/renderers/process/test_process_dashboard_payload.py``, because the
arithmetic they cover moved to the compute layer. That relocation is the
whole point of the change, so the tests follow the code.

``test_the_card_renders_the_same_from_a_real_database`` closes the loop end
to end: database, compute, card, screen.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui_sections import (  # noqa: E402
    process_section,
)
from traceml_ai.renderers.process.dashboard_compute import (  # noqa: E402
    ProcessDashboardComputer,
)
from traceml_ai.renderers.process.dashboard_models import (  # noqa: E402
    ChartSeries,
    ChartTrace,
    GpuSnapshot,
    MetricRollup,
    ProcessDashboardPayload,
    ProcessHistoryEntry,
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


def _payload(
    *,
    n: int = 1,
    cpu: float = 10.0,
    ram: float = 2.0 * GB,
    ram_total: float = 16.0 * GB,
    gpu_used: float = None,
    gpu_total: float = 16.0 * GB,
    imbalance: float = None,
) -> ProcessDashboardPayload:
    stamps = tuple(1_700_000_000.0 + i for i in range(n))
    gpu_block = (
        None
        if gpu_used is None
        else GpuSnapshot(
            used_bytes=gpu_used,
            total_bytes=gpu_total,
            headroom_bytes=gpu_total - gpu_used,
            rank=0,
            used_imbalance_bytes=imbalance or 0.0,
        )
    )
    history = tuple(
        ProcessHistoryEntry(
            seq=i,
            ts=stamps[i],
            cpu_percent_max=cpu,
            ram_used_bytes_max=ram,
            ram_total_bytes=ram_total,
            gpu=gpu_block,
        )
        for i in range(n)
    )
    return ProcessDashboardPayload(
        history=history,
        window_len=n,
        cpu=MetricRollup(now=cpu, p95=cpu, p50=cpu),
        ram=MetricRollup(now=ram, p95=ram, total=ram_total),
        gpu=(
            None
            if gpu_used is None
            else MetricRollup(now=gpu_used, p95=gpu_used, total=gpu_total)
        ),
        gpu_used_imbalance_bytes=imbalance,
        chart=ChartSeries(
            ram_percent=ChartTrace(
                label="RAM",
                timestamps=stamps,
                values=tuple((ram / ram_total) * 100.0 for _ in range(n)),
            ),
            gpu_percent=(
                None
                if gpu_used is None
                else ChartTrace(
                    label="GPU mem",
                    timestamps=stamps,
                    values=tuple(
                        (gpu_used / gpu_total) * 100.0 for _ in range(n)
                    ),
                )
            ),
        ),
    )


# --- what lands on screen ------------------------------------------------
def test_a_cpu_only_payload_renders_cpu_and_ram_and_marks_gpu_absent():
    panel = _panel()
    process_section.update_process_section(
        panel, _payload(n=2, cpu=50.0, ram=4.0 * GB)
    )
    assert "50" in panel["kpis"]["cpu"].content
    assert "4.00" in panel["kpis"]["ram"].content
    assert panel["kpis"]["gmem"].content == "N/A"
    assert panel["kpis"]["gimb"].content == "—"
    assert panel["win"].text == "last 2 samples"


def test_a_gpu_payload_renders_memory_and_imbalance():
    panel = _panel()
    process_section.update_process_section(
        panel, _payload(n=2, gpu_used=6.0 * GB, imbalance=1.5 * GB)
    )
    assert "6.00" in panel["kpis"]["gmem"].content
    assert "1.50" in panel["kpis"]["gimb"].content


def test_an_unavailable_imbalance_renders_a_dash_not_a_zero():
    panel = _panel()
    process_section.update_process_section(
        panel, _payload(gpu_used=6.0 * GB, imbalance=None)
    )
    assert "—" in panel["kpis"]["gimb"].content


def test_the_chart_plots_ram_as_a_percentage_of_its_total():
    panel = _panel()
    process_section.update_process_section(
        panel, _payload(ram=4.0 * GB, ram_total=16.0 * GB)
    )
    ram_series = panel["chart"].options["series"][0]["data"]
    assert len(ram_series) == 1
    assert ram_series[0][1] == pytest.approx(25.0)


def test_the_chart_plots_gpu_memory_as_a_percentage_of_its_total():
    panel = _panel()
    process_section.update_process_section(
        panel, _payload(gpu_used=4.0 * GB, gpu_total=16.0 * GB)
    )
    gpu_series = panel["chart"].options["series"][1]["data"]
    assert gpu_series[0][1] == pytest.approx(25.0)


def test_both_axes_share_one_ceiling():
    panel = _panel()
    process_section.update_process_section(
        panel, _payload(ram=8.0 * GB, ram_total=16.0 * GB)
    )
    options = panel["chart"].options
    assert options["yAxis"][0]["max"] == options["yAxis"][1]["max"]


def test_the_window_length_is_stated_by_the_payload():
    panel = _panel()
    process_section.update_process_section(panel, _payload(n=100))
    assert panel["win"].text == "last 100 samples"


def test_an_empty_payload_leaves_the_card_untouched():
    panel = _panel()
    process_section.update_process_section(panel, ProcessDashboardPayload())
    assert panel["win"].text == ""
    assert panel["chart"].updates == 0


def test_a_payload_of_the_wrong_type_is_ignored():
    panel = _panel()
    process_section.update_process_section(panel, None)
    process_section.update_process_section(panel, {"history": []})
    assert panel["win"].text == ""


def test_a_sample_without_a_usable_time_is_not_plotted():
    """A zero timestamp cannot be placed on a time axis, so it is dropped
    rather than drawn at the epoch."""
    payload = _payload(n=1)
    payload = ProcessDashboardPayload(
        history=payload.history,
        window_len=payload.window_len,
        cpu=payload.cpu,
        ram=payload.ram,
        chart=ChartSeries(
            ram_percent=ChartTrace(
                label="RAM", timestamps=(0.0,), values=(25.0,)
            )
        ),
    )
    panel = _panel()
    process_section.update_process_section(panel, payload)
    assert panel["chart"].options["series"][0]["data"] == []


# --- the whole path ------------------------------------------------------
def test_the_card_renders_the_same_from_a_real_database(tmp_path):
    """Database to screen, through the real compute layer.

    The unit tests above hand the card a constructed payload; this one
    proves the layer that builds it agrees, so a boundary change cannot
    pass both halves while breaking the join between them.
    """
    import sqlite3

    from traceml_ai.aggregator.sqlite_writers.process import init_schema

    path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(path)
    init_schema(conn)
    for seq in (1, 2):
        conn.execute(
            "INSERT INTO process_samples (recv_ts_ns, rank, seq, "
            "sample_ts_s, cpu_percent, ram_used_bytes, ram_total_bytes, "
            "gpu_available, gpu_mem_used_bytes, gpu_mem_reserved_bytes, "
            "gpu_mem_total_bytes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                0,
                0,
                seq,
                1_700_000_000.0 + seq,
                50.0,
                4.0 * GB,
                16.0 * GB,
                1,
                6.0 * GB,
                7.0 * GB,
                16.0 * GB,
            ),
        )
    conn.commit()
    conn.close()

    payload = ProcessDashboardComputer(db_path=str(path)).compute()
    panel = _panel()
    process_section.update_process_section(panel, payload)

    assert panel["win"].text == "last 2 samples"
    assert "50" in panel["kpis"]["cpu"].content
    assert "4.00" in panel["kpis"]["ram"].content
    assert "6.00" in panel["kpis"]["gmem"].content
    ram_series = panel["chart"].options["series"][0]["data"]
    assert ram_series[0][1] == pytest.approx(25.0)
    gpu_series = panel["chart"].options["series"][1]["data"]
    assert gpu_series[0][1] == pytest.approx(37.5)
