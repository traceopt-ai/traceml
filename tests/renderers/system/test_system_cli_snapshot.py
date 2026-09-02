# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The terminal card's GPU aggregate, over devices that did not report.

The dashboard and the terminal card read the same rows and answered this
differently: the dashboard learned to skip a device the sampler could not
read, the terminal card kept averaging its zeros in. Both surfaces are in
this package, so a fix that lands in one of them is half a fix.
"""

from __future__ import annotations

import sqlite3

from rich.console import Console

from tests.renderers.system.conftest import GB, gpu
from traceml_ai.renderers.system.cli_compute import SystemCLIComputer
from traceml_ai.renderers.system.renderer import SystemRenderer

NVML_FAILED = {
    "gpu_idx": 1,
    "util": 0.0,
    "mem_used_bytes": 0.0,
    "mem_total_bytes": 0.0,
    "temperature_c": 0.0,
    "power_usage_w": 0.0,
    "power_limit_w": 0.0,
}


def _render_text(path: str) -> str:
    console = Console(record=True, width=100, color_system=None)
    console.print(SystemRenderer(path).get_panel_renderable())
    return console.export_text()


def test_four_busy_gpus_average_to_full(system_db):
    path = system_db(ticks=20, gpus=lambda seq: [gpu(i) for i in range(4)])
    out = SystemCLIComputer(path).compute()
    assert out["gpu_util_avg"] == 100.0
    assert out["gpu_util_skew"] == 0.0


def test_an_nvml_failed_gpu_is_not_averaged_in(system_db):
    """The zeros are a failed read, not an idle device.

    Before this, a healthy four-GPU host read 75% with a 100-point skew.
    """
    path = system_db(
        ticks=20,
        gpus=lambda seq: [gpu(0), dict(NVML_FAILED), gpu(2), gpu(3)],
    )
    out = SystemCLIComputer(path).compute()
    assert out["gpu_util_devices"] == 3
    assert out["gpu_util_avg"] == 100.0
    assert out["gpu_util_skew"] == 0.0
    # Power is a sum over the same rows and must drop the failed device
    # rather than adding its 0 W.
    assert out["gpu_power_usage"] == 66.0 * 3


def test_a_null_util_column_is_not_a_zero_reading(system_db):
    """The other trigger: the device reports, the metric does not.

    `reported` is about the DEVICE. The util mean counts READINGS, so a
    live device with no util value is absent from it.
    """
    path = system_db(
        ticks=20,
        gpus=lambda seq: [gpu(0), gpu(1, util=None), gpu(2), gpu(3)],
    )
    out = SystemCLIComputer(path).compute()
    assert out["gpu_util_devices"] == 3
    assert out["gpu_util_avg"] == 100.0
    assert out["gpu_util_skew"] == 0.0


def test_no_util_readings_render_as_unavailable(system_db):
    """An empty set of readings is unavailable, not an idle measurement."""
    path = system_db(
        ticks=20,
        gpus=lambda seq: [dict(NVML_FAILED, gpu_idx=i) for i in range(4)],
    )

    out = SystemCLIComputer(path).compute()
    assert out["gpu_util_devices"] == 0
    assert out["gpu_util_total"] is None
    assert out["gpu_util_avg"] is None

    text = _render_text(path)
    assert "GPU UTIL N/A" in text
    assert "GPU UTIL 0.0%" not in text


def _null_host_readings(path: str) -> None:
    """Drop the CPU and memory readings on every row.

    Written directly rather than through the fixture: passing None for
    ``cpu_percent`` means "unspecified" there and substitutes a default,
    so the fixture cannot express a NULL column. The dashboard tests take
    the same route for the same reason.
    """
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "UPDATE system_samples SET cpu_percent = NULL, "
            "ram_used_bytes = NULL"
        )
        conn.commit()
    finally:
        conn.close()


def test_an_unread_host_is_not_an_idle_host(system_db):
    """No CPU or RAM reading on the newest row is not 0%.

    The terminal card reads one row rather than a window, so the trigger
    is narrower than the dashboard's, but the confusion is identical: the
    card read "CPU 0.0%" and "RAM 0.0%" for a host whose readings simply
    never arrived.
    """
    path = system_db(ticks=5)
    _null_host_readings(path)

    out = SystemCLIComputer(path).compute()

    assert out["cpu"] is None
    assert out["ram_used"] is None
    # The capacity is a separate reading and this row still carries it.
    assert out["ram_total"] == 16.0 * GB
    assert "N/A" in _render_text(path)


def test_live_devices_with_unread_metrics_abstain(system_db):
    """Reported devices whose metric columns are all NULL.

    The device guard does not fire: these cards report a memory total and
    a power limit, so they are live. Only the utilisation mean was
    covered, because only utilisation counts its readings. Memory summed
    to 0 bytes, temperature reported 0.0 degrees, and headroom reported
    the entire card free, which is the one that points the wrong way
    about pressure.
    """
    path = system_db(
        ticks=5,
        gpus=lambda seq: [
            gpu(i, util=None, temp=None, power=None, mem_used=None)
            for i in range(4)
        ],
    )

    out = SystemCLIComputer(path).compute()

    assert out["gpu_util_avg"] is None  # already correct
    assert out["gpu_mem_used"] is None
    assert out["gpu_temp_max"] is None
    assert out["gpu_power_usage"] is None
    # Free memory is measured against a level, so with no level there is
    # no headroom. 16.1 GB here read as a completely free card.
    assert out["gpu_mem_headroom_min"] is None
    # The capacities the devices DID report are still reported.
    assert out["gpu_mem_total"] == 4 * 16.1 * GB
    assert out["gpu_power_limit"] == 4 * 70.0

    text = _render_text(path)
    assert "0.0°C" not in text
    assert "GPU TMP" in text and "N/A" in text


def test_a_partly_read_host_still_reports(system_db):
    """One device reading is enough; abstention is for none at all."""
    path = system_db(
        ticks=5,
        gpus=lambda seq: [gpu(0)]
        + [
            gpu(i, util=None, temp=None, power=None, mem_used=None)
            for i in range(1, 4)
        ],
    )

    out = SystemCLIComputer(path).compute()

    assert out["gpu_util_avg"] == 100.0
    assert out["gpu_util_devices"] == 1
    assert out["gpu_temp_max"] == 45.0
    assert out["gpu_power_usage"] == 66.0
    assert out["gpu_mem_used"] == 6.3 * GB
