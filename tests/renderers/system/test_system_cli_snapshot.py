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

from tests.renderers.system.conftest import gpu
from traceml_ai.renderers.system.cli_compute import SystemCLIComputer

NVML_FAILED = {
    "gpu_idx": 1,
    "util": 0.0,
    "mem_used_bytes": 0.0,
    "mem_total_bytes": 0.0,
    "temperature_c": 0.0,
    "power_usage_w": 0.0,
    "power_limit_w": 0.0,
}


def _avg(snapshot) -> float:
    """The average the terminal renderer prints, computed its way."""
    devices = snapshot.get("gpu_util_devices") or snapshot.get("gpu_count")
    return snapshot["gpu_util_total"] / max(int(devices or 0), 1)


def test_four_busy_gpus_average_to_full(system_db):
    path = system_db(ticks=20, gpus=lambda seq: [gpu(i) for i in range(4)])
    out = SystemCLIComputer(path).compute()
    assert _avg(out) == 100.0
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
    assert _avg(out) == 100.0
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
    assert _avg(out) == 100.0
    assert out["gpu_util_skew"] == 0.0
