# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts and reporting helpers for System telemetry renderers.

``SystemCLISnapshot`` defines the terminal payload. Reporting helpers are
shared by the dashboard and terminal computations so both surfaces interpret
missing GPU rows consistently. Rolling-series policy remains in
``renderers.shared.run_series``.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


def positive(value: Any) -> Optional[float]:
    """A number above zero, or None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0.0 else None


def gpu_reported(row: Any) -> bool:
    """Whether a GPU row contains a positive hardware-capacity signal.

    The current sampler writes ``None`` when NVML cannot read a device.
    Older traces may contain an all-zero placeholder for the same state;
    neither form is a measurement of 0 W, 0 C, or 0 GB.

    Lives here rather than in one computer because BOTH System surfaces
    aggregate the same rows: the dashboard and the terminal card. When it
    existed in only one of them, the terminal card averaged a failed
    device in as a real zero and reported a healthy four-GPU host at 75%.
    """
    return (
        positive(row["mem_total_bytes"]) is not None
        or positive(row["power_limit_w"]) is not None
    )


@dataclass(frozen=True)
class SystemCLISnapshot:
    """Compact CLI snapshot for system telemetry."""

    cpu: float
    ram_used: float
    ram_total: float

    gpu_available: bool
    gpu_count: int

    gpu_util_total: Optional[float]
    # The display-ready mean over `gpu_util_devices`. Compute owns this
    # arithmetic; renderers only format the result.
    gpu_util_avg: Optional[float]
    gpu_util_skew: Optional[float]
    gpu_mem_used: Optional[float]
    gpu_mem_total: Optional[float]
    gpu_mem_headroom_min: Optional[float]
    gpu_mem_headroom_min_idx: Optional[int]

    gpu_temp_max: Optional[float]
    gpu_power_usage: Optional[float]
    gpu_power_limit: Optional[float]

    # Devices the util total was summed over, which is NOT gpu_count when
    # a device failed to report. Zero means the current sample had no util
    # readings; None preserves compatibility with payloads that predate
    # this field.
    gpu_util_devices: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu": self.cpu,
            "ram_used": self.ram_used,
            "ram_total": self.ram_total,
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count,
            "gpu_util_total": self.gpu_util_total,
            "gpu_util_avg": self.gpu_util_avg,
            "gpu_util_devices": self.gpu_util_devices,
            "gpu_mem_used": self.gpu_mem_used,
            "gpu_mem_total": self.gpu_mem_total,
            "gpu_temp_max": self.gpu_temp_max,
            "gpu_power_usage": self.gpu_power_usage,
            "gpu_power_limit": self.gpu_power_limit,
            "gpu_util_skew": self.gpu_util_skew,
            "gpu_mem_headroom_min": self.gpu_mem_headroom_min,
            "gpu_mem_headroom_min_idx": self.gpu_mem_headroom_min_idx,
        }
