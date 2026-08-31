# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Shared models and SQLite helpers for system telemetry."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Whole-run CPU history uses a duration-based rolling window before sampling.
# The duration keeps the aggregation consistent across sampling cadences.
_ROLL_MIN_S = 30.0
_ROLL_MAX_S = 300.0
_ROLL_FRACTION = 50.0  # about a fiftieth of the run
_MAX_RUN_POINTS = 120


def choose_window_s(span_s: float) -> float:
    """The rolling window for a run of ``span_s`` seconds, in round steps."""
    if span_s <= 0:
        return _ROLL_MIN_S
    raw = max(_ROLL_MIN_S, min(_ROLL_MAX_S, span_s / _ROLL_FRACTION))
    for step in (30.0, 60.0, 120.0, 300.0):
        if raw <= step:
            return step
    return _ROLL_MAX_S


@dataclass(frozen=True)
class SystemCLISnapshot:
    """Compact CLI snapshot for system telemetry."""

    cpu: float
    ram_used: float
    ram_total: float

    gpu_available: bool
    gpu_count: int

    gpu_util_total: Optional[float]
    gpu_util_skew: Optional[float]
    gpu_mem_used: Optional[float]
    gpu_mem_total: Optional[float]
    gpu_mem_headroom_min: Optional[float]
    gpu_mem_headroom_min_idx: Optional[int]

    gpu_temp_max: Optional[float]
    gpu_power_usage: Optional[float]
    gpu_power_limit: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu": self.cpu,
            "ram_used": self.ram_used,
            "ram_total": self.ram_total,
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count,
            "gpu_util_total": self.gpu_util_total,
            "gpu_mem_used": self.gpu_mem_used,
            "gpu_mem_total": self.gpu_mem_total,
            "gpu_temp_max": self.gpu_temp_max,
            "gpu_power_usage": self.gpu_power_usage,
            "gpu_power_limit": self.gpu_power_limit,
            "gpu_util_skew": self.gpu_util_skew,
            "gpu_mem_headroom_min": self.gpu_mem_headroom_min,
            "gpu_mem_headroom_min_idx": self.gpu_mem_headroom_min_idx,
        }


@dataclass(frozen=True)
class SystemDashboardPayload:
    """Dashboard payload for system telemetry."""

    window_len: int
    gpu_available: bool
    rollups: Dict[str, Any]
    # Series includes both flat sample arrays and structured whole-run
    # histories, so its values are intentionally heterogeneous.
    series: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_len": self.window_len,
            "gpu_available": self.gpu_available,
            "rollups": self.rollups,
            "series": self.series,
        }
