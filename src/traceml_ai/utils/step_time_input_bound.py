"""Input-bound clock selection for Step Time event payloads.

This module is intentionally diagnosis-only. Public step-time metrics keep
using their existing `duration_ms` semantics, while input-bound diagnosis uses
explicit CPU/GPU clocks from the raw event payload.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

DATALOADER_EVENT_NAME = "_traceml_internal:dataloader_next"
STEP_TIME_EVENT_NAME = "_traceml_internal:step_time"

INPUT_WAIT_MS_KEY = "input_wait_ms"
INPUT_BOUND_STEP_MS_KEY = "input_bound_step_ms"
INPUT_BOUND_CLOCK_IS_GPU_KEY = "input_bound_clock_is_gpu"

INPUT_BOUND_CLOCK_CPU = "cpu"
INPUT_BOUND_CLOCK_GPU = "gpu"


@dataclass(frozen=True)
class InputBoundTiming:
    """
    Selected timing values used only for INPUT_BOUND diagnosis.

    `input_wait_ms` and `step_time_ms` use the same clock. When both dataloader
    and step envelope GPU event timings are present, the clock is `gpu`;
    otherwise CPU timing is used when both CPU fields are present.
    """

    input_wait_ms: float
    step_time_ms: float
    clock: str

    @property
    def clock_is_gpu(self) -> float:
        """Return a numeric marker suitable for per-rank timing maps."""
        return 1.0 if self.clock == INPUT_BOUND_CLOCK_GPU else 0.0


def _safe_non_negative_float(value: Any) -> Optional[float]:
    """Return a finite non-negative float, or None for missing values."""
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return max(0.0, out)


def _sum_clock(by_device: Any, field: str) -> Optional[float]:
    """Sum one timing field across devices, preserving missing-vs-zero."""
    if not isinstance(by_device, Mapping):
        return None

    total = 0.0
    found = False
    for stats in by_device.values():
        if not isinstance(stats, Mapping):
            continue
        value = _safe_non_negative_float(stats.get(field))
        if value is None:
            continue
        total += value
        found = True

    return total if found else None


def input_bound_timing_from_events(events: Any) -> Optional[InputBoundTiming]:
    """
    Select explicit clocks for input-bound diagnosis from one step event map.

    Selection order:
    1. dataloader `gpu_ms` and step envelope `gpu_ms` when both are present.
    2. dataloader `cpu_ms` and step envelope `cpu_ms` when both are present.

    `duration_ms` is deliberately ignored so INPUT_BOUND is not coupled to
    compatibility/public metric semantics.
    """
    if not isinstance(events, Mapping):
        return None

    dataloader = events.get(DATALOADER_EVENT_NAME)
    step_time = events.get(STEP_TIME_EVENT_NAME)

    dataloader_gpu_ms = _sum_clock(dataloader, "gpu_ms")
    step_gpu_ms = _sum_clock(step_time, "gpu_ms")
    if dataloader_gpu_ms is not None and step_gpu_ms is not None:
        return InputBoundTiming(
            input_wait_ms=float(dataloader_gpu_ms),
            step_time_ms=float(step_gpu_ms),
            clock=INPUT_BOUND_CLOCK_GPU,
        )

    dataloader_cpu_ms = _sum_clock(dataloader, "cpu_ms")
    step_cpu_ms = _sum_clock(step_time, "cpu_ms")
    if dataloader_cpu_ms is not None and step_cpu_ms is not None:
        return InputBoundTiming(
            input_wait_ms=float(dataloader_cpu_ms),
            step_time_ms=float(step_cpu_ms),
            clock=INPUT_BOUND_CLOCK_CPU,
        )

    return None


def input_bound_timing_fields(events: Any) -> dict[str, float]:
    """Return numeric per-step timing fields for diagnosis maps."""
    timing = input_bound_timing_from_events(events)
    if timing is None:
        return {}
    return {
        INPUT_WAIT_MS_KEY: float(timing.input_wait_ms),
        INPUT_BOUND_STEP_MS_KEY: float(timing.step_time_ms),
        INPUT_BOUND_CLOCK_IS_GPU_KEY: float(timing.clock_is_gpu),
    }
