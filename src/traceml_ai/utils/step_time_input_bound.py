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

INPUT_WAIT_CPU_MS_KEY = "input_wait_cpu_ms"
INPUT_WAIT_GPU_MS_KEY = "input_wait_gpu_ms"
STEP_TIME_CPU_MS_KEY = "step_time_cpu_ms"
STEP_TIME_GPU_MS_KEY = "step_time_gpu_ms"

INPUT_BOUND_CLOCK_CPU = "cpu"
INPUT_BOUND_CLOCK_GPU = "gpu"


@dataclass(frozen=True)
class InputBoundTiming:
    """
    Explicit timing values used only for INPUT_BOUND diagnosis.

    `input_wait_gpu_ms` is a GPU-stream observed wait/gap around input fetch,
    not GPU dataloader work. When the GPU pair is present, INPUT_BOUND selects
    `input_wait_gpu_ms` and `step_time_gpu_ms`. CPU fields may also be recorded
    in that payload, but they are selected only when the GPU pair is absent and
    the CPU pair is present.
    """

    input_wait_cpu_ms: Optional[float] = None
    input_wait_gpu_ms: Optional[float] = None
    step_time_cpu_ms: Optional[float] = None
    step_time_gpu_ms: Optional[float] = None

    @property
    def has_gpu_pair(self) -> bool:
        """Return whether both GPU-clock fields are available."""
        return (
            self.input_wait_gpu_ms is not None
            and self.step_time_gpu_ms is not None
        )

    @property
    def has_cpu_pair(self) -> bool:
        """Return whether both CPU-clock fields are available."""
        return (
            self.input_wait_cpu_ms is not None
            and self.step_time_cpu_ms is not None
        )

    @property
    def clock(self) -> str:
        """Return the clock selected for INPUT_BOUND diagnosis."""
        return (
            INPUT_BOUND_CLOCK_GPU
            if self.has_gpu_pair
            else INPUT_BOUND_CLOCK_CPU
        )


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

    The returned object carries complete CPU and GPU pairs independently. When
    both pairs exist, the GPU pair is the selected diagnosis clock; the CPU pair
    remains available only as explicit recorded data, not as the selected value.

    `duration_ms` is deliberately ignored so INPUT_BOUND is not coupled to
    compatibility/public metric semantics.
    """
    if not isinstance(events, Mapping):
        return None

    dataloader = events.get(DATALOADER_EVENT_NAME)
    step_time = events.get(STEP_TIME_EVENT_NAME)

    dataloader_gpu_ms = _sum_clock(dataloader, "gpu_ms")
    step_gpu_ms = _sum_clock(step_time, "gpu_ms")
    dataloader_cpu_ms = _sum_clock(dataloader, "cpu_ms")
    step_cpu_ms = _sum_clock(step_time, "cpu_ms")

    has_gpu_pair = dataloader_gpu_ms is not None and step_gpu_ms is not None
    has_cpu_pair = dataloader_cpu_ms is not None and step_cpu_ms is not None
    if has_gpu_pair or has_cpu_pair:
        return InputBoundTiming(
            input_wait_cpu_ms=(
                float(dataloader_cpu_ms) if has_cpu_pair else None
            ),
            input_wait_gpu_ms=(
                float(dataloader_gpu_ms) if has_gpu_pair else None
            ),
            step_time_cpu_ms=float(step_cpu_ms) if has_cpu_pair else None,
            step_time_gpu_ms=float(step_gpu_ms) if has_gpu_pair else None,
        )

    return None


def input_bound_timing_fields(events: Any) -> dict[str, float]:
    """Return explicit per-step timing fields for diagnosis maps."""
    timing = input_bound_timing_from_events(events)
    if timing is None:
        return {}

    fields: dict[str, float] = {}
    if timing.has_cpu_pair:
        fields[INPUT_WAIT_CPU_MS_KEY] = float(timing.input_wait_cpu_ms)
        fields[STEP_TIME_CPU_MS_KEY] = float(timing.step_time_cpu_ms)
    if timing.has_gpu_pair:
        fields[INPUT_WAIT_GPU_MS_KEY] = float(timing.input_wait_gpu_ms)
        fields[STEP_TIME_GPU_MS_KEY] = float(timing.step_time_gpu_ms)
    return fields
