"""Selected-clock Step Time metrics for diagnosis.

This module is intentionally diagnosis-only. Public Step Time renderers and
summary rollups continue to use their existing ``duration_ms`` semantics. The
diagnosis path selects one clock for a whole analyzed window:

- GPU when every rank/step has GPU event timing for the step envelope,
  dataloader/input wait, and traced phase events present in the window.
- CPU otherwise, using explicit ``cpu_ms`` fields.

The selected metrics are exposed with generic names so diagnosis rules do not
need to know whether CPU or GPU fields backed the window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional, Sequence

import numpy as np

from traceml_ai.renderers.step_time.schema import (
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeSeries,
    StepCombinedTimeSummary,
)

DiagnosisClock = Literal["cpu", "gpu"]

DATALOADER_EVENT_NAME = "_traceml_internal:dataloader_next"
STEP_TIME_EVENT_NAME = "_traceml_internal:step_time"

INPUT_WAIT_KEY = "input_wait"
DIAGNOSIS_CLOCK_KEY = "diagnosis_clock"

INPUT_WAIT_CPU_MS_KEY = "input_wait_cpu_ms"
INPUT_WAIT_GPU_MS_KEY = "input_wait_gpu_ms"
STEP_TIME_CPU_MS_KEY = "step_time_cpu_ms"
STEP_TIME_GPU_MS_KEY = "step_time_gpu_ms"

EVENT_ALIASES: Dict[str, str] = {
    INPUT_WAIT_KEY: DATALOADER_EVENT_NAME,
    "dataloader_fetch": DATALOADER_EVENT_NAME,
    "h2d": "_traceml_internal:h2d_time",
    "forward": "_traceml_internal:forward_time",
    "backward": "_traceml_internal:backward_time",
    "optimizer_step": "_traceml_internal:optimizer_step",
    "step_time": STEP_TIME_EVENT_NAME,
}

_SELECTED_METRICS: tuple[str, ...] = (
    INPUT_WAIT_KEY,
    "h2d",
    "forward",
    "backward",
    "optimizer_step",
    "step_time",
)

_REQUIRED_GPU_METRICS: tuple[str, ...] = (INPUT_WAIT_KEY, "step_time")


@dataclass(frozen=True)
class DiagnosisTimingWindow:
    """Selected-clock per-rank Step Time metrics for one analysis window."""

    clock: DiagnosisClock = "cpu"
    per_rank_timing: Dict[int, Dict[str, float]] = field(default_factory=dict)


def _safe_non_negative_float(value: Any) -> Optional[float]:
    """Return a finite non-negative float, or ``None`` for missing values."""
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


def _event_payload(events: Any, metric_key: str) -> Any:
    if not isinstance(events, Mapping):
        return None
    return events.get(EVENT_ALIASES.get(metric_key, metric_key))


def _event_is_present(events: Any, metric_key: str) -> bool:
    payload = _event_payload(events, metric_key)
    return isinstance(payload, Mapping) and bool(payload)


def _event_cpu_ms(events: Any, metric_key: str) -> Optional[float]:
    return _sum_clock(_event_payload(events, metric_key), "cpu_ms")


def _event_gpu_ms(events: Any, metric_key: str) -> Optional[float]:
    return _sum_clock(_event_payload(events, metric_key), "gpu_ms")


def diagnosis_clock_fields_from_events(events: Any) -> Dict[str, float]:
    """
    Return explicit CPU/GPU fields from one raw Step Time event payload.

    These fields are intermediate data for final-summary diagnosis. They are
    not public report metrics.
    """
    if not isinstance(events, Mapping):
        return {}

    out: Dict[str, float] = {}
    for metric_key in _SELECTED_METRICS:
        cpu_ms = _event_cpu_ms(events, metric_key)
        gpu_ms = _event_gpu_ms(events, metric_key)
        if cpu_ms is not None:
            out[_metric_field(metric_key, "cpu")] = float(cpu_ms)
        if gpu_ms is not None:
            out[_metric_field(metric_key, "gpu")] = float(gpu_ms)

    return out


def _event_has_required_gpu(events: Any) -> bool:
    for metric_key in _REQUIRED_GPU_METRICS:
        if _event_gpu_ms(events, metric_key) is None:
            return False

    for metric_key in _SELECTED_METRICS:
        if metric_key in _REQUIRED_GPU_METRICS:
            continue
        if _event_is_present(events, metric_key) and (
            _event_gpu_ms(events, metric_key) is None
        ):
            return False

    return True


def _select_clock_from_events(
    per_rank_steps: Mapping[int, Mapping[int, Mapping[str, Any]]],
    steps: Sequence[int],
) -> DiagnosisClock:
    if not per_rank_steps or not steps:
        return "cpu"

    for step_map in per_rank_steps.values():
        for step in steps:
            if not _event_has_required_gpu(step_map.get(int(step), {})):
                return "cpu"
    return "gpu"


def _metric_from_events(
    events: Any,
    metric_key: str,
    *,
    clock: DiagnosisClock,
) -> float:
    value = (
        _event_gpu_ms(events, metric_key)
        if clock == "gpu"
        else _event_cpu_ms(events, metric_key)
    )
    return float(value) if value is not None else 0.0


def _build_rank_timing(
    *,
    input_wait: float,
    dataloader_fetch: float,
    h2d: float,
    forward: float,
    backward: float,
    optimizer: float,
    step_time: float,
) -> Dict[str, float]:
    residual = max(0.0, step_time - h2d - forward - backward - optimizer)

    return {
        INPUT_WAIT_KEY: float(input_wait),
        "dataloader_fetch": float(dataloader_fetch),
        "h2d": float(h2d),
        "forward": float(forward),
        "backward": float(backward),
        "optimizer_step": float(optimizer),
        "step_time": float(step_time),
        "residual_proxy": residual,
        "total_step": input_wait + step_time,
    }


def build_diagnosis_timing_from_events(
    per_rank_steps: Mapping[int, Mapping[int, Mapping[str, Any]]],
    steps: Sequence[int],
    *,
    aggregate: Literal["sum", "average"] = "sum",
) -> DiagnosisTimingWindow:
    """Build selected-clock diagnosis metrics from raw event payloads."""
    selected_steps = [int(step) for step in steps]
    if not per_rank_steps or not selected_steps:
        return DiagnosisTimingWindow()

    clock = _select_clock_from_events(per_rank_steps, selected_steps)
    divisor = float(len(selected_steps)) if aggregate == "average" else 1.0
    per_rank_timing: Dict[int, Dict[str, float]] = {}

    for rank, step_map in per_rank_steps.items():
        totals = {metric_key: 0.0 for metric_key in _SELECTED_METRICS}
        dataloader_fetch = 0.0
        for step in selected_steps:
            events = step_map.get(int(step), {})
            dataloader_fetch += _metric_from_events(
                events,
                "dataloader_fetch",
                clock="cpu",
            )
            for metric_key in _SELECTED_METRICS:
                totals[metric_key] += _metric_from_events(
                    events,
                    metric_key,
                    clock=clock,
                )
        if divisor > 1.0:
            totals = {
                metric_key: value / divisor
                for metric_key, value in totals.items()
            }
            dataloader_fetch = dataloader_fetch / divisor
        per_rank_timing[int(rank)] = _build_rank_timing(
            input_wait=totals.get(INPUT_WAIT_KEY, 0.0),
            dataloader_fetch=dataloader_fetch,
            h2d=totals.get("h2d", 0.0),
            forward=totals.get("forward", 0.0),
            backward=totals.get("backward", 0.0),
            optimizer=totals.get("optimizer_step", 0.0),
            step_time=totals.get("step_time", 0.0),
        )

    return DiagnosisTimingWindow(
        clock=clock,
        per_rank_timing=per_rank_timing,
    )


def _metric_field(metric_key: str, clock: str) -> str:
    if metric_key in {INPUT_WAIT_KEY, "dataloader_fetch"}:
        return f"input_wait_{clock}_ms"
    if metric_key == "step_time":
        return f"step_time_{clock}_ms"
    return f"{metric_key}_{clock}_ms"


def _metric_from_step_metrics(
    metrics: Mapping[str, Any],
    metric_key: str,
    *,
    clock: DiagnosisClock,
) -> float:
    if clock == "gpu":
        value = _safe_non_negative_float(
            metrics.get(_metric_field(metric_key, "gpu"))
        )
        return float(value) if value is not None else 0.0

    value = _safe_non_negative_float(
        metrics.get(_metric_field(metric_key, "cpu"))
    )
    return float(value) if value is not None else 0.0


def _step_metrics_has_required_gpu(metrics: Mapping[str, Any]) -> bool:
    for metric_key in _REQUIRED_GPU_METRICS:
        if (
            _safe_non_negative_float(
                metrics.get(_metric_field(metric_key, "gpu"))
            )
            is None
        ):
            return False

    for metric_key in _SELECTED_METRICS:
        if metric_key in _REQUIRED_GPU_METRICS:
            continue
        cpu_present = (
            _safe_non_negative_float(
                metrics.get(_metric_field(metric_key, "cpu"))
            )
            is not None
        )
        gpu_present = (
            _safe_non_negative_float(
                metrics.get(_metric_field(metric_key, "gpu"))
            )
            is not None
        )
        if cpu_present and not gpu_present:
            return False

    return True


def _select_clock_from_step_metrics(
    per_rank_step_metrics: Mapping[int, Mapping[int, Mapping[str, float]]],
    steps: Sequence[int],
) -> DiagnosisClock:
    if not per_rank_step_metrics or not steps:
        return "cpu"

    for step_map in per_rank_step_metrics.values():
        for step in steps:
            metrics = step_map.get(int(step), {})
            if not _step_metrics_has_required_gpu(metrics):
                return "cpu"
    return "gpu"


def build_diagnosis_timing_from_step_metrics(
    per_rank_step_metrics: Mapping[int, Mapping[int, Mapping[str, float]]],
    steps: Sequence[int],
    *,
    aggregate: Literal["sum", "average"] = "average",
) -> DiagnosisTimingWindow:
    """Build selected-clock diagnosis metrics from per-step summary metrics."""
    selected_steps = [int(step) for step in steps]
    if not per_rank_step_metrics or not selected_steps:
        return DiagnosisTimingWindow()

    clock = _select_clock_from_step_metrics(
        per_rank_step_metrics,
        selected_steps,
    )
    divisor = float(len(selected_steps)) if aggregate == "average" else 1.0
    per_rank_timing: Dict[int, Dict[str, float]] = {}

    for rank, step_map in per_rank_step_metrics.items():
        totals = {metric_key: 0.0 for metric_key in _SELECTED_METRICS}
        dataloader_fetch = 0.0
        for step in selected_steps:
            metrics = step_map.get(int(step), {})
            dataloader_fetch += _metric_from_step_metrics(
                metrics,
                "dataloader_fetch",
                clock="cpu",
            )
            for metric_key in _SELECTED_METRICS:
                totals[metric_key] += _metric_from_step_metrics(
                    metrics,
                    metric_key,
                    clock=clock,
                )
        if divisor > 1.0:
            totals = {
                metric_key: value / divisor
                for metric_key, value in totals.items()
            }
            dataloader_fetch = dataloader_fetch / divisor
        per_rank_timing[int(rank)] = _build_rank_timing(
            input_wait=totals.get(INPUT_WAIT_KEY, 0.0),
            dataloader_fetch=dataloader_fetch,
            h2d=totals.get("h2d", 0.0),
            forward=totals.get("forward", 0.0),
            backward=totals.get("backward", 0.0),
            optimizer=totals.get("optimizer_step", 0.0),
            step_time=totals.get("step_time", 0.0),
        )

    return DiagnosisTimingWindow(
        clock=clock,
        per_rank_timing=per_rank_timing,
    )


def build_diagnosis_metrics_from_timing(
    per_rank_timing: Mapping[int, Mapping[str, float]],
    *,
    coverage: StepCombinedTimeCoverage,
    include_series: bool = False,
    series_steps: Optional[Sequence[int]] = None,
    per_rank_step_timing: Optional[
        Mapping[int, Mapping[int, Mapping[str, float]]]
    ] = None,
    worst_rank_override: Optional[int] = None,
) -> list[StepCombinedTimeMetric]:
    """Build diagnosis-only metrics from selected-clock per-rank timing."""
    ranks = sorted(int(rank) for rank in per_rank_timing)
    if not ranks:
        return []

    metrics: list[StepCombinedTimeMetric] = []
    for metric_key in (
        INPUT_WAIT_KEY,
        "h2d",
        "forward",
        "backward",
        "optimizer_step",
        "step_time",
        "residual_proxy",
    ):
        values = {
            rank: _safe_non_negative_float(
                per_rank_timing.get(rank, {}).get(metric_key)
            )
            or 0.0
            for rank in ranks
        }
        arr = np.asarray([values[rank] for rank in ranks], dtype=np.float64)
        if arr.size == 0:
            continue
        median_total = float(np.median(arr))
        worst_idx = int(np.argmax(arr))
        worst_total = float(arr[worst_idx])
        worst_rank = int(ranks[worst_idx])
        if metric_key == "step_time" and worst_rank_override is not None:
            worst_rank = int(worst_rank_override)

        if coverage.ranks_present <= 1:
            median_total = worst_total
            skew_ratio = 0.0
            skew_pct = 0.0
        elif median_total > 0.0:
            skew_ratio = worst_total / median_total
            skew_pct = (worst_total - median_total) / median_total
        else:
            skew_ratio = 0.0
            skew_pct = 0.0

        series = None
        if include_series and metric_key != "residual_proxy":
            step_ids = [int(step) for step in (series_steps or ())]
            if step_ids and per_rank_step_timing:
                median_y: list[float] = []
                worst_y: list[float] = []
                for step in step_ids:
                    step_values = np.asarray(
                        [
                            _safe_non_negative_float(
                                per_rank_step_timing.get(rank, {})
                                .get(step, {})
                                .get(metric_key)
                            )
                            or 0.0
                            for rank in ranks
                        ],
                        dtype=np.float64,
                    )
                    median_y.append(
                        float(np.median(step_values))
                        if step_values.size
                        else 0.0
                    )
                    worst_y.append(
                        float(np.max(step_values)) if step_values.size else 0.0
                    )
                series = StepCombinedTimeSeries(
                    steps=step_ids,
                    median=median_y,
                    worst=worst_y,
                    sum=[0.0] * len(step_ids),
                )

        metrics.append(
            StepCombinedTimeMetric(
                metric=metric_key,
                clock="mixed",
                series=series,
                summary=StepCombinedTimeSummary(
                    window_size=int(coverage.expected_steps),
                    steps_used=int(coverage.steps_used),
                    median_total=float(median_total),
                    worst_total=float(worst_total),
                    worst_rank=int(worst_rank),
                    skew_ratio=float(skew_ratio),
                    skew_pct=float(skew_pct),
                ),
                coverage=coverage,
            )
        )

    return metrics


__all__ = [
    "DIAGNOSIS_CLOCK_KEY",
    "DiagnosisClock",
    "DiagnosisTimingWindow",
    "EVENT_ALIASES",
    "INPUT_WAIT_CPU_MS_KEY",
    "INPUT_WAIT_GPU_MS_KEY",
    "INPUT_WAIT_KEY",
    "STEP_TIME_CPU_MS_KEY",
    "STEP_TIME_GPU_MS_KEY",
    "build_diagnosis_metrics_from_timing",
    "build_diagnosis_timing_from_events",
    "build_diagnosis_timing_from_step_metrics",
    "diagnosis_clock_fields_from_events",
]
