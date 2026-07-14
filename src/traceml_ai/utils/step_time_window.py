"""Canonical selected-clock Step Time windows.

This module is the shared Step Time pipeline used by live renderers and final
summary reporting. It aligns completed steps, selects one timing clock for the
whole window, builds per-rank average metrics, and exposes the metrics consumed
by diagnosis and presentation layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
)

import numpy as np

from traceml_ai.renderers.step_time.schema import (
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeSeries,
    StepCombinedTimeSummary,
)
from traceml_ai.utils.step_windows import common_suffix_steps

if TYPE_CHECKING:
    from traceml_ai.diagnostics.common import DiagnosticResult
    from traceml_ai.diagnostics.step_time.api import StepDiagnosis
    from traceml_ai.diagnostics.step_time.policy import StepTimeDiagnosisPolicy

DiagnosisClock = Literal["cpu", "gpu"]

DATALOADER_EVENT_NAME = "_traceml_internal:dataloader_next"
STEP_TIME_EVENT_NAME = "_traceml_internal:step_time"

INPUT_WAIT_KEY = "input_wait"
DIAGNOSIS_CLOCK_KEY = "diagnosis_clock"
DATALOADER_FETCH_KEY = "dataloader_fetch"
STEP_TIME_CPU_KEY = "step_time_cpu"

EVENT_ALIASES: Dict[str, str] = {
    INPUT_WAIT_KEY: DATALOADER_EVENT_NAME,
    "h2d": "_traceml_internal:h2d_time",
    "forward": "_traceml_internal:forward_time",
    "backward": "_traceml_internal:backward_time",
    "optimizer_step": "_traceml_internal:optimizer_step",
    "step_time": STEP_TIME_EVENT_NAME,
}

SELECTED_METRICS: tuple[str, ...] = (
    INPUT_WAIT_KEY,
    "h2d",
    "forward",
    "backward",
    "optimizer_step",
    "step_time",
)

DISPLAY_METRICS: tuple[str, ...] = (
    INPUT_WAIT_KEY,
    "h2d",
    "forward",
    "backward",
    "optimizer_step",
    "step_time",
    "residual_proxy",
)

WINDOW_AVERAGE_METRICS: tuple[str, ...] = DISPLAY_METRICS + (
    DATALOADER_FETCH_KEY,
    STEP_TIME_CPU_KEY,
)

REQUIRED_GPU_METRICS: tuple[str, ...] = (INPUT_WAIT_KEY, "step_time")


@dataclass(frozen=True)
class StepTimeWindow:
    """Aligned selected-clock Step Time data for one analysis window."""

    clock: DiagnosisClock = "cpu"
    steps: list[int] = field(default_factory=list)
    coverage: StepCombinedTimeCoverage = field(
        default_factory=lambda: StepCombinedTimeCoverage(
            expected_steps=0,
            steps_used=0,
            completed_step=0,
            world_size=0,
            ranks_present=0,
            incomplete=False,
        )
    )
    per_rank_step_timing: Dict[int, Dict[int, Dict[str, float]]] = field(
        default_factory=dict
    )
    per_rank_timing: Dict[int, Dict[str, float]] = field(default_factory=dict)
    metrics: list[StepCombinedTimeMetric] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        """Return the aligned step-window block used by final_summary."""
        return {
            "alignment": "common_steps",
            "aligned_steps_analyzed": int(self.coverage.steps_used),
            "start_step": self.steps[0] if self.steps else None,
            "end_step": self.steps[-1] if self.steps else None,
            "window_size": int(self.coverage.expected_steps),
            DIAGNOSIS_CLOCK_KEY: self.clock,
        }


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


def _event_has_required_gpu(events: Any) -> bool:
    for metric_key in REQUIRED_GPU_METRICS:
        if _event_gpu_ms(events, metric_key) is None:
            return False

    for metric_key in SELECTED_METRICS:
        if metric_key in REQUIRED_GPU_METRICS:
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
    step_time_cpu: float,
) -> Dict[str, float]:
    residual = max(0.0, step_time - h2d - forward - backward - optimizer)

    return {
        INPUT_WAIT_KEY: float(input_wait),
        DATALOADER_FETCH_KEY: float(dataloader_fetch),
        "h2d": float(h2d),
        "forward": float(forward),
        "backward": float(backward),
        "optimizer_step": float(optimizer),
        "step_time": float(step_time),
        STEP_TIME_CPU_KEY: float(step_time_cpu),
        "residual_proxy": residual,
        "total_step": input_wait + step_time,
    }


def _average_rank_timing(
    per_rank_step_timing: Mapping[int, Mapping[int, Mapping[str, float]]],
    steps: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    divisor = float(len(steps)) if steps else 1.0
    for rank, step_map in per_rank_step_timing.items():
        totals = {metric_key: 0.0 for metric_key in WINDOW_AVERAGE_METRICS}
        for step in steps:
            metrics = step_map.get(int(step), {})
            for metric_key in WINDOW_AVERAGE_METRICS:
                value = _safe_non_negative_float(metrics.get(metric_key))
                totals[metric_key] += (
                    float(value) if value is not None else 0.0
                )
        out[int(rank)] = {
            metric_key: float(value / divisor)
            for metric_key, value in totals.items()
        }
        out[int(rank)]["total_step"] = out[int(rank)].get(
            INPUT_WAIT_KEY, 0.0
        ) + out[int(rank)].get("step_time", 0.0)
    return out


def _selected_step_timing_from_events(
    per_rank_steps: Mapping[int, Mapping[int, Mapping[str, Any]]],
    steps: Sequence[int],
    *,
    clock: DiagnosisClock,
) -> Dict[int, Dict[int, Dict[str, float]]]:
    out: Dict[int, Dict[int, Dict[str, float]]] = {}
    for rank, step_map in per_rank_steps.items():
        rank_timing: Dict[int, Dict[str, float]] = {}
        for step in steps:
            events = step_map.get(int(step), {})
            selected = {
                metric_key: _metric_from_events(
                    events, metric_key, clock=clock
                )
                for metric_key in SELECTED_METRICS
            }
            rank_timing[int(step)] = _build_rank_timing(
                input_wait=selected.get(INPUT_WAIT_KEY, 0.0),
                dataloader_fetch=_event_cpu_ms(events, INPUT_WAIT_KEY) or 0.0,
                h2d=selected.get("h2d", 0.0),
                forward=selected.get("forward", 0.0),
                backward=selected.get("backward", 0.0),
                optimizer=selected.get("optimizer_step", 0.0),
                step_time=selected.get("step_time", 0.0),
                step_time_cpu=_event_cpu_ms(events, "step_time") or 0.0,
            )
        out[int(rank)] = rank_timing
    return out


def _empty_coverage(
    *,
    max_rows: int,
    completed_step: int,
    world_size: int,
    ranks_present: int,
) -> StepCombinedTimeCoverage:
    return StepCombinedTimeCoverage(
        expected_steps=max(1, int(max_rows)),
        steps_used=0,
        completed_step=int(completed_step),
        world_size=max(0, int(world_size)),
        ranks_present=max(0, int(ranks_present)),
        incomplete=False,
    )


def _metric_values(
    per_rank_timing: Mapping[int, Mapping[str, float]],
    metric_key: str,
) -> Dict[int, float]:
    return {
        int(rank): (_safe_non_negative_float(values.get(metric_key)) or 0.0)
        for rank, values in per_rank_timing.items()
    }


def _worst_rank_by_total_step(
    per_rank_timing: Mapping[int, Mapping[str, float]],
) -> Optional[int]:
    if not per_rank_timing:
        return None
    return max(
        (int(rank) for rank in per_rank_timing),
        key=lambda rank: (
            _safe_non_negative_float(
                per_rank_timing.get(rank, {}).get("total_step")
            )
            or 0.0,
            -rank,
        ),
    )


def build_step_time_metrics(
    per_rank_timing: Mapping[int, Mapping[str, float]],
    *,
    coverage: StepCombinedTimeCoverage,
    clock: DiagnosisClock,
    series_steps: Optional[Sequence[int]] = None,
    per_rank_step_timing: Optional[
        Mapping[int, Mapping[int, Mapping[str, float]]]
    ] = None,
    worst_rank_override: Optional[int] = None,
) -> list[StepCombinedTimeMetric]:
    """Build selected-clock average metrics for diagnosis and display."""
    ranks = sorted(int(rank) for rank in per_rank_timing)
    if not ranks:
        return []

    metrics: list[StepCombinedTimeMetric] = []
    for metric_key in DISPLAY_METRICS:
        values = _metric_values(per_rank_timing, metric_key)
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
        step_ids = [int(step) for step in (series_steps or ())]
        if step_ids and per_rank_step_timing:
            median_y: list[float] = []
            worst_y: list[float] = []
            sum_y: list[float] = []
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
                    float(np.median(step_values)) if step_values.size else 0.0
                )
                worst_y.append(
                    float(np.max(step_values)) if step_values.size else 0.0
                )
                sum_y.append(
                    float(np.sum(step_values)) if step_values.size else 0.0
                )
            series = StepCombinedTimeSeries(
                steps=step_ids,
                median=median_y,
                worst=worst_y,
                sum=sum_y,
            )

        metrics.append(
            StepCombinedTimeMetric(
                metric=metric_key,
                clock=clock,
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


def build_step_time_window_from_events(
    per_rank_steps: Mapping[int, Mapping[int, Mapping[str, Any]]],
    *,
    max_rows: int,
    expected_ranks: Optional[Sequence[int]] = None,
    completed_step: Optional[int] = None,
) -> StepTimeWindow:
    """Build one selected-clock window directly from raw event payloads."""
    expected = [
        int(rank) for rank in (expected_ranks or per_rank_steps.keys())
    ]
    observed_steps = {
        int(rank): {int(step): events for step, events in step_map.items()}
        for rank, step_map in per_rank_steps.items()
        if step_map
    }
    latest_step = (
        int(completed_step)
        if completed_step is not None
        else max(
            (
                max(step_map)
                for step_map in observed_steps.values()
                if step_map
            ),
            default=0,
        )
    )
    steps = common_suffix_steps(observed_steps, max_rows)
    if not observed_steps or not steps:
        return StepTimeWindow(
            coverage=_empty_coverage(
                max_rows=max_rows,
                completed_step=latest_step,
                world_size=len(expected),
                ranks_present=len(observed_steps),
            )
        )

    clock = _select_clock_from_events(observed_steps, steps)
    per_rank_step_timing = _selected_step_timing_from_events(
        observed_steps,
        steps,
        clock=clock,
    )
    per_rank_timing = _average_rank_timing(per_rank_step_timing, steps)
    coverage = StepCombinedTimeCoverage(
        expected_steps=max(1, int(max_rows)),
        steps_used=len(steps),
        completed_step=int(steps[-1]),
        world_size=len(expected),
        ranks_present=len(per_rank_step_timing),
        incomplete=(len(per_rank_step_timing) < len(expected)),
    )
    worst_rank = _worst_rank_by_total_step(per_rank_timing)
    metrics = build_step_time_metrics(
        per_rank_timing,
        coverage=coverage,
        clock=clock,
        series_steps=steps,
        per_rank_step_timing=per_rank_step_timing,
        worst_rank_override=worst_rank,
    )
    return StepTimeWindow(
        clock=clock,
        steps=[int(step) for step in steps],
        coverage=coverage,
        per_rank_step_timing=per_rank_step_timing,
        per_rank_timing=per_rank_timing,
        metrics=metrics,
    )


def diagnose_step_time_window(
    window: StepTimeWindow,
    *,
    policy: "StepTimeDiagnosisPolicy",
) -> "DiagnosticResult[StepDiagnosis]":
    """Run Step Time diagnosis over one canonical selected-clock window."""
    from traceml_ai.diagnostics.step_time.api import (
        build_step_diagnosis_result,
        build_step_warmup_diagnosis,
    )

    if not window.metrics:
        return build_step_diagnosis_result([], thresholds=policy.thresholds)
    if int(window.coverage.steps_used) < int(policy.min_steps_for_diag):
        return build_step_warmup_diagnosis(
            steps_used=int(window.coverage.steps_used),
            required_steps=int(policy.min_steps_for_diag),
        )
    return build_step_diagnosis_result(
        window.metrics,
        thresholds=policy.thresholds,
        per_rank_timing=window.per_rank_timing,
        diagnosis_clock=window.clock,
    )


def public_step_time_metric_values(
    timing: Mapping[str, float]
) -> Dict[str, float]:
    """Map window timing to stable final_summary metric names.

    Selected-clock fields expose diagnosis timing. Dataloader and total-step
    compatibility fields use explicit CPU timings retained by the window.
    """
    forward = float(timing.get("forward", 0.0))
    backward = float(timing.get("backward", 0.0))
    optimizer = float(timing.get("optimizer_step", 0.0))
    compute = forward + backward + optimizer
    input_wait = float(timing.get(INPUT_WAIT_KEY, 0.0))
    step_time = float(timing.get("step_time", 0.0))
    dataloader_fetch = float(timing.get(DATALOADER_FETCH_KEY, 0.0))
    step_time_cpu = float(timing.get(STEP_TIME_CPU_KEY, 0.0))
    h2d = float(timing.get("h2d", 0.0))
    residual_value = timing.get("residual_proxy")
    if residual_value is None:
        residual = max(0.0, step_time - h2d - compute)
    else:
        residual = max(0.0, float(residual_value))
    return {
        "total_step_ms": dataloader_fetch + step_time_cpu,
        "dataloader_ms": dataloader_fetch,
        "input_wait_ms": input_wait,
        "step_time_ms": step_time,
        "h2d_ms": h2d,
        "compute_ms": compute,
        "residual_ms": residual,
        "forward_ms": forward,
        "backward_ms": backward,
        "optimizer_ms": optimizer,
    }


__all__ = [
    "DIAGNOSIS_CLOCK_KEY",
    "DATALOADER_EVENT_NAME",
    "DATALOADER_FETCH_KEY",
    "DISPLAY_METRICS",
    "DiagnosisClock",
    "EVENT_ALIASES",
    "INPUT_WAIT_KEY",
    "SELECTED_METRICS",
    "STEP_TIME_EVENT_NAME",
    "STEP_TIME_CPU_KEY",
    "StepTimeWindow",
    "build_step_time_metrics",
    "build_step_time_window_from_events",
    "diagnose_step_time_window",
    "public_step_time_metric_values",
]
