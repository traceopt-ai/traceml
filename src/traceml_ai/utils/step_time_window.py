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

REQUIRED_GPU_METRICS: tuple[str, ...] = (INPUT_WAIT_KEY, "step_time")


@dataclass(frozen=True)
class StepTimeWindow:
    """Aligned selected-clock Step Time data for one analysis window."""

    clock: DiagnosisClock = "cpu"
    steps: list[int] = field(default_factory=list)
    expected_ranks: tuple[int, ...] = ()
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

    @property
    def rank_universe(self) -> tuple[int, ...]:
        """Return expected ranks, or observed ranks for direct fixtures."""
        if self.expected_ranks:
            return self.expected_ranks
        return tuple(sorted(int(rank) for rank in self.per_rank_timing))

    def ranks_for(self, metric: str) -> tuple[int, ...]:
        """Return expected ranks carrying one canonical sparse metric."""
        key = str(metric)
        return tuple(
            rank
            for rank in self.rank_universe
            if key in self.per_rank_timing.get(rank, {})
        )

    def eligible_ranks(self, metrics: Sequence[str]) -> tuple[int, ...]:
        """Return ranks carrying every requested metric in this window."""
        keys = tuple(str(metric) for metric in metrics)
        if not keys:
            return self.rank_universe
        return tuple(
            rank
            for rank in self.rank_universe
            if all(key in self.per_rank_timing.get(rank, {}) for key in keys)
        )

    def is_complete(self, metric: str) -> bool:
        """Return whether every expected rank measured one metric."""
        ranks = self.rank_universe
        return bool(ranks) and self.ranks_for(metric) == ranks

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
) -> Optional[float]:
    """Return the selected-clock value for one metric, or None if missing."""
    value = (
        _event_gpu_ms(events, metric_key)
        if clock == "gpu"
        else _event_cpu_ms(events, metric_key)
    )
    return float(value) if value is not None else None


_COMPUTE_KEYS: tuple[str, ...] = ("forward", "backward", "optimizer_step")

# Occurrence-driven metrics legitimately skip steps: the optimizer under
# gradient accumulation and H2D when a step performs no transfers. Every
# other selected metric must occur on every bracketed step, so partial
# presence there means lost instrumentation, not measured zeros.
_OCCURRENCE_METRICS: frozenset[str] = frozenset(("optimizer_step", "h2d"))


def median_iteration_component_share(
    per_rank_timing: Mapping[int, Mapping[str, float]],
    component: str,
) -> Optional[float]:
    """Return one component's median share of complete rank iterations.

    Only ranks carrying the component, input wait, and the step envelope are
    eligible. ``compute`` is derived only when all three compute phases exist.
    No eligible rank returns ``None``; an eligible measured zero returns
    ``0.0``.
    """
    shares: list[float] = []
    for values in per_rank_timing.values():
        if INPUT_WAIT_KEY not in values or "step_time" not in values:
            continue
        if component == "compute":
            if not all(key in values for key in _COMPUTE_KEYS):
                continue
            components = [
                _safe_non_negative_float(values[key]) for key in _COMPUTE_KEYS
            ]
            if any(value is None for value in components):
                continue
            numerator = sum(float(value) for value in components)
        else:
            if component not in values:
                continue
            numerator = _safe_non_negative_float(values[component])
            if numerator is None:
                continue

        input_wait = _safe_non_negative_float(values[INPUT_WAIT_KEY])
        step_time = _safe_non_negative_float(values["step_time"])
        if input_wait is None or step_time is None:
            continue
        iteration = input_wait + step_time
        if iteration > 0.0:
            shares.append(float(numerator) / iteration)

    if not shares:
        return None
    return float(np.median(np.asarray(shares, dtype=np.float64)))


def _rank_metric_availability(
    step_map: Mapping[int, Mapping[str, Any]],
    steps: Sequence[int],
    *,
    clock: DiagnosisClock,
) -> frozenset[str]:
    """Return the metric keys considered measured for this rank's window.

    Availability is metric-specific. Occurrence-driven metrics
    (``optimizer_step``, ``h2d``) are available when measured in at
    least one aligned step; their absent steps are true zeros. Every
    other metric must be measured in every aligned step: intermittent
    presence means the instrumentation dropped out mid-window, and
    averaging the fragments would understate the phase and leak the
    missing work into the residual.
    """
    if not steps:
        return frozenset()
    seen_counts: Dict[str, int] = {}
    for step in steps:
        events = step_map.get(int(step), {})
        for metric_key in SELECTED_METRICS:
            if (
                _metric_from_events(events, metric_key, clock=clock)
                is not None
            ):
                seen_counts[metric_key] = seen_counts.get(metric_key, 0) + 1
        if _event_cpu_ms(events, INPUT_WAIT_KEY) is not None:
            seen_counts[DATALOADER_FETCH_KEY] = (
                seen_counts.get(DATALOADER_FETCH_KEY, 0) + 1
            )
        if _event_cpu_ms(events, "step_time") is not None:
            seen_counts[STEP_TIME_CPU_KEY] = (
                seen_counts.get(STEP_TIME_CPU_KEY, 0) + 1
            )
    total = len(steps)
    available = {
        metric_key
        for metric_key, count in seen_counts.items()
        if count == total or metric_key in _OCCURRENCE_METRICS
    }
    return frozenset(available)


def _add_derived_step_metrics(
    timing: Dict[str, float],
    *,
    available: frozenset[str],
    clock: DiagnosisClock,
) -> None:
    """Attach derived metrics whose required inputs are available.

    ``residual_proxy`` needs the step envelope plus every compute phase.
    H2D events are occurrence-driven (a fully instrumented run with no
    host-to-device copies emits none), so an unavailable H2D contributes
    zero rather than blocking the residual. ``total_step`` needs the
    input wait plus the step envelope.
    """
    del clock
    compute_available = all(key in available for key in _COMPUTE_KEYS)
    if "step_time" in available and compute_available:
        timing["residual_proxy"] = max(
            0.0,
            timing["step_time"]
            - timing.get("h2d", 0.0)
            - timing["forward"]
            - timing["backward"]
            - timing["optimizer_step"],
        )
    if INPUT_WAIT_KEY in available and "step_time" in available:
        timing["total_step"] = timing[INPUT_WAIT_KEY] + timing["step_time"]


def _average_rank_timing(
    per_rank_step_timing: Mapping[int, Mapping[int, Mapping[str, float]]],
    steps: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    """Average each rank's measured metrics over the aligned steps.

    Only keys present in the rank's step rows are averaged, so a metric
    missing from the whole window stays absent instead of averaging to a
    fake zero. The divisor stays the full step count: an available
    metric absent at one step did no work at that step.
    """
    out: Dict[int, Dict[str, float]] = {}
    divisor = float(len(steps)) if steps else 1.0
    for rank, step_map in per_rank_step_timing.items():
        keys: set[str] = set()
        for step in steps:
            keys.update(step_map.get(int(step), {}).keys())
        totals = {metric_key: 0.0 for metric_key in keys}
        for step in steps:
            metrics = step_map.get(int(step), {})
            for metric_key in keys:
                value = _safe_non_negative_float(metrics.get(metric_key))
                totals[metric_key] += (
                    float(value) if value is not None else 0.0
                )
        averaged = {
            metric_key: float(value / divisor)
            for metric_key, value in totals.items()
        }
        if "total_step" in averaged:
            # Re-derive from the averaged components so complete-data
            # output stays bit-identical to the historical
            # avg(input_wait) + avg(step_time) formulation.
            averaged["total_step"] = averaged.get(
                INPUT_WAIT_KEY, 0.0
            ) + averaged.get("step_time", 0.0)
        out[int(rank)] = averaged
    return out


def _selected_step_timing_from_events(
    per_rank_steps: Mapping[int, Mapping[int, Mapping[str, Any]]],
    steps: Sequence[int],
    *,
    clock: DiagnosisClock,
) -> Dict[int, Dict[int, Dict[str, float]]]:
    """Build sparse per-step timing rows from raw event payloads.

    A metric key appears only when it was measured somewhere in the
    rank's window; a measured zero stays ``0.0``.
    """
    out: Dict[int, Dict[int, Dict[str, float]]] = {}
    for rank, step_map in per_rank_steps.items():
        available = _rank_metric_availability(step_map, steps, clock=clock)
        rank_timing: Dict[int, Dict[str, float]] = {}
        for step in steps:
            events = step_map.get(int(step), {})
            timing: Dict[str, float] = {}
            for metric_key in SELECTED_METRICS:
                if metric_key not in available:
                    continue
                value = _metric_from_events(events, metric_key, clock=clock)
                timing[metric_key] = float(value) if value is not None else 0.0
            if DATALOADER_FETCH_KEY in available:
                timing[DATALOADER_FETCH_KEY] = (
                    _event_cpu_ms(events, INPUT_WAIT_KEY) or 0.0
                )
            if STEP_TIME_CPU_KEY in available:
                timing[STEP_TIME_CPU_KEY] = (
                    _event_cpu_ms(events, "step_time") or 0.0
                )
            _add_derived_step_metrics(timing, available=available, clock=clock)
            rank_timing[int(step)] = timing
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
    """Return rank values for ranks that measured this metric.

    Ranks without the key are excluded so a missing metric cannot enter
    medians or worst-rank picks as a fake zero; a measured zero stays.
    """
    out: Dict[int, float] = {}
    for rank, values in per_rank_timing.items():
        if metric_key not in values:
            continue
        value = _safe_non_negative_float(values.get(metric_key))
        if value is not None:
            out[int(rank)] = value
    return out


def worst_rank_by_total_step(
    per_rank_timing: Mapping[int, Mapping[str, float]],
) -> Optional[int]:
    candidates = _metric_values(per_rank_timing, "total_step")
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda rank: (candidates[rank], -rank),
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
) -> list[StepCombinedTimeMetric]:
    """Build selected-clock average metrics for diagnosis and display.

    A metric measured by no rank emits no entry at all; a metric
    measured by a subset of ranks builds its statistics over exactly
    those ranks. Each summary's ``worst_total`` and ``worst_rank`` come
    from one ordering: the rank named is always the rank that produced
    the reported worst value for that same metric.
    """
    ranks = sorted(int(rank) for rank in per_rank_timing)
    if not ranks:
        return []

    metrics: list[StepCombinedTimeMetric] = []
    for metric_key in DISPLAY_METRICS:
        values = _metric_values(per_rank_timing, metric_key)
        if not values:
            continue
        metric_ranks = sorted(values)
        arr = np.asarray(
            [values[rank] for rank in metric_ranks], dtype=np.float64
        )

        median_total = float(np.median(arr))
        worst_idx = int(np.argmax(arr))
        worst_total = float(arr[worst_idx])
        worst_rank = int(metric_ranks[worst_idx])

        if len(metric_ranks) <= 1:
            median_total = worst_total
            skew_ratio = None
            skew_pct = None
        elif median_total > 0.0:
            skew_ratio = worst_total / median_total
            skew_pct = (worst_total - median_total) / median_total
        elif worst_total <= 0.0:
            skew_ratio = 0.0
            skew_pct = 0.0
        else:
            # Relative skew is undefined when the median is zero but at
            # least one rank is non-zero. Reporting 0% would claim equality.
            skew_ratio = None
            skew_pct = None

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
                        for rank in metric_ranks
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
                    skew_ratio=(
                        float(skew_ratio) if skew_ratio is not None else None
                    ),
                    skew_pct=(
                        float(skew_pct) if skew_pct is not None else None
                    ),
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
    expected = tuple(
        sorted(
            {int(rank) for rank in (expected_ranks or per_rank_steps.keys())}
        )
    )
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
            expected_ranks=expected,
            coverage=_empty_coverage(
                max_rows=max_rows,
                completed_step=latest_step,
                world_size=len(expected),
                ranks_present=len(observed_steps),
            ),
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
    metrics = build_step_time_metrics(
        per_rank_timing,
        coverage=coverage,
        clock=clock,
        series_steps=steps,
        per_rank_step_timing=per_rank_step_timing,
    )
    return StepTimeWindow(
        clock=clock,
        steps=[int(step) for step in steps],
        expected_ranks=expected,
        coverage=coverage,
        per_rank_step_timing=per_rank_step_timing,
        per_rank_timing=per_rank_timing,
        metrics=metrics,
    )


def diagnose_step_time_window(
    window: StepTimeWindow,
    *,
    policy: "StepTimeDiagnosisPolicy",
    training_strategy: str = "ddp",
) -> "DiagnosticResult[StepDiagnosis]":
    """Run shared Step Time diagnosis over one selected-clock window."""
    from traceml_ai.diagnostics.step_time.api import (
        build_step_diagnosis_result,
    )

    if not window.metrics:
        return build_step_diagnosis_result([], thresholds=policy.thresholds)
    return build_step_diagnosis_result(
        window.metrics,
        thresholds=policy.thresholds,
        per_rank_timing=window.per_rank_timing,
        expected_ranks=window.rank_universe,
        diagnosis_clock=window.clock,
        training_strategy=training_strategy,
    )


def public_step_time_metric_values(
    timing: Mapping[str, float],
) -> Dict[str, Optional[float]]:
    """Map window timing to stable final_summary metric names.

    Selected-clock fields expose diagnosis timing. Dataloader and total-step
    compatibility fields use explicit CPU timings retained by the window.
    Every public key is always present; a metric whose signal was never
    measured in the window is ``None``, while a measured zero stays ``0.0``.
    """

    def _value(key: str) -> Optional[float]:
        if key not in timing:
            return None
        return float(timing[key])

    forward = _value("forward")
    backward = _value("backward")
    optimizer = _value("optimizer_step")
    compute = (
        forward + backward + optimizer
        if None not in (forward, backward, optimizer)
        else None
    )
    dataloader_fetch = _value(DATALOADER_FETCH_KEY)
    step_time_cpu = _value(STEP_TIME_CPU_KEY)
    total_step = (
        dataloader_fetch + step_time_cpu
        if None not in (dataloader_fetch, step_time_cpu)
        else None
    )
    residual_value = _value("residual_proxy")
    residual = max(0.0, residual_value) if residual_value is not None else None
    return {
        "total_step_ms": total_step,
        "dataloader_ms": dataloader_fetch,
        "input_wait_ms": _value(INPUT_WAIT_KEY),
        "step_time_ms": _value("step_time"),
        "h2d_ms": _value("h2d"),
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
    "median_iteration_component_share",
    "public_step_time_metric_values",
    "worst_rank_by_total_step",
]
