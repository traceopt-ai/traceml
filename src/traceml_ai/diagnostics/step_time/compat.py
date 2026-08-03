# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated adapters for the released rank-mapping diagnosis API.

Nothing in TraceML's production pipeline imports this module. It exists for
one compatibility release so external callers can move from sparse rank maps
to :func:`diagnose_step_time_window` without a patch-release import failure.
"""

from __future__ import annotations

import math
import statistics
import warnings
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from traceml_ai.step_time.model import (
    StepTimeCoverage,
    StepTimeMetric,
    StepTimeRankFacts,
    StepTimeValues,
    StepTimeWindow,
)

from .policy import DEFAULT_THRESHOLDS, DiagnosisThresholds

if TYPE_CHECKING:
    from traceml_ai.diagnostics.common import DiagnosticResult

    from .api import StepDiagnosis


def _warn(name: str) -> None:
    warnings.warn(
        f"{name}() is deprecated; pass StepTimeWindow to "
        "diagnose_step_time_window() instead.",
        DeprecationWarning,
        stacklevel=4,
    )


def _safe_non_negative_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0.0, result) if math.isfinite(result) else None


def _values_from_mapping(values: Mapping[str, float]) -> StepTimeValues:
    """Translate one historical sparse rank row into typed facts."""

    def optional(key: str) -> Optional[float]:
        return float(values[key]) if key in values else None

    forward = optional("forward")
    backward = optional("backward")
    optimizer = optional("optimizer_step")
    compute = optional("compute")
    if compute is None and None not in (forward, backward, optimizer):
        compute = float(forward + backward + optimizer)

    input_wait = optional("input_wait")
    step_time = optional("step_time")
    total_step = optional("total_step")
    if total_step is None and input_wait is not None and step_time is not None:
        total_step = input_wait + step_time

    dataloader_cpu = optional("dataloader_fetch")
    step_time_cpu = optional("step_time_cpu")
    return StepTimeValues(
        input_wait_ms=input_wait,
        h2d_ms=optional("h2d"),
        forward_ms=forward,
        backward_ms=backward,
        optimizer_step_ms=optimizer,
        step_time_ms=step_time,
        compute_ms=compute,
        residual_ms=optional("residual_proxy"),
        total_step_ms=total_step,
        dataloader_cpu_ms=dataloader_cpu,
        step_time_cpu_ms=step_time_cpu,
        total_step_cpu_ms=(
            dataloader_cpu + step_time_cpu
            if dataloader_cpu is not None and step_time_cpu is not None
            else None
        ),
    )


def _component_share(
    timing: Mapping[int, Mapping[str, float]],
    component: str,
) -> Optional[float]:
    """Return the historical median component/iteration share."""
    shares: list[float] = []
    compute_keys = ("forward", "backward", "optimizer_step")
    for values in timing.values():
        if "input_wait" not in values or "step_time" not in values:
            continue
        if component == "compute":
            if not all(key in values for key in compute_keys):
                continue
            components = tuple(
                _safe_non_negative_float(values[key]) for key in compute_keys
            )
            if any(value is None for value in components):
                continue
            numerator = sum(float(value) for value in components)
        else:
            if component not in values:
                continue
            safe_component = _safe_non_negative_float(values[component])
            if safe_component is None:
                continue
            numerator = safe_component

        input_wait = _safe_non_negative_float(values["input_wait"])
        step_time = _safe_non_negative_float(values["step_time"])
        if input_wait is None or step_time is None:
            continue
        iteration = input_wait + step_time
        if iteration > 0.0:
            shares.append(max(0.0, numerator) / iteration)
    return float(statistics.median(shares)) if shares else None


def _rank_values(window: StepTimeWindow, metric: str) -> dict[int, float]:
    return {
        facts.global_rank: float(_safe_non_negative_float(value) or 0.0)
        for facts in window.rank_facts
        if (value := facts.average.value(metric)) is not None
    }


def _top_ranks(values: Mapping[int, float]) -> list[dict[str, Any]]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    if not ordered:
        return []
    median_value = sorted(value for _rank, value in ordered)[len(ordered) // 2]
    return [
        {
            "rank": int(rank),
            "value_ms": max(0.0, float(value)),
            "excess_vs_median_ms": max(0.0, float(value) - median_value),
            "pct_vs_median": (
                max(0.0, float(value) - median_value) / median_value
                if median_value > 0.0
                else None
            ),
        }
        for rank, value in ordered[:3]
    ]


def _typed_share(window: StepTimeWindow, metric: str) -> Optional[float]:
    prepared = {
        "input_wait": window.input_wait_share,
        "h2d": window.h2d_share,
        "compute": window.compute_share,
        "residual_proxy": window.residual_share,
    }
    if metric in prepared:
        return prepared[metric]
    shares = []
    for facts in window.rank_facts:
        numerator = facts.average.value(metric)
        input_wait = facts.average.input_wait_ms
        step_time = facts.average.step_time_ms
        if numerator is None or input_wait is None or step_time is None:
            continue
        iteration = input_wait + step_time
        if iteration > 0.0:
            shares.append(float(numerator) / iteration)
    return float(statistics.median(shares)) if shares else None


def build_metric_attribution(window: StepTimeWindow) -> dict[str, Any]:
    """Reproduce the opt-in attribution shape of the released API."""
    by_key = {metric.metric: metric for metric in window.metrics}
    single_rank = (
        window.coverage.world_size <= 1 or window.coverage.ranks_present <= 1
    )
    phases = {
        "input_wait": "input",
        "h2d": "h2d",
        "forward": "forward",
        "backward": "backward",
        "optimizer_step": "optimizer",
        "residual_proxy": "residual",
        "step_time": "step",
    }
    attribution: dict[str, Any] = {}
    for key, phase in phases.items():
        metric = by_key.get(key)
        values = _rank_values(window, key)
        attribution[key] = {
            "metric": key,
            "phase": phase,
            "median_total_ms": (
                float(_safe_non_negative_float(metric.median_total) or 0.0)
                if metric
                else 0.0
            ),
            "worst_total_ms": (
                float(_safe_non_negative_float(metric.worst_total) or 0.0)
                if metric
                else 0.0
            ),
            "worst_rank": metric.worst_rank if metric else None,
            "skew_pct": (
                _safe_non_negative_float(metric.skew_pct)
                if metric is not None and not single_rank
                else None
            ),
            "share_pct": _typed_share(window, key),
            "top_ranks": _top_ranks(values),
        }

    compute = _rank_values(window, "compute")
    median = float(statistics.median(compute.values())) if compute else 0.0
    worst_rank = (
        max(compute, key=lambda rank: (compute[rank], -rank))
        if compute
        else None
    )
    worst = float(compute[worst_rank]) if worst_rank is not None else 0.0
    skew = None
    if len(compute) >= 2:
        if median > 0.0:
            skew = (worst - median) / median
        elif worst <= 0.0:
            skew = 0.0
    attribution["compute"] = {
        "metric": "compute",
        "phase": "compute",
        "median_total_ms": median,
        "worst_total_ms": worst,
        "worst_rank": worst_rank,
        "skew_pct": max(0.0, skew) if skew is not None else None,
        "share_pct": window.compute_share,
        "top_ranks": _top_ranks(compute),
    }
    return attribution


def _window_from_rank_averages(
    metrics: Sequence[StepTimeMetric],
    *,
    per_rank_timing: Optional[Mapping[int, Mapping[str, float]]],
    expected_ranks: Optional[Sequence[int]],
    diagnosis_clock: str,
    training_strategy: str,
) -> StepTimeWindow:
    """Build the one typed window required by the canonical diagnoser."""
    timing = per_rank_timing or {}
    rank_facts = tuple(
        StepTimeRankFacts(
            global_rank=int(rank),
            average=_values_from_mapping(values),
        )
        for rank, values in sorted(timing.items())
    )
    step_metric = next(
        (metric for metric in metrics if metric.metric == "step_time"),
        None,
    )
    measured = {rank for metric in metrics for rank in metric.measured_ranks}
    expected = tuple(
        sorted(
            int(rank)
            for rank in (
                expected_ranks if expected_ranks is not None else timing
            )
        )
    )
    series_steps = (
        list(step_metric.series.steps)
        if step_metric is not None and step_metric.series is not None
        else []
    )
    population = len(expected) if expected else len(measured)
    ranks_present = len(rank_facts) if rank_facts else len(measured)

    def carrying(*names: str) -> tuple[int, ...]:
        return tuple(
            facts.global_rank
            for facts in rank_facts
            if all(facts.average.value(name) is not None for name in names)
        )

    strategy = str(training_strategy or "ddp").strip().lower()
    totals = {
        facts.global_rank: facts.average.total_step_ms
        for facts in rank_facts
        if facts.average.total_step_ms is not None
    }
    median_total = (
        float(statistics.median(totals.values())) if totals else None
    )
    representative = (
        min(
            totals,
            key=lambda rank: (
                abs(float(totals[rank]) - float(median_total)),
                float(totals[rank]),
                rank,
            ),
        )
        if totals and median_total is not None
        else None
    )
    worst = (
        max(totals, key=lambda rank: (float(totals[rank]), -rank))
        if totals
        else None
    )
    composition_names = tuple(
        name
        for name in (
            "input_wait",
            "forward",
            "backward",
            "optimizer_step",
            "residual_proxy",
        )
        if any(facts.average.value(name) is not None for facts in rank_facts)
    )
    return StepTimeWindow(
        clock="gpu" if str(diagnosis_clock).lower() == "gpu" else "cpu",
        training_strategy=strategy,
        steps=series_steps,
        expected_ranks=expected,
        coverage=StepTimeCoverage(
            expected_steps=step_metric.window_size if step_metric else 0,
            steps_used=step_metric.steps_used if step_metric else 0,
            completed_step=(
                int(series_steps[-1])
                if series_steps
                else (step_metric.steps_used if step_metric else 0)
            ),
            world_size=population,
            ranks_present=ranks_present,
            incomplete=ranks_present < population,
        ),
        rank_facts=rank_facts,
        metrics=list(metrics),
        iteration_ranks=carrying("input_wait", "step_time"),
        compute_ranks=carrying(
            "input_wait",
            "step_time",
            "forward",
            "backward",
            "optimizer_step",
        ),
        composition_ranks=carrying("step_time", *composition_names),
        straggler_ranks=tuple(
            facts.global_rank
            for facts in rank_facts
            if facts.average.input_wait_ms is not None
            and (facts.average.step_time_ms or 0.0) > 0.0
            and (facts.average.backward_ms or 0.0) > 0.0
            and (strategy != "fsdp" or (facts.average.forward_ms or 0.0) > 0.0)
        ),
        median_total_step_ms=median_total,
        representative_rank=representative,
        representative_total_step_ms=(
            float(totals[representative])
            if representative is not None
            else None
        ),
        worst_rank=worst,
        worst_total_step_ms=(
            float(totals[worst]) if worst is not None else None
        ),
        input_wait_share=_component_share(timing, "input_wait"),
        h2d_share=_component_share(timing, "h2d"),
        compute_share=_component_share(timing, "compute"),
        residual_share=_component_share(timing, "residual_proxy"),
    )


def _diagnose(
    metrics: Sequence[StepTimeMetric],
    thresholds: DiagnosisThresholds,
    *,
    per_rank_timing: Optional[Mapping[int, Mapping[str, float]]],
    expected_ranks: Optional[Sequence[int]],
    diagnosis_clock: str,
    training_strategy: str,
) -> "DiagnosticResult[StepDiagnosis]":
    from .api import diagnose_step_time_window
    from .policy import StepTimeDiagnosisPolicy

    window = _window_from_rank_averages(
        metrics,
        per_rank_timing=per_rank_timing,
        expected_ranks=expected_ranks,
        diagnosis_clock=diagnosis_clock,
        training_strategy=training_strategy,
    )
    return diagnose_step_time_window(
        window,
        policy=StepTimeDiagnosisPolicy(
            name="deprecated-rank-map",
            thresholds=thresholds,
        ),
        training_strategy=training_strategy,
        include_attribution=True,
    )


def build_step_diagnosis_result(
    metrics: Sequence[StepTimeMetric],
    thresholds: DiagnosisThresholds = DEFAULT_THRESHOLDS,
    *,
    per_rank_timing: Optional[Mapping[int, Mapping[str, float]]] = None,
    expected_ranks: Optional[Sequence[int]] = None,
    diagnosis_clock: str = "cpu",
    training_strategy: str = "ddp",
) -> "DiagnosticResult[StepDiagnosis]":
    """Run the deprecated rank-mapping API through canonical diagnosis."""
    _warn("build_step_diagnosis_result")
    return _diagnose(
        metrics,
        thresholds,
        per_rank_timing=per_rank_timing,
        expected_ranks=expected_ranks,
        diagnosis_clock=diagnosis_clock,
        training_strategy=training_strategy,
    )


def build_step_diagnosis(
    metrics: Sequence[StepTimeMetric],
    thresholds: DiagnosisThresholds = DEFAULT_THRESHOLDS,
    *,
    per_rank_timing: Optional[Mapping[int, Mapping[str, float]]] = None,
    expected_ranks: Optional[Sequence[int]] = None,
    diagnosis_clock: str = "cpu",
    training_strategy: str = "ddp",
) -> "StepDiagnosis":
    """Return the primary diagnosis from the deprecated rank-map input."""
    _warn("build_step_diagnosis")
    primary = _diagnose(
        metrics,
        thresholds,
        per_rank_timing=per_rank_timing,
        expected_ranks=expected_ranks,
        diagnosis_clock=diagnosis_clock,
        training_strategy=training_strategy,
    ).primary
    from .api import StepDiagnosis

    if not isinstance(primary, StepDiagnosis):
        raise TypeError(
            "build_step_diagnosis_result() must return StepDiagnosis "
            "as primary"
        )
    return primary


__all__ = [
    "build_metric_attribution",
    "build_step_diagnosis",
    "build_step_diagnosis_result",
]
