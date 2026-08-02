"""Compatibility entry points for canonical Step Time analysis.

New production code should use
:class:`traceml_ai.step_time.analysis.StepTimeAnalyzer` with normalized
repository snapshots. This module keeps the historical raw fixture,
diagnosis, and rank-mapping helpers while consumers migrate in PR5 through
PR9; it does not own the canonical analysis algorithm.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence

import numpy as np

from traceml_ai.step_time.analysis import (
    DATALOADER_FETCH_KEY,
    DISPLAY_METRICS,
    INPUT_WAIT_KEY,
    SELECTED_METRICS,
    STEP_TIME_CPU_KEY,
    StepTimeAnalyzer,
)
from traceml_ai.step_time.model import (
    DIAGNOSIS_CLOCK_KEY,
    STEP_TIME_EVENT_NAMES,
    DiagnosisClock,
    StepTimeCoverage,
    StepTimeMetric,
    StepTimeRankFacts,
    StepTimeRepositorySnapshot,
    StepTimeSourceCursor,
    StepTimeSourceRow,
    StepTimeValues,
    StepTimeWindow,
)
from traceml_ai.step_time.sqlite import normalize_step_time_events

if TYPE_CHECKING:
    from traceml_ai.diagnostics.common import DiagnosticResult
    from traceml_ai.diagnostics.step_time.api import StepDiagnosis
    from traceml_ai.diagnostics.step_time.policy import StepTimeDiagnosisPolicy

DATALOADER_EVENT_NAME = STEP_TIME_EVENT_NAMES[INPUT_WAIT_KEY]
STEP_TIME_EVENT_NAME = STEP_TIME_EVENT_NAMES["step_time"]
EVENT_ALIASES: Dict[str, str] = dict(STEP_TIME_EVENT_NAMES)


def _safe_non_negative_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result):
        return None
    return max(0.0, result)


def build_step_time_window_from_events(
    per_rank_steps: Mapping[int, Mapping[int, Any]],
    *,
    max_rows: int,
    expected_ranks: Optional[Sequence[int]] = None,
    completed_step: Optional[int] = None,
    training_strategy: str = "ddp",
) -> StepTimeWindow:
    """Analyze historical raw-event fixtures through the canonical analyzer.

    TODO(PR9): Remove this compatibility input after external fixtures and
    callers construct normalized repository snapshots directly.
    """
    expected = tuple(
        sorted(
            {int(rank) for rank in (expected_ranks or per_rank_steps.keys())}
        )
    )
    rows: list[StepTimeSourceRow] = []
    source_id = 0
    for rank, step_map in per_rank_steps.items():
        for step, events in step_map.items():
            source_id += 1
            rows.append(
                StepTimeSourceRow(
                    source_id=source_id,
                    global_rank=int(rank),
                    step=int(step),
                    metrics=normalize_step_time_events(events) or {},
                )
            )

    window = StepTimeAnalyzer().analyze(
        StepTimeRepositorySnapshot(
            rows=tuple(rows),
            global_ranks=expected,
            training_strategy=str(training_strategy),
            cursor=StepTimeSourceCursor(
                latest_step=(
                    int(completed_step) if completed_step is not None else None
                )
            ),
        ),
        window_size=max_rows,
    )
    if completed_step is not None and not window.steps:
        window = replace(
            window,
            coverage=replace(
                window.coverage,
                completed_step=int(completed_step),
            ),
        )
    return window


def diagnose_step_time_window(
    window: StepTimeWindow,
    *,
    policy: "StepTimeDiagnosisPolicy",
    training_strategy: Optional[str] = None,
    include_attribution: bool = False,
) -> "DiagnosticResult[StepDiagnosis]":
    """Delegate the historical import path to canonical window diagnosis.

    Rich per-metric attribution is opt-in because no current runtime surface
    consumes it. The released mapping-based API still requests it explicitly.
    """
    from traceml_ai.diagnostics.step_time.api import (
        diagnose_step_time_window as diagnose,
    )

    return diagnose(
        window,
        policy=policy,
        training_strategy=training_strategy,
        include_attribution=include_attribution,
    )


def median_iteration_component_share(
    per_rank_timing: Mapping[int, Mapping[str, float]],
    component: str,
) -> Optional[float]:
    """Return a legacy mapping's median component/iteration share.

    Canonical windows already carry the four reusable shares. This adapter
    remains for diagnosis-rule callers until PR5.
    """
    shares: list[float] = []
    compute_keys = ("forward", "backward", "optimizer_step")
    for values in per_rank_timing.values():
        if INPUT_WAIT_KEY not in values or "step_time" not in values:
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

        input_wait = _safe_non_negative_float(values[INPUT_WAIT_KEY])
        step_time = _safe_non_negative_float(values["step_time"])
        if input_wait is None or step_time is None:
            continue
        iteration = input_wait + step_time
        if iteration > 0.0:
            shares.append(max(0.0, numerator) / iteration)

    if not shares:
        return None
    return float(np.median(np.asarray(shares, dtype=np.float64)))


def _values_from_legacy_mapping(
    values: Mapping[str, float],
) -> StepTimeValues:
    """Translate one released sparse rank row into canonical typed values."""

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
    total_step_cpu = (
        dataloader_cpu + step_time_cpu
        if dataloader_cpu is not None and step_time_cpu is not None
        else None
    )
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
        total_step_cpu_ms=total_step_cpu,
    )


def build_step_time_window_from_rank_averages(
    metrics: Sequence[StepTimeMetric],
    *,
    per_rank_timing: Optional[Mapping[int, Mapping[str, float]]] = None,
    expected_ranks: Optional[Sequence[int]] = None,
    diagnosis_clock: str = "cpu",
    training_strategy: str = "ddp",
) -> StepTimeWindow:
    """Adapt the released rank-mapping diagnosis input to typed facts.

    TODO(PR9): Remove after external callers migrate to window diagnosis.
    This is the only production conversion from legacy rank averages.
    """
    timing = per_rank_timing or {}
    rank_facts = tuple(
        StepTimeRankFacts(
            global_rank=int(rank),
            average=_values_from_legacy_mapping(values),
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
                expected_ranks if expected_ranks is not None else timing.keys()
            )
        )
    )
    ranks_present = len(rank_facts) if rank_facts else len(measured)
    series_steps = (
        list(step_metric.series.steps)
        if step_metric is not None and step_metric.series is not None
        else []
    )
    population_size = len(expected) if expected else len(measured)
    coverage = StepTimeCoverage(
        expected_steps=(step_metric.window_size if step_metric else 0),
        steps_used=(step_metric.steps_used if step_metric else 0),
        completed_step=(
            int(series_steps[-1])
            if series_steps
            else (step_metric.steps_used if step_metric else 0)
        ),
        world_size=population_size,
        ranks_present=ranks_present,
        incomplete=ranks_present < population_size,
    )

    def carrying(*names: str) -> tuple[int, ...]:
        return tuple(
            facts.global_rank
            for facts in rank_facts
            if all(facts.average.value(name) is not None for name in names)
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
    composition = carrying("step_time", *composition_names)
    strategy = str(training_strategy).strip().lower()
    straggler = tuple(
        facts.global_rank
        for facts in rank_facts
        if facts.average.input_wait_ms is not None
        and (facts.average.step_time_ms or 0.0) > 0.0
        and (facts.average.backward_ms or 0.0) > 0.0
        and (strategy != "fsdp" or (facts.average.forward_ms or 0.0) > 0.0)
    )
    totals = {
        facts.global_rank: facts.average.total_step_ms
        for facts in rank_facts
        if facts.average.total_step_ms is not None
    }
    median_total = (
        float(np.median(np.asarray(tuple(totals.values()))))
        if totals
        else None
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
    return StepTimeWindow(
        clock=("gpu" if str(diagnosis_clock).lower() == "gpu" else "cpu"),
        training_strategy=strategy,
        steps=series_steps,
        expected_ranks=expected,
        coverage=coverage,
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
        composition_ranks=composition,
        straggler_ranks=straggler,
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
        input_wait_share=median_iteration_component_share(
            timing,
            "input_wait",
        ),
        h2d_share=median_iteration_component_share(timing, "h2d"),
        compute_share=median_iteration_component_share(timing, "compute"),
        residual_share=median_iteration_component_share(
            timing,
            "residual_proxy",
        ),
    )


def public_step_time_metric_values(
    timing: Mapping[str, float] | StepTimeValues,
) -> Dict[str, Optional[float]]:
    """Map canonical timing facts to stable final-summary metric names.

    Selected-clock fields expose diagnosis timing. Dataloader and total-step
    compatibility fields use the retained CPU clock. Every public key is
    present; unavailable stays ``None`` and measured zero stays ``0.0``.
    """
    if isinstance(timing, StepTimeValues):
        return {
            "total_step_ms": timing.total_step_cpu_ms,
            "dataloader_ms": timing.dataloader_cpu_ms,
            "input_wait_ms": timing.input_wait_ms,
            "step_time_ms": timing.step_time_ms,
            "h2d_ms": timing.h2d_ms,
            "compute_ms": timing.compute_ms,
            "residual_ms": timing.residual_ms,
            "forward_ms": timing.forward_ms,
            "backward_ms": timing.backward_ms,
            "optimizer_ms": timing.optimizer_step_ms,
        }

    def value(key: str) -> Optional[float]:
        return float(timing[key]) if key in timing else None

    forward = value("forward")
    backward = value("backward")
    optimizer = value("optimizer_step")
    compute = (
        forward + backward + optimizer
        if None not in (forward, backward, optimizer)
        else None
    )
    dataloader = value(DATALOADER_FETCH_KEY)
    step_time_cpu = value(STEP_TIME_CPU_KEY)
    total_step = (
        dataloader + step_time_cpu
        if dataloader is not None and step_time_cpu is not None
        else None
    )
    residual_value = value("residual_proxy")
    return {
        "total_step_ms": total_step,
        "dataloader_ms": dataloader,
        "input_wait_ms": value(INPUT_WAIT_KEY),
        "step_time_ms": value("step_time"),
        "h2d_ms": value("h2d"),
        "compute_ms": compute,
        "residual_ms": (
            max(0.0, residual_value) if residual_value is not None else None
        ),
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
    "build_step_time_window_from_events",
    "build_step_time_window_from_rank_averages",
    "diagnose_step_time_window",
    "median_iteration_component_share",
    "public_step_time_metric_values",
]
