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
    StepTimeRepositorySnapshot,
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
            latest_step_observed=(
                int(completed_step) if completed_step is not None else None
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


def worst_rank_by_total_step(
    per_rank_timing: Mapping[int, Mapping[str, float]],
) -> Optional[int]:
    """Return the slowest rank carrying a selected-clock total step."""
    candidates: dict[int, float] = {}
    for rank, values in per_rank_timing.items():
        if "total_step" not in values:
            continue
        value = _safe_non_negative_float(values["total_step"])
        if value is not None:
            candidates[int(rank)] = value
    if not candidates:
        return None
    return max(candidates, key=lambda rank: (candidates[rank], -rank))


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
    "diagnose_step_time_window",
    "median_iteration_component_share",
    "public_step_time_metric_values",
    "worst_rank_by_total_step",
]
