"""
Prepared analysis context for step-time diagnostics.

This module centralizes all metric normalization and derived timing signals so
individual diagnosis rules can stay small and focused. The intent is:

- build one context from one analyzed window
- let multiple rules evaluate that same context
- avoid re-computing totals, shares, skew, and rank attribution in each rule
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Sequence

from traceml_ai.renderers.step_time.schema import StepCombinedTimeMetric
from traceml_ai.utils.training_strategy import normalize_training_strategy

if TYPE_CHECKING:
    from .policy import DiagnosisThresholds


@dataclass(frozen=True)
class ComputeSignal:
    """
    Dominant compute-phase signal used for attribution and bound
    classification.
    """

    label: str
    share: float
    skew: float
    median_ms: float
    worst_ms: float
    excess_ms: float
    worst_rank: Optional[int]


@dataclass(frozen=True)
class _CleanStragglerEvidence:
    """
    Rank-local straggler evidence after discounting backward delay that can be
    explained by non-backward rank skew.
    """

    kind: str
    status: str
    component: str
    metric: str
    phase: str
    score: float
    worst_rank: Optional[int]
    clean_step_median_ms: float
    clean_step_worst_ms: float
    clean_step_slack_ms: float
    typical_step_ms: float
    top_excess_ms: float
    second_excess_ms: float
    component_excesses_ms: Dict[str, float]


@dataclass(frozen=True)
class StepTimeAnalysisContext:
    """
    Normalized step-time analysis state shared by all step-time diagnosis
    rules.

    `training_strategy` is advisory run context from runtime metadata. It is
    not a public metric; rules may use it to choose attribution logic.
    """

    thresholds: "DiagnosisThresholds"
    single_rank: bool
    steps_used: int
    overall_worst_rank: Optional[int]
    training_strategy: str

    step_metric: StepCombinedTimeMetric
    input_wait_metric: Optional[StepCombinedTimeMetric]
    h2d_metric: Optional[StepCombinedTimeMetric]
    residual_metric: Optional[StepCombinedTimeMetric]
    forward_metric: Optional[StepCombinedTimeMetric]
    backward_metric: Optional[StepCombinedTimeMetric]
    optimizer_metric: Optional[StepCombinedTimeMetric]

    step_total: float
    residual_total: float
    compute_total: float

    residual_share: float
    compute_share: float
    input_bound_share: float

    input_bound_skew: float
    compute_skew: float
    input_bound_worst_rank: Optional[int]
    diagnosis_clock: str
    input_wait_total: float
    input_bound_step_total: float
    iteration_time_total: float

    largest_compute: Optional[ComputeSignal]

    rank_values: Dict[str, Dict[int, float]]
    clean_rank_values: Dict[str, Dict[int, float]]
    clean_straggler: Optional[_CleanStragglerEvidence]


def non_negative_finite(value: float) -> float:
    """
    Return a safe non-negative finite float.
    """
    try:
        out = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(0.0, out)


def share(value: float, total: float) -> float:
    """
    Return a safe non-negative share in `[0, 1]`.
    """
    total_safe = non_negative_finite(total)
    if total_safe <= 0.0:
        return 0.0
    return max(0.0, non_negative_finite(value) / total_safe)


def _median(values: Sequence[float]) -> float:
    """Return the median of safe non-negative finite values."""
    clean = sorted(non_negative_finite(value) for value in values)
    n = len(clean)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return float(clean[mid])
    return float((clean[mid - 1] + clean[mid]) / 2.0)


def _rank_stats(
    rank_values: Dict[int, float],
) -> tuple[float, float, Optional[int], float]:
    """Return median, worst value, worst rank, and worst-minus-median."""
    if not rank_values:
        return 0.0, 0.0, None, 0.0
    clean = {
        int(rank): non_negative_finite(value)
        for rank, value in rank_values.items()
    }
    median = _median(tuple(clean.values()))
    worst_rank = max(clean, key=lambda rank: (clean[rank], -int(rank)))
    worst = clean[worst_rank]
    return median, worst, int(worst_rank), max(0.0, worst - median)


def metric_median_total(metric: Optional[StepCombinedTimeMetric]) -> float:
    """
    Return the median-rank total for one metric.
    """
    if metric is None:
        return 0.0
    return non_negative_finite(metric.summary.median_total)


def metric_worst_total(metric: Optional[StepCombinedTimeMetric]) -> float:
    """
    Return the worst-rank total for one metric.
    """
    if metric is None:
        return 0.0
    return non_negative_finite(metric.summary.worst_total)


def metric_total(
    metric: Optional[StepCombinedTimeMetric],
    *,
    single_rank: bool,
) -> float:
    """
    Return the visible total used for diagnosis.

    - single-rank: use worst_total
    - multi-rank: use median_total
    """
    if metric is None:
        return 0.0
    raw = (
        metric.summary.worst_total
        if single_rank
        else metric.summary.median_total
    )
    return non_negative_finite(raw)


def metric_skew(
    metric: Optional[StepCombinedTimeMetric],
    *,
    single_rank: bool,
) -> float:
    """
    Return cross-rank skew for multi-rank runs, else 0.
    """
    if metric is None or single_rank:
        return 0.0
    return non_negative_finite(metric.summary.skew_pct)


def metric_worst_rank(
    metric: Optional[StepCombinedTimeMetric],
) -> Optional[int]:
    """
    Return the metric's worst rank, if available.
    """
    if metric is None or metric.summary.worst_rank is None:
        return None
    try:
        return int(metric.summary.worst_rank)
    except Exception:
        return None


def compute_total(
    *,
    forward: Optional[StepCombinedTimeMetric],
    backward: Optional[StepCombinedTimeMetric],
    optimizer: Optional[StepCombinedTimeMetric],
    single_rank: bool,
) -> float:
    """
    Return typical compute total for the step.
    """
    return (
        metric_total(forward, single_rank=single_rank)
        + metric_total(backward, single_rank=single_rank)
        + metric_total(optimizer, single_rank=single_rank)
    )


def metric_excess(metric: Optional[StepCombinedTimeMetric]) -> float:
    """
    Return worst-vs-median excess for one timing metric.
    """
    return max(0.0, metric_worst_total(metric) - metric_median_total(metric))


def _clean_rank_timings(
    per_rank_timing: Dict[int, Dict[str, float]],
) -> Dict[str, Dict[int, float]]:
    """
    Return clean rank-local timings used for straggler diagnosis.

    The policy discounts backward time that can be explained by another rank's
    non-backward work in the same averaged/aligned window:

        residual_r = current residual_proxy_r
        non_bwd_r = INPUT_WAIT_r + H2D_r + FWD_r + OPT_r + RESIDUAL_r
        clean_bwd_r = max(0, BWD_r - max(0, max(non_bwd) - non_bwd_r))
        clean_compute_r = FWD_r + clean_bwd_r + OPT_r
        clean_step_r = INPUT_WAIT_r + H2D_r + clean_compute_r + RESIDUAL_r
        score = (max(clean_step) - median(clean_step)) / median(actual_step)
    """
    if not per_rank_timing:
        return {}

    ranks = sorted(int(rank) for rank in per_rank_timing)
    non_bwd = {
        rank: (
            non_negative_finite(per_rank_timing[rank].get("input_wait", 0.0))
            + non_negative_finite(per_rank_timing[rank].get("h2d", 0.0))
            + non_negative_finite(per_rank_timing[rank].get("forward", 0.0))
            + non_negative_finite(
                per_rank_timing[rank].get("optimizer_step", 0.0)
            )
            + non_negative_finite(
                per_rank_timing[rank].get("residual_proxy", 0.0)
            )
        )
        for rank in ranks
    }
    non_bwd_max = max(non_bwd.values()) if non_bwd else 0.0

    out = {
        "input_wait": {},
        "h2d": {},
        "forward": {},
        "backward": {},
        "optimizer_step": {},
        "residual_proxy": {},
        "total_step": {},
        "non_backward": {},
        "clean_backward": {},
        "clean_compute": {},
        "clean_step": {},
    }
    for rank in ranks:
        values = per_rank_timing[rank]
        input_wait = non_negative_finite(values.get("input_wait", 0.0))
        h2d = non_negative_finite(values.get("h2d", 0.0))
        forward = non_negative_finite(values.get("forward", 0.0))
        backward = non_negative_finite(values.get("backward", 0.0))
        optimizer = non_negative_finite(values.get("optimizer_step", 0.0))
        residual = non_negative_finite(values.get("residual_proxy", 0.0))
        total_step = non_negative_finite(values.get("total_step", 0.0))

        explained_bwd_delay = max(0.0, non_bwd_max - non_bwd[rank])
        clean_backward = max(0.0, backward - explained_bwd_delay)
        clean_compute = forward + clean_backward + optimizer
        clean_step = input_wait + h2d + clean_compute + residual

        out["input_wait"][rank] = input_wait
        out["h2d"][rank] = h2d
        out["forward"][rank] = forward
        out["backward"][rank] = backward
        out["optimizer_step"][rank] = optimizer
        out["residual_proxy"][rank] = residual
        out["total_step"][rank] = total_step
        out["non_backward"][rank] = non_bwd[rank]
        out["clean_backward"][rank] = clean_backward
        out["clean_compute"][rank] = clean_compute
        out["clean_step"][rank] = clean_step

    return out


def _clean_straggler_evidence(
    *,
    clean_rank_values: Dict[str, Dict[int, float]],
    score_threshold: float,
    dominance_tolerance: float,
) -> Optional[_CleanStragglerEvidence]:
    """Return rank-local clean-step straggler evidence, if material."""
    clean_steps = clean_rank_values.get("clean_step", {})
    actual_steps = clean_rank_values.get("total_step", {})
    if len(clean_steps) <= 1 or not actual_steps:
        return None

    median_clean, worst_clean, worst_rank, clean_slack = _rank_stats(
        clean_steps
    )
    typical_step = _median(tuple(actual_steps.values()))
    if typical_step <= 0.0:
        return None

    score = clean_slack / typical_step
    if score < non_negative_finite(score_threshold):
        return None
    if worst_rank is None:
        return None

    components = {
        "input": (
            "INPUT_STRAGGLER",
            "INPUT STRAGGLER",
            "input_wait",
            "input",
        ),
        "compute": (
            "COMPUTE_STRAGGLER",
            "COMPUTE STRAGGLER",
            "compute",
            "compute",
        ),
        "h2d": ("H2D_STRAGGLER", "H2D STRAGGLER", "h2d", "h2d"),
        "residual": (
            "RESIDUAL_STRAGGLER",
            "RESIDUAL STRAGGLER",
            "residual_proxy",
            "residual",
        ),
    }
    source_by_component = {
        "input": clean_rank_values.get("input_wait", {}),
        "compute": clean_rank_values.get("clean_compute", {}),
        "h2d": clean_rank_values.get("h2d", {}),
        "residual": clean_rank_values.get("residual_proxy", {}),
    }
    component_excesses = {
        name: max(
            0.0,
            non_negative_finite(values.get(worst_rank, 0.0))
            - _median(tuple(values.values())),
        )
        for name, values in source_by_component.items()
    }
    ordered = sorted(
        component_excesses.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    top_component, top_excess = ordered[0]
    second_excess = ordered[1][1] if len(ordered) > 1 else 0.0

    tolerance = max(1.0, non_negative_finite(dominance_tolerance))
    if top_excess <= 0.0 or top_excess < tolerance * second_excess:
        kind, status, metric, phase = (
            "STRAGGLER",
            "STRAGGLER",
            "step_time",
            "mixed",
        )
        component = "mixed"
    else:
        kind, status, metric, phase = components[top_component]
        component = top_component

    return _CleanStragglerEvidence(
        kind=kind,
        status=status,
        component=component,
        metric=metric,
        phase=phase,
        score=score,
        worst_rank=worst_rank,
        clean_step_median_ms=median_clean,
        clean_step_worst_ms=worst_clean,
        clean_step_slack_ms=clean_slack,
        typical_step_ms=typical_step,
        top_excess_ms=top_excess,
        second_excess_ms=second_excess,
        component_excesses_ms=component_excesses,
    )


def largest_compute_phase(
    *,
    forward: Optional[StepCombinedTimeMetric],
    backward: Optional[StepCombinedTimeMetric],
    optimizer: Optional[StepCombinedTimeMetric],
    step_total: float,
    single_rank: bool,
) -> Optional[ComputeSignal]:
    """
    Pick the compute component with the largest typical share.
    """
    candidates: list[ComputeSignal] = []

    for label, metric in (
        ("Forward", forward),
        ("Backward", backward),
        ("Optimizer", optimizer),
    ):
        if metric is None:
            continue

        total = metric_total(metric, single_rank=single_rank)
        if total <= 0.0:
            continue

        candidates.append(
            ComputeSignal(
                label=label,
                share=share(total, step_total),
                skew=metric_skew(metric, single_rank=single_rank),
                median_ms=metric_median_total(metric),
                worst_ms=metric_worst_total(metric),
                excess_ms=metric_excess(metric),
                worst_rank=metric_worst_rank(metric),
            )
        )

    if not candidates:
        return None
    return max(candidates, key=lambda item: item.share)


def rank_values_from_metric(
    metric: Optional[StepCombinedTimeMetric],
) -> Dict[int, float]:
    """
    Best-effort rank -> value extraction for one metric.

    Live renderer metrics do not carry the full per-rank map, so this falls
    back to a single worst-rank entry. Summary-mode callers can provide richer
    `per_rank_timing` and that information will be used instead.
    """
    if metric is None:
        return {}

    rank = metric_worst_rank(metric)
    if rank is None:
        return {}

    return {int(rank): metric_worst_total(metric)}


def build_step_time_context(
    *,
    metrics: Sequence[StepCombinedTimeMetric],
    thresholds: "DiagnosisThresholds",
    per_rank_timing: Optional[Dict[int, Dict[str, float]]] = None,
    diagnosis_clock: str = "cpu",
    training_strategy: str = "ddp",
) -> StepTimeAnalysisContext:
    """
    Build one normalized context shared by all step-time diagnosis rules.
    """
    by_key = {metric.metric: metric for metric in metrics}

    step_metric = by_key["step_time"]
    input_wait_metric = by_key.get("input_wait")
    h2d_metric = by_key.get("h2d")
    residual_metric = by_key.get("residual_proxy")
    forward_metric = by_key.get("forward")
    backward_metric = by_key.get("backward")
    optimizer_metric = by_key.get("optimizer_step")

    coverage = step_metric.coverage
    single_rank = (coverage.world_size <= 1) or (coverage.ranks_present <= 1)
    steps_used = int(step_metric.summary.steps_used)
    overall_worst_rank = metric_worst_rank(step_metric)

    step_total = metric_total(step_metric, single_rank=single_rank)
    residual_total = metric_total(residual_metric, single_rank=single_rank)
    compute_total_value = compute_total(
        forward=forward_metric,
        backward=backward_metric,
        optimizer=optimizer_metric,
        single_rank=single_rank,
    )
    largest_compute = largest_compute_phase(
        forward=forward_metric,
        backward=backward_metric,
        optimizer=optimizer_metric,
        step_total=step_total,
        single_rank=single_rank,
    )

    rank_values = {
        "input_wait": rank_values_from_metric(input_wait_metric),
        "h2d": rank_values_from_metric(h2d_metric),
        "forward": rank_values_from_metric(forward_metric),
        "backward": rank_values_from_metric(backward_metric),
        "optimizer_step": rank_values_from_metric(optimizer_metric),
        "step_time": rank_values_from_metric(step_metric),
        "residual_proxy": rank_values_from_metric(residual_metric),
    }

    local_per_rank_timing = {
        int(rank): {str(k): non_negative_finite(v) for k, v in values.items()}
        for rank, values in (per_rank_timing or {}).items()
    }
    if local_per_rank_timing:
        rank_values = {
            "input_wait": {
                rank: non_negative_finite(values.get("input_wait", 0.0))
                for rank, values in local_per_rank_timing.items()
            },
            "h2d": {
                rank: non_negative_finite(values.get("h2d", 0.0))
                for rank, values in local_per_rank_timing.items()
            },
            "forward": {
                rank: non_negative_finite(values.get("forward", 0.0))
                for rank, values in local_per_rank_timing.items()
            },
            "backward": {
                rank: non_negative_finite(values.get("backward", 0.0))
                for rank, values in local_per_rank_timing.items()
            },
            "optimizer_step": {
                rank: non_negative_finite(values.get("optimizer_step", 0.0))
                for rank, values in local_per_rank_timing.items()
            },
            "step_time": {
                rank: non_negative_finite(values.get("step_time", 0.0))
                for rank, values in local_per_rank_timing.items()
            },
            "residual_proxy": {
                rank: non_negative_finite(values.get("residual_proxy", 0.0))
                for rank, values in local_per_rank_timing.items()
            },
        }

    input_candidates = {
        rank: values
        for rank, values in local_per_rank_timing.items()
        if "input_wait" in values and "step_time" in values
    }

    input_wait_rank_values = {
        rank: non_negative_finite(values.get("input_wait", 0.0))
        for rank, values in input_candidates.items()
    }
    input_step_rank_values = {
        rank: non_negative_finite(values.get("step_time", 0.0))
        for rank, values in input_candidates.items()
    }
    input_wait_median, input_wait_worst, input_wait_worst_rank, input_slack = (
        _rank_stats(input_wait_rank_values)
    )
    input_step_median, input_step_worst, _, _ = _rank_stats(
        input_step_rank_values
    )
    input_wait_total = input_wait_worst if single_rank else input_wait_median
    input_bound_step_total = (
        input_step_worst if single_rank else input_step_median
    )
    iteration_time_total = input_wait_total + input_bound_step_total
    input_bound_skew = (
        0.0
        if single_rank or input_wait_median <= 0.0
        else input_slack / input_wait_median
    )
    clean_rank_values = _clean_rank_timings(local_per_rank_timing)
    clean_straggler = _clean_straggler_evidence(
        clean_rank_values=clean_rank_values,
        score_threshold=thresholds.straggler_score_warn,
        dominance_tolerance=thresholds.straggler_dominance_tolerance,
    )
    clean_compute_values = clean_rank_values.get("clean_compute", {})
    _compute_median, _, _, _compute_slack = _rank_stats(clean_compute_values)
    compute_skew_value = (
        (_compute_slack / _compute_median) if _compute_median > 0.0 else 0.0
    )

    return StepTimeAnalysisContext(
        thresholds=thresholds,
        single_rank=single_rank,
        steps_used=steps_used,
        overall_worst_rank=overall_worst_rank,
        training_strategy=normalize_training_strategy(training_strategy),
        step_metric=step_metric,
        input_wait_metric=input_wait_metric,
        h2d_metric=h2d_metric,
        residual_metric=residual_metric,
        forward_metric=forward_metric,
        backward_metric=backward_metric,
        optimizer_metric=optimizer_metric,
        step_total=step_total,
        residual_total=residual_total,
        compute_total=compute_total_value,
        residual_share=share(residual_total, step_total),
        compute_share=share(compute_total_value, step_total),
        input_bound_share=share(input_wait_total, iteration_time_total),
        input_bound_skew=input_bound_skew,
        compute_skew=compute_skew_value,
        input_bound_worst_rank=input_wait_worst_rank,
        diagnosis_clock=(
            "gpu" if str(diagnosis_clock).lower() == "gpu" else "cpu"
        ),
        input_wait_total=input_wait_total,
        input_bound_step_total=input_bound_step_total,
        iteration_time_total=iteration_time_total,
        largest_compute=largest_compute,
        rank_values=rank_values,
        clean_rank_values=clean_rank_values,
        clean_straggler=clean_straggler,
    )


__all__ = [
    "ComputeSignal",
    "StepTimeAnalysisContext",
    "build_step_time_context",
    "compute_total",
    "largest_compute_phase",
    "metric_median_total",
    "metric_excess",
    "metric_skew",
    "metric_total",
    "metric_worst_rank",
    "metric_worst_total",
    "non_negative_finite",
    "normalize_training_strategy",
    "rank_values_from_metric",
    "share",
]
