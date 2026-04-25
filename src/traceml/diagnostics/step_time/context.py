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

from traceml.renderers.step_time.schema import StepCombinedTimeMetric

if TYPE_CHECKING:
    from .api import DiagnosisThresholds


@dataclass(frozen=True)
class ComputeSignal:
    """
    Dominant compute-phase signal used for attribution and bound classification.
    """

    label: str
    share: float
    skew: float
    worst_rank: Optional[int]


@dataclass(frozen=True)
class StepTimeAnalysisContext:
    """
    Normalized step-time analysis state shared by all step-time diagnosis rules.
    """

    metrics: Sequence[StepCombinedTimeMetric]
    thresholds: "DiagnosisThresholds"
    single_rank: bool
    steps_used: int
    overall_worst_rank: Optional[int]

    step_metric: StepCombinedTimeMetric
    dataloader_metric: Optional[StepCombinedTimeMetric]
    wait_metric: Optional[StepCombinedTimeMetric]
    forward_metric: Optional[StepCombinedTimeMetric]
    backward_metric: Optional[StepCombinedTimeMetric]
    optimizer_metric: Optional[StepCombinedTimeMetric]

    step_total: float
    dataloader_total: float
    wait_total: float
    compute_total: float

    dataloader_share: float
    wait_share: float
    compute_share: float

    dataloader_skew: float
    compute_skew: float
    dataloader_worst_rank: Optional[int]
    compute_worst_rank: Optional[int]

    dominant_compute: Optional[ComputeSignal]
    largest_compute: Optional[ComputeSignal]

    input_straggler_score: float
    compute_straggler_score: float

    rank_values: Dict[str, Dict[int, float]]
    per_rank_timing: Dict[int, Dict[str, float]]


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


def compute_median_total(
    *,
    forward: Optional[StepCombinedTimeMetric],
    backward: Optional[StepCombinedTimeMetric],
    optimizer: Optional[StepCombinedTimeMetric],
) -> float:
    """
    Return median-rank compute total.
    """
    return (
        metric_median_total(forward)
        + metric_median_total(backward)
        + metric_median_total(optimizer)
    )


def compute_worst_total(
    *,
    forward: Optional[StepCombinedTimeMetric],
    backward: Optional[StepCombinedTimeMetric],
    optimizer: Optional[StepCombinedTimeMetric],
) -> float:
    """
    Return worst-rank compute total.
    """
    return (
        metric_worst_total(forward)
        + metric_worst_total(backward)
        + metric_worst_total(optimizer)
    )


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


def typical_local_burden(
    *,
    dataloader: Optional[StepCombinedTimeMetric],
    forward: Optional[StepCombinedTimeMetric],
    backward: Optional[StepCombinedTimeMetric],
    optimizer: Optional[StepCombinedTimeMetric],
) -> float:
    """
    Return the typical local burden used to normalize straggler scores.
    """
    return metric_median_total(dataloader) + compute_median_total(
        forward=forward,
        backward=backward,
        optimizer=optimizer,
    )


def input_straggler_score(
    *,
    dataloader: Optional[StepCombinedTimeMetric],
    forward: Optional[StepCombinedTimeMetric],
    backward: Optional[StepCombinedTimeMetric],
    optimizer: Optional[StepCombinedTimeMetric],
) -> float:
    """
    Return the normalized input straggler score.
    """
    typical = typical_local_burden(
        dataloader=dataloader,
        forward=forward,
        backward=backward,
        optimizer=optimizer,
    )
    if typical <= 0.0:
        return 0.0

    excess = max(
        0.0,
        metric_worst_total(dataloader) - metric_median_total(dataloader),
    )
    return excess / typical


def compute_straggler_score(
    *,
    dataloader: Optional[StepCombinedTimeMetric],
    forward: Optional[StepCombinedTimeMetric],
    backward: Optional[StepCombinedTimeMetric],
    optimizer: Optional[StepCombinedTimeMetric],
) -> float:
    """
    Return the normalized compute straggler score.
    """
    typical = typical_local_burden(
        dataloader=dataloader,
        forward=forward,
        backward=backward,
        optimizer=optimizer,
    )
    if typical <= 0.0:
        return 0.0

    excess = max(
        0.0,
        compute_worst_total(
            forward=forward,
            backward=backward,
            optimizer=optimizer,
        )
        - compute_median_total(
            forward=forward,
            backward=backward,
            optimizer=optimizer,
        ),
    )
    return excess / typical


def dominant_compute_signal(
    *,
    forward: Optional[StepCombinedTimeMetric],
    backward: Optional[StepCombinedTimeMetric],
    optimizer: Optional[StepCombinedTimeMetric],
    step_total: float,
    single_rank: bool,
) -> Optional[ComputeSignal]:
    """
    Pick the compute component with strongest skew, then strongest share.
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
                worst_rank=metric_worst_rank(metric),
            )
        )

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.skew, item.share))


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
) -> StepTimeAnalysisContext:
    """
    Build one normalized context shared by all step-time diagnosis rules.
    """
    by_key = {metric.metric: metric for metric in metrics}

    step_metric = by_key["step_time"]
    dataloader_metric = by_key.get("dataloader_fetch")
    wait_metric = by_key.get("wait_proxy")
    forward_metric = by_key.get("forward")
    backward_metric = by_key.get("backward")
    optimizer_metric = by_key.get("optimizer_step")

    coverage = step_metric.coverage
    single_rank = (coverage.world_size <= 1) or (coverage.ranks_present <= 1)
    steps_used = int(step_metric.summary.steps_used)
    overall_worst_rank = metric_worst_rank(step_metric)

    step_total = metric_total(step_metric, single_rank=single_rank)
    dataloader_total = metric_total(dataloader_metric, single_rank=single_rank)
    wait_total = metric_total(wait_metric, single_rank=single_rank)
    compute_total_value = compute_total(
        forward=forward_metric,
        backward=backward_metric,
        optimizer=optimizer_metric,
        single_rank=single_rank,
    )

    dominant_compute = dominant_compute_signal(
        forward=forward_metric,
        backward=backward_metric,
        optimizer=optimizer_metric,
        step_total=step_total,
        single_rank=single_rank,
    )
    largest_compute = largest_compute_phase(
        forward=forward_metric,
        backward=backward_metric,
        optimizer=optimizer_metric,
        step_total=step_total,
        single_rank=single_rank,
    )

    compute_skew_value = (
        dominant_compute.skew if dominant_compute is not None else 0.0
    )
    compute_rank = (
        dominant_compute.worst_rank
        if dominant_compute is not None
        else overall_worst_rank
    )

    rank_values = {
        "dataloader_fetch": rank_values_from_metric(dataloader_metric),
        "forward": rank_values_from_metric(forward_metric),
        "backward": rank_values_from_metric(backward_metric),
        "optimizer_step": rank_values_from_metric(optimizer_metric),
        "step_time": rank_values_from_metric(step_metric),
        "wait_proxy": rank_values_from_metric(wait_metric),
    }

    local_per_rank_timing = {
        int(rank): {str(k): non_negative_finite(v) for k, v in values.items()}
        for rank, values in (per_rank_timing or {}).items()
    }
    if local_per_rank_timing:
        rank_values = {
            "dataloader_fetch": {
                rank: non_negative_finite(values.get("dataloader_fetch", 0.0))
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
            "wait_proxy": {
                rank: non_negative_finite(values.get("wait_proxy", 0.0))
                for rank, values in local_per_rank_timing.items()
            },
        }

    return StepTimeAnalysisContext(
        metrics=metrics,
        thresholds=thresholds,
        single_rank=single_rank,
        steps_used=steps_used,
        overall_worst_rank=overall_worst_rank,
        step_metric=step_metric,
        dataloader_metric=dataloader_metric,
        wait_metric=wait_metric,
        forward_metric=forward_metric,
        backward_metric=backward_metric,
        optimizer_metric=optimizer_metric,
        step_total=step_total,
        dataloader_total=dataloader_total,
        wait_total=wait_total,
        compute_total=compute_total_value,
        dataloader_share=share(dataloader_total, step_total),
        wait_share=share(wait_total, step_total),
        compute_share=share(compute_total_value, step_total),
        dataloader_skew=metric_skew(
            dataloader_metric, single_rank=single_rank
        ),
        compute_skew=compute_skew_value,
        dataloader_worst_rank=metric_worst_rank(dataloader_metric),
        compute_worst_rank=compute_rank,
        dominant_compute=dominant_compute,
        largest_compute=largest_compute,
        input_straggler_score=input_straggler_score(
            dataloader=dataloader_metric,
            forward=forward_metric,
            backward=backward_metric,
            optimizer=optimizer_metric,
        ),
        compute_straggler_score=compute_straggler_score(
            dataloader=dataloader_metric,
            forward=forward_metric,
            backward=backward_metric,
            optimizer=optimizer_metric,
        ),
        rank_values=rank_values,
        per_rank_timing=local_per_rank_timing,
    )


__all__ = [
    "ComputeSignal",
    "StepTimeAnalysisContext",
    "build_step_time_context",
    "compute_median_total",
    "compute_total",
    "compute_worst_total",
    "dominant_compute_signal",
    "largest_compute_phase",
    "metric_median_total",
    "metric_skew",
    "metric_total",
    "metric_worst_rank",
    "metric_worst_total",
    "non_negative_finite",
    "rank_values_from_metric",
    "share",
]
