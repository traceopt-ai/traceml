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
class _RankStragglerPair:
    """
    Culprit/victim pair for visible distributed rank skew.

    The culprit is the rank with the smallest visible synchronization phase:
    it likely arrived late and therefore waited least. The victim is the upper
    actual median rank by that same visible phase, representing a real rank
    that paid typical waiting cost.
    """

    culprit_rank: int
    victim_rank: int
    culprit_value_ms: float
    victim_value_ms: float
    cost_ms: float


@dataclass(frozen=True)
class _RankStragglerEvidence:
    """
    Culprit-first rank straggler evidence.

    The score is visible wait cost paid by the victim rank, normalized by that
    victim's selected-clock iteration time. Component attribution compares the
    culprit's own input/H2D/forward values against the victim rank and names a
    cause only when it covers enough of the visible cost.
    """

    kind: str
    status: str
    component: str
    metric: str
    phase: str
    score: float
    culprit_rank: int
    victim_rank: int
    visible_metric: str
    visible_culprit_ms: float
    visible_victim_ms: float
    visible_cost_ms: float
    iteration_time_ms: float
    component_excesses_ms: Dict[str, float]
    component_coverage: Dict[str, float]


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
    h2d_share: float

    input_bound_skew: float
    compute_skew: float
    input_bound_worst_rank: Optional[int]
    diagnosis_clock: str
    input_wait_total: float
    input_bound_step_total: float
    iteration_time_total: float

    largest_compute: Optional[ComputeSignal]

    rank_values: Dict[str, Dict[int, float]]
    rank_straggler: Optional[_RankStragglerEvidence]


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


def _rank_straggler_pair(
    visible_values: Dict[int, float],
) -> Optional[_RankStragglerPair]:
    """
    Return the culprit and victim ranks for visible synchronization skew.

    The culprit is the rank with the smallest visible phase. The victim is the
    upper actual median rank by that same phase. Both are real observed ranks,
    which keeps two-rank and even-world-size comparisons easy to explain.
    """
    if len(visible_values) <= 1:
        return None

    clean = {
        int(rank): non_negative_finite(value)
        for rank, value in visible_values.items()
    }
    ordered = sorted(clean, key=lambda rank: (clean[rank], int(rank)))
    if len(ordered) <= 1:
        return None

    culprit_rank = int(ordered[0])
    victim_rank = int(ordered[len(ordered) // 2])
    culprit_value = clean[culprit_rank]
    victim_value = clean[victim_rank]
    return _RankStragglerPair(
        culprit_rank=culprit_rank,
        victim_rank=victim_rank,
        culprit_value_ms=culprit_value,
        victim_value_ms=victim_value,
        cost_ms=max(0.0, victim_value - culprit_value),
    )


def _rank_metric_value(
    per_rank_timing: Dict[int, Dict[str, float]],
    rank: int,
    metric: str,
) -> float:
    """Return a safe metric value for one rank."""
    return non_negative_finite(
        per_rank_timing.get(int(rank), {}).get(str(metric), 0.0)
    )


def _build_rank_straggler_evidence(
    *,
    per_rank_timing: Dict[int, Dict[str, float]],
    score_threshold: float,
    cause_coverage_threshold: float,
    training_strategy: str,
) -> Optional[_RankStragglerEvidence]:
    """
    Build culprit/victim rank straggler evidence for distributed runs.

    DDP/default uses backward as the visible synchronization phase. FSDP uses
    forward + backward because FSDP can communicate on both sides of module
    execution. Only ranks with measured visible-phase anchors and a measured
    step envelope are eligible. Component attribution compares the culprit rank
    directly to the victim rank and emits a subtype only when one component
    explains enough of the visible cost.
    """
    if len(per_rank_timing) <= 1:
        return None

    strategy = normalize_training_strategy(training_strategy)
    visible_metric = "forward_backward" if strategy == "fsdp" else "backward"
    eligible_ranks = []
    for rank in sorted(int(rank) for rank in per_rank_timing):
        forward = _rank_metric_value(per_rank_timing, rank, "forward")
        backward = _rank_metric_value(per_rank_timing, rank, "backward")
        step_time = _rank_metric_value(per_rank_timing, rank, "step_time")
        if step_time <= 0.0 or backward <= 0.0:
            continue
        if strategy == "fsdp" and forward <= 0.0:
            continue
        eligible_ranks.append(rank)
    if len(eligible_ranks) <= 1:
        return None

    visible_values = {
        int(rank): (
            _rank_metric_value(per_rank_timing, rank, "forward")
            + _rank_metric_value(per_rank_timing, rank, "backward")
            if strategy == "fsdp"
            else _rank_metric_value(per_rank_timing, rank, "backward")
        )
        for rank in eligible_ranks
    }
    pair = _rank_straggler_pair(visible_values)
    if pair is None:
        return None

    culprit = pair.culprit_rank
    victim = pair.victim_rank
    iteration_time = _rank_metric_value(
        per_rank_timing, victim, "input_wait"
    ) + _rank_metric_value(per_rank_timing, victim, "step_time")

    score = pair.cost_ms / iteration_time
    threshold = non_negative_finite(score_threshold)
    if score < threshold:
        return None

    input_excess = max(
        0.0,
        _rank_metric_value(per_rank_timing, culprit, "input_wait")
        - _rank_metric_value(per_rank_timing, victim, "input_wait"),
    )
    h2d_excess = max(
        0.0,
        _rank_metric_value(per_rank_timing, culprit, "h2d")
        - _rank_metric_value(per_rank_timing, victim, "h2d"),
    )
    if strategy == "fsdp":
        # FSDP interleaves collectives with forward/backward, so without
        # explicit collective timing this rule should not emit compute
        # stragglers from forward excess.
        forward_excess = 0.0
    else:
        culprit_forward = _rank_metric_value(
            per_rank_timing, culprit, "forward"
        )
        victim_forward = _rank_metric_value(per_rank_timing, victim, "forward")
        forward_excess = (
            max(0.0, culprit_forward - victim_forward)
            if culprit_forward > 0.0 and victim_forward > 0.0
            else 0.0
        )

    component_excesses = {
        "input": input_excess,
        "h2d": h2d_excess,
        "compute": forward_excess,
    }
    component_coverage = {
        component: min(1.0, excess / pair.cost_ms)
        for component, excess in component_excesses.items()
    }
    components = {
        "input": ("INPUT_STRAGGLER", "INPUT STRAGGLER", "input_wait", "input"),
        "h2d": ("H2D_STRAGGLER", "H2D STRAGGLER", "h2d", "h2d"),
        "compute": (
            "COMPUTE_STRAGGLER",
            "COMPUTE STRAGGLER",
            "forward",
            "forward",
        ),
    }
    ordered = sorted(
        component_coverage.items(),
        key=lambda item: (
            -item[1],
            ("input", "h2d", "compute").index(item[0]),
        ),
    )
    top_component, top_coverage = ordered[0]
    if top_coverage > 0.0 and top_coverage >= non_negative_finite(
        cause_coverage_threshold
    ):
        kind, status, metric, phase = components[top_component]
        component = top_component
    else:
        kind, status, metric, phase = (
            "STRAGGLER",
            "STRAGGLER",
            visible_metric,
            "sync",
        )
        component = "sync_or_unattributed"

    return _RankStragglerEvidence(
        kind=kind,
        status=status,
        component=component,
        metric=metric,
        phase=phase,
        score=score,
        culprit_rank=culprit,
        victim_rank=victim,
        visible_metric=visible_metric,
        visible_culprit_ms=pair.culprit_value_ms,
        visible_victim_ms=pair.victim_value_ms,
        visible_cost_ms=pair.cost_ms,
        iteration_time_ms=iteration_time,
        component_excesses_ms=component_excesses,
        component_coverage=component_coverage,
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


def compute_rank_values_from_components(
    rank_values: Dict[str, Dict[int, float]],
) -> Dict[int, float]:
    """
    Return rank -> forward + backward + optimizer_step from rank-value maps.
    """
    forward = rank_values.get("forward", {})
    backward = rank_values.get("backward", {})
    optimizer = rank_values.get("optimizer_step", {})
    out: Dict[int, float] = {}
    for rank in sorted(set(forward) | set(backward) | set(optimizer)):
        out[int(rank)] = (
            non_negative_finite(forward.get(rank, 0.0))
            + non_negative_finite(backward.get(rank, 0.0))
            + non_negative_finite(optimizer.get(rank, 0.0))
        )
    return out


def _median_iteration_component_share(
    per_rank_timing: Dict[int, Dict[str, float]],
    component: str,
) -> float:
    """Return the median selected-clock iteration share for one component."""
    shares = []
    for values in per_rank_timing.values():
        if not {
            "input_wait",
            "step_time",
            component,
        }.issubset(values):
            continue
        iteration = non_negative_finite(
            values["input_wait"]
        ) + non_negative_finite(values["step_time"])
        if iteration > 0.0:
            shares.append(
                share(non_negative_finite(values[component]), iteration)
            )
    return _median(shares)


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
    rank_straggler = _build_rank_straggler_evidence(
        per_rank_timing=local_per_rank_timing,
        score_threshold=thresholds.straggler_score_warn,
        cause_coverage_threshold=thresholds.straggler_cause_coverage_min,
        training_strategy=training_strategy,
    )
    compute_rank_values = compute_rank_values_from_components(rank_values)
    _compute_median, _, _, _compute_slack = _rank_stats(compute_rank_values)
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
        residual_share=_median_iteration_component_share(
            local_per_rank_timing,
            "residual_proxy",
        ),
        compute_share=share(compute_total_value, step_total),
        input_bound_share=_median_iteration_component_share(
            local_per_rank_timing,
            "input_wait",
        ),
        h2d_share=_median_iteration_component_share(
            local_per_rank_timing,
            "h2d",
        ),
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
        rank_straggler=rank_straggler,
    )


__all__ = [
    "ComputeSignal",
    "StepTimeAnalysisContext",
    "build_step_time_context",
    "compute_total",
    "compute_rank_values_from_components",
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
