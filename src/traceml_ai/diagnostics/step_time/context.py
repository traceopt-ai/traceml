"""Prepared facts consumed by Step Time diagnosis rules.

The analyzer owns timing semantics and reusable statistics. This module adds
only diagnosis-specific interpretation, such as thresholded rank-straggler
evidence. It reads the typed :class:`StepTimeWindow` directly and never
reconstructs the historical rank/metric dictionary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Sequence

from traceml_ai.step_time.model import (
    StepTimeMetric,
    StepTimeRankFacts,
    StepTimeWindow,
)
from traceml_ai.utils.training_strategy import normalize_training_strategy

if TYPE_CHECKING:
    from .policy import DiagnosisThresholds


@dataclass(frozen=True)
class ComputeSignal:
    """Dominant compute phase for diagnosis attribution."""

    label: str
    share: float
    skew: Optional[float]
    median_ms: float
    worst_ms: float
    excess_ms: float
    worst_rank: Optional[int]


@dataclass(frozen=True)
class _RankStragglerPair:
    """Real culprit/victim ranks ordered by visible synchronization cost."""

    culprit_rank: int
    victim_rank: int
    culprit_value_ms: float
    victim_value_ms: float
    cost_ms: float


@dataclass(frozen=True)
class _RankStragglerEvidence:
    """Thresholded distributed-rank evidence shared by the rule set."""

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
    """Diagnosis-ready view over one canonical analyzed window."""

    thresholds: "DiagnosisThresholds"
    window: StepTimeWindow
    single_rank: bool
    steps_used: int
    overall_worst_rank: Optional[int]
    training_strategy: str

    step_metric: StepTimeMetric
    input_wait_metric: Optional[StepTimeMetric]
    h2d_metric: Optional[StepTimeMetric]
    residual_metric: Optional[StepTimeMetric]
    forward_metric: Optional[StepTimeMetric]
    backward_metric: Optional[StepTimeMetric]
    optimizer_metric: Optional[StepTimeMetric]

    step_total: float
    residual_total: float
    compute_total: float

    residual_share: Optional[float]
    compute_share: Optional[float]
    input_bound_share: Optional[float]
    h2d_share: Optional[float]

    input_bound_skew: Optional[float]
    compute_skew: Optional[float]
    input_bound_worst_rank: Optional[int]
    diagnosis_clock: str
    input_wait_total: float
    input_bound_step_total: float
    iteration_time_total: float

    largest_compute: Optional[ComputeSignal]
    rank_straggler: Optional[_RankStragglerEvidence]
    missing_signals: tuple[str, ...] = ()
    signal_coverage: tuple[tuple[str, str], ...] = ()


def non_negative_finite(value: float) -> float:
    """Return a finite non-negative float, or zero for unusable input."""
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(0.0, out)


def share(value: float, total: float) -> float:
    """Return a safe non-negative share."""
    total_safe = non_negative_finite(total)
    if total_safe <= 0.0:
        return 0.0
    return max(0.0, non_negative_finite(value) / total_safe)


def _median(values: Sequence[float]) -> float:
    clean = sorted(non_negative_finite(value) for value in values)
    if not clean:
        return 0.0
    middle = len(clean) // 2
    if len(clean) % 2:
        return float(clean[middle])
    return float((clean[middle - 1] + clean[middle]) / 2.0)


def _rank_stats(
    values: Sequence[tuple[int, float]],
) -> tuple[float, float, Optional[int], float]:
    """Return median, worst, worst rank, and worst-minus-median."""
    if not values:
        return 0.0, 0.0, None, 0.0
    clean = tuple(
        (int(rank), non_negative_finite(value)) for rank, value in values
    )
    median = _median(tuple(value for _rank, value in clean))
    worst_rank, worst = max(clean, key=lambda item: (item[1], -item[0]))
    return median, worst, worst_rank, max(0.0, worst - median)


def metric_median_total(metric: Optional[StepTimeMetric]) -> float:
    """Return the mathematical median for one metric."""
    if metric is None:
        return 0.0
    return non_negative_finite(metric.median_total)


def metric_worst_total(metric: Optional[StepTimeMetric]) -> float:
    """Return the largest measured-rank value for one metric."""
    if metric is None:
        return 0.0
    return non_negative_finite(metric.worst_total)


def metric_total(
    metric: Optional[StepTimeMetric],
    *,
    single_rank: bool,
) -> float:
    """Return the single-rank value or multi-rank mathematical median."""
    return (
        metric_worst_total(metric)
        if single_rank
        else metric_median_total(metric)
    )


def metric_skew(
    metric: Optional[StepTimeMetric],
    *,
    single_rank: bool,
) -> Optional[float]:
    """Return cross-rank skew when at least two ranks measured a metric."""
    if metric is None or single_rank or metric.skew_pct is None:
        return None
    return non_negative_finite(metric.skew_pct)


def metric_worst_rank(
    metric: Optional[StepTimeMetric],
) -> Optional[int]:
    """Return the rank carrying one metric's worst value."""
    if metric is None or metric.worst_rank is None:
        return None
    try:
        return int(metric.worst_rank)
    except (TypeError, ValueError, OverflowError):
        return None


def compute_total(
    *,
    forward: Optional[StepTimeMetric],
    backward: Optional[StepTimeMetric],
    optimizer: Optional[StepTimeMetric],
    single_rank: bool,
) -> float:
    """Return the visible aggregate compute total used by diagnosis."""
    return sum(
        metric_total(metric, single_rank=single_rank)
        for metric in (forward, backward, optimizer)
    )


def metric_excess(metric: Optional[StepTimeMetric]) -> float:
    """Return one metric's worst-minus-median excess."""
    return max(0.0, metric_worst_total(metric) - metric_median_total(metric))


def _rank_straggler_pair(
    visible_values: Sequence[tuple[int, float]],
) -> Optional[_RankStragglerPair]:
    if len(visible_values) <= 1:
        return None
    ordered = sorted(
        (
            (int(rank), non_negative_finite(value))
            for rank, value in visible_values
        ),
        key=lambda item: (item[1], item[0]),
    )
    culprit_rank, culprit_value = ordered[0]
    victim_rank, victim_value = ordered[len(ordered) // 2]
    return _RankStragglerPair(
        culprit_rank=culprit_rank,
        victim_rank=victim_rank,
        culprit_value_ms=culprit_value,
        victim_value_ms=victim_value,
        cost_ms=max(0.0, victim_value - culprit_value),
    )


def _rank_value(facts: StepTimeRankFacts, metric: str) -> float:
    return non_negative_finite(facts.average.value(metric) or 0.0)


def _measured(facts: StepTimeRankFacts, metric: str) -> bool:
    return facts.average.value(metric) is not None


def _build_rank_straggler_evidence(
    window: StepTimeWindow,
    *,
    score_threshold: float,
    cause_coverage_threshold: float,
    training_strategy: str,
) -> tuple[Optional[_RankStragglerEvidence], int]:
    """Build diagnosis-specific evidence from the analyzer's rank cohort."""
    eligible = tuple(
        facts
        for rank in window.straggler_ranks
        if (facts := window.rank(rank)) is not None
    )
    if len(eligible) <= 1:
        return None, len(eligible)

    fsdp = training_strategy == "fsdp"
    pair = _rank_straggler_pair(
        tuple(
            (
                facts.global_rank,
                (
                    _rank_value(facts, "forward")
                    + _rank_value(facts, "backward")
                    if fsdp
                    else _rank_value(facts, "backward")
                ),
            )
            for facts in eligible
        )
    )
    if pair is None:
        return None, len(eligible)

    by_rank = {facts.global_rank: facts for facts in eligible}
    culprit = by_rank[pair.culprit_rank]
    victim = by_rank[pair.victim_rank]
    iteration_time = _rank_value(victim, "input_wait") + _rank_value(
        victim,
        "step_time",
    )
    score = pair.cost_ms / iteration_time
    if score < non_negative_finite(score_threshold):
        return None, len(eligible)

    component_excesses = {
        "input": max(
            0.0,
            _rank_value(culprit, "input_wait")
            - _rank_value(victim, "input_wait"),
        )
    }
    if _measured(culprit, "h2d") and _measured(victim, "h2d"):
        component_excesses["h2d"] = max(
            0.0,
            _rank_value(culprit, "h2d") - _rank_value(victim, "h2d"),
        )
    if fsdp:
        component_excesses["compute"] = 0.0
    elif _measured(culprit, "forward") and _measured(victim, "forward"):
        culprit_forward = _rank_value(culprit, "forward")
        victim_forward = _rank_value(victim, "forward")
        component_excesses["compute"] = (
            max(0.0, culprit_forward - victim_forward)
            if culprit_forward > 0.0 and victim_forward > 0.0
            else 0.0
        )

    coverage = {
        component: min(1.0, excess / pair.cost_ms)
        for component, excess in component_excesses.items()
        if pair.cost_ms > 0.0
    }
    ordered = sorted(
        coverage,
        key=lambda component: (
            -coverage[component],
            ("input", "h2d", "compute").index(component),
        ),
    )
    cause = ordered[0]
    cause_coverage = coverage[cause]
    attributed = cause_coverage > 0.0 and cause_coverage >= (
        non_negative_finite(cause_coverage_threshold)
    )

    if attributed and cause == "input":
        kind, status, metric, phase = (
            "INPUT_STRAGGLER",
            "INPUT STRAGGLER",
            "input_wait",
            "input",
        )
    elif attributed and cause == "h2d":
        kind, status, metric, phase = (
            "H2D_STRAGGLER",
            "H2D STRAGGLER",
            "h2d",
            "h2d",
        )
    elif attributed and cause == "compute":
        kind, status, metric, phase = (
            "COMPUTE_STRAGGLER",
            "COMPUTE STRAGGLER",
            "forward",
            "forward",
        )
    else:
        cause = "sync_or_unattributed"
        kind, status, metric, phase = (
            "STRAGGLER",
            "STRAGGLER",
            "forward_backward" if fsdp else "backward",
            "sync",
        )

    return _RankStragglerEvidence(
        kind=kind,
        status=status,
        component=cause,
        metric=metric,
        phase=phase,
        score=score,
        culprit_rank=pair.culprit_rank,
        victim_rank=pair.victim_rank,
        visible_metric="forward_backward" if fsdp else "backward",
        visible_culprit_ms=pair.culprit_value_ms,
        visible_victim_ms=pair.victim_value_ms,
        visible_cost_ms=pair.cost_ms,
        iteration_time_ms=iteration_time,
        component_excesses_ms=component_excesses,
        component_coverage=coverage,
    ), len(eligible)


def largest_compute_phase(
    *,
    window: StepTimeWindow,
    forward: Optional[StepTimeMetric],
    backward: Optional[StepTimeMetric],
    optimizer: Optional[StepTimeMetric],
    step_total: float,
    single_rank: bool,
) -> Optional[ComputeSignal]:
    """Pick the largest phase using the analyzer's coherent compute cohort."""
    cohort = tuple(
        facts
        for rank in window.compute_ranks
        if (facts := window.rank(rank)) is not None
    )
    if window.rank_facts:
        if not cohort:
            return None
        iteration_total = _median(
            tuple(
                _rank_value(facts, "input_wait")
                + _rank_value(facts, "step_time")
                for facts in cohort
            )
        )
        candidates: list[ComputeSignal] = []
        for label, key in (
            ("Forward", "forward"),
            ("Backward", "backward"),
            ("Optimizer", "optimizer_step"),
        ):
            median, worst, worst_rank, excess = _rank_stats(
                tuple(
                    (facts.global_rank, _rank_value(facts, key))
                    for facts in cohort
                )
            )
            if median <= 0.0:
                continue
            candidates.append(
                ComputeSignal(
                    label=label,
                    share=share(median, iteration_total),
                    skew=(excess / median if len(cohort) >= 2 else None),
                    median_ms=median,
                    worst_ms=worst,
                    excess_ms=excess,
                    worst_rank=worst_rank,
                )
            )
        return (
            max(candidates, key=lambda item: item.median_ms)
            if candidates
            else None
        )

    candidates = []
    for label, metric in (
        ("Forward", forward),
        ("Backward", backward),
        ("Optimizer", optimizer),
    ):
        total = metric_total(metric, single_rank=single_rank)
        if metric is None or total <= 0.0:
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
    return max(candidates, key=lambda item: item.share) if candidates else None


_MISSING_SIGNAL_ORDER: tuple[str, ...] = (
    "input_wait",
    "forward",
    "backward",
    "optimizer_step",
    "step_time",
)


def _missing_signal_facts(
    window: StepTimeWindow,
    *,
    input_share: Optional[float],
    compute_share: Optional[float],
    residual_share: Optional[float],
    rank_straggler: Optional[_RankStragglerEvidence],
    straggler_eligible_ranks: int,
    training_strategy: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    ranks = len(window.rank_universe)
    if ranks == 0:
        return (), ()
    counts = {
        metric: len(window.ranks_for(metric))
        for metric in _MISSING_SIGNAL_ORDER
    }
    needed: set[str] = set()
    if len(window.rank_facts) < ranks:
        needed.update(_MISSING_SIGNAL_ORDER)
    if input_share is None:
        needed.update(("input_wait", "step_time"))
    if compute_share is None or residual_share is None:
        needed.update(
            (
                "forward",
                "backward",
                "optimizer_step",
                "input_wait",
                "step_time",
            )
        )
    if ranks > 1 and rank_straggler is None and straggler_eligible_ranks < 2:
        needed.update(("backward", "input_wait", "step_time"))
        if training_strategy == "fsdp":
            needed.add("forward")

    missing = tuple(
        metric
        for metric in _MISSING_SIGNAL_ORDER
        if metric in needed and counts[metric] < ranks
    )
    return missing, tuple(
        (metric, f"{counts[metric]}/{ranks}") for metric in missing
    )


def build_step_time_context(
    *,
    window: StepTimeWindow,
    thresholds: "DiagnosisThresholds",
    training_strategy: Optional[str] = None,
) -> StepTimeAnalysisContext:
    """Build diagnosis-specific facts from one canonical window."""
    metrics = {metric.metric: metric for metric in window.metrics}
    step_metric = metrics["step_time"]
    input_wait_metric = metrics.get("input_wait")
    h2d_metric = metrics.get("h2d")
    residual_metric = metrics.get("residual_proxy")
    forward_metric = metrics.get("forward")
    backward_metric = metrics.get("backward")
    optimizer_metric = metrics.get("optimizer_step")

    coverage = window.coverage
    single_rank = coverage.world_size <= 1 or coverage.ranks_present <= 1
    strategy = normalize_training_strategy(
        training_strategy or window.training_strategy
    )
    step_total = metric_total(step_metric, single_rank=single_rank)

    iteration_cohort = tuple(
        facts
        for rank in window.iteration_ranks
        if (facts := window.rank(rank)) is not None
    )
    input_median, input_worst, input_worst_rank, input_excess = _rank_stats(
        tuple(
            (facts.global_rank, _rank_value(facts, "input_wait"))
            for facts in iteration_cohort
        )
    )
    step_median, step_worst, _rank, _excess = _rank_stats(
        tuple(
            (facts.global_rank, _rank_value(facts, "step_time"))
            for facts in iteration_cohort
        )
    )
    input_total = input_worst if single_rank else input_median
    input_step_total = step_worst if single_rank else step_median
    input_skew = None
    if len(iteration_cohort) >= 2:
        if input_median > 0.0:
            input_skew = input_excess / input_median
        elif input_worst <= 0.0:
            input_skew = 0.0

    rank_straggler, eligible_count = _build_rank_straggler_evidence(
        window,
        score_threshold=thresholds.straggler_score_warn,
        cause_coverage_threshold=thresholds.straggler_cause_coverage_min,
        training_strategy=strategy,
    )
    compute_cohort = tuple(
        facts
        for rank in window.compute_ranks
        if (facts := window.rank(rank)) is not None
    )
    compute_median, compute_worst, _rank, compute_excess = _rank_stats(
        tuple(
            (facts.global_rank, _rank_value(facts, "compute"))
            for facts in compute_cohort
        )
    )
    compute_skew = None
    if len(compute_cohort) >= 2:
        if compute_median > 0.0:
            compute_skew = compute_excess / compute_median
        elif compute_worst <= 0.0:
            compute_skew = 0.0

    missing, signal_coverage = _missing_signal_facts(
        window,
        input_share=window.input_wait_share,
        compute_share=window.compute_share,
        residual_share=window.residual_share,
        rank_straggler=rank_straggler,
        straggler_eligible_ranks=eligible_count,
        training_strategy=strategy,
    )
    return StepTimeAnalysisContext(
        thresholds=thresholds,
        window=window,
        single_rank=single_rank,
        steps_used=int(coverage.steps_used),
        overall_worst_rank=(
            window.worst_rank
            if window.worst_rank is not None
            else metric_worst_rank(step_metric)
        ),
        training_strategy=strategy,
        step_metric=step_metric,
        input_wait_metric=input_wait_metric,
        h2d_metric=h2d_metric,
        residual_metric=residual_metric,
        forward_metric=forward_metric,
        backward_metric=backward_metric,
        optimizer_metric=optimizer_metric,
        step_total=step_total,
        residual_total=metric_total(
            residual_metric,
            single_rank=single_rank,
        ),
        compute_total=compute_total(
            forward=forward_metric,
            backward=backward_metric,
            optimizer=optimizer_metric,
            single_rank=single_rank,
        ),
        residual_share=window.residual_share,
        compute_share=window.compute_share,
        input_bound_share=window.input_wait_share,
        h2d_share=window.h2d_share,
        input_bound_skew=input_skew,
        compute_skew=compute_skew,
        input_bound_worst_rank=input_worst_rank,
        diagnosis_clock=window.clock,
        input_wait_total=input_total,
        input_bound_step_total=input_step_total,
        iteration_time_total=input_total + input_step_total,
        largest_compute=largest_compute_phase(
            window=window,
            forward=forward_metric,
            backward=backward_metric,
            optimizer=optimizer_metric,
            step_total=step_total,
            single_rank=single_rank,
        ),
        rank_straggler=rank_straggler,
        missing_signals=missing,
        signal_coverage=signal_coverage,
    )


__all__ = [
    "ComputeSignal",
    "StepTimeAnalysisContext",
    "build_step_time_context",
    "compute_total",
    "largest_compute_phase",
    "metric_excess",
    "metric_median_total",
    "metric_skew",
    "metric_total",
    "metric_worst_rank",
    "metric_worst_total",
    "non_negative_finite",
    "normalize_training_strategy",
    "share",
]
