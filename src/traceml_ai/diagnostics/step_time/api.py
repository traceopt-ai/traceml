# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Step-time diagnosis shared by live renderers and summaries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Literal, Optional, Sequence, cast

from traceml_ai.renderers.step_time.schema import StepCombinedTimeMetric

from ..common import (
    BaseDiagnosis,
    DiagnosticIssue,
    DiagnosticResult,
    Severity,
    sort_issues,
    validate_confidence,
)
from .context import (
    build_step_time_context,
    metric_median_total,
    metric_skew,
    metric_total,
    metric_worst_rank,
    metric_worst_total,
    non_negative_finite,
    share,
)
from .policy import DEFAULT_THRESHOLDS, DiagnosisThresholds
from .rules import run_step_time_rules
from .trend import DEFAULT_STEP_TREND_HEURISTICS, build_step_trend_note

DiagnosisKind = Literal[
    "NO_DATA",
    "WARMUP",
    "BALANCED",
    "STRAGGLER",
    "INPUT_STRAGGLER",
    "COMPUTE_STRAGGLER",
    "H2D_STRAGGLER",
    "RESIDUAL_STRAGGLER",
    "INPUT_BOUND",
    "COMPUTE_BOUND",
    "RESIDUAL_HEAVY",
]

_STATUS_BY_KIND: dict[DiagnosisKind, str] = {
    "NO_DATA": "NO DATA",
    "WARMUP": "WARMUP",
    "BALANCED": "BALANCED",
    "STRAGGLER": "STRAGGLER",
    "INPUT_STRAGGLER": "INPUT STRAGGLER",
    "COMPUTE_STRAGGLER": "COMPUTE STRAGGLER",
    "H2D_STRAGGLER": "H2D STRAGGLER",
    "RESIDUAL_STRAGGLER": "RESIDUAL STRAGGLER",
    "INPUT_BOUND": "INPUT-BOUND",
    "COMPUTE_BOUND": "COMPUTE-BOUND",
    "RESIDUAL_HEAVY": "RESIDUAL-HEAVY",
}

_PRIMARY_KIND_PRIORITY: dict[str, int] = {
    "STRAGGLER": 50,
    "INPUT_STRAGGLER": 40,
    "COMPUTE_STRAGGLER": 40,
    "H2D_STRAGGLER": 40,
    "RESIDUAL_STRAGGLER": 40,
    "INPUT_BOUND": 30,
    "RESIDUAL_HEAVY": 20,
    "COMPUTE_BOUND": 10,
}


@dataclass(frozen=True)
class StepDiagnosis(BaseDiagnosis):
    """
    Primary diagnosis payload used by runtime renderers and summaries.
    """

    kind: DiagnosisKind
    steps_used: int
    worst_rank: Optional[int] = None
    note: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        validate_confidence(self.confidence)


def _mk_diag(
    *,
    kind: DiagnosisKind,
    severity: Severity,
    reason: str,
    action: str,
    steps_used: int,
    worst_rank: Optional[int] = None,
    note: Optional[str] = None,
) -> StepDiagnosis:
    return StepDiagnosis(
        kind=kind,
        severity=severity,
        status=_STATUS_BY_KIND[kind],
        reason=reason,
        action=action,
        steps_used=int(steps_used),
        worst_rank=worst_rank,
        note=note,
    )


def build_step_warmup_diagnosis(
    *,
    steps_used: int,
    required_steps: int,
    max_steps_used: Optional[int] = None,
) -> DiagnosticResult[StepDiagnosis]:
    """
    Build the explicit partial-data diagnosis for a non-empty timing window.

    ``NO_DATA`` is reserved for missing or unusable timing data. ``WARMUP``
    means TraceML has timing samples, but fewer than the configured minimum for
    a stable summary diagnosis.
    """
    low = max(0, int(steps_used))
    high = max(low, int(max_steps_used if max_steps_used is not None else low))
    required = max(1, int(required_steps))
    available = f"{low}" if low == high else f"{low}-{high}"
    suffix = "step" if high == 1 else "steps"
    primary = _mk_diag(
        kind="WARMUP",
        severity="info",
        reason=(
            f"Only {available} {suffix} per rank available; summary "
            f"diagnosis requires {required}."
        ),
        action="Use a longer run for a stable timing diagnosis.",
        steps_used=low,
    )
    return DiagnosticResult(primary=primary)


def _merge_note(base: Optional[str], extra: Optional[str]) -> Optional[str]:
    if not extra:
        return base
    if not base:
        return extra
    return f"{base} {extra}"


def _pct(value: float) -> str:
    """
    Format a ratio as a percentage string.
    """
    return f"{non_negative_finite(value) * 100.0:.1f}%"


def _rank_str(rank: Optional[int]) -> str:
    """
    Format a rank identifier for UI text.
    """
    return f"r{rank}" if rank is not None else "—"


def _severity(value: float, crit_threshold: float) -> Severity:
    """
    Map a scalar signal to warn or crit severity.
    """
    return "crit" if non_negative_finite(value) >= crit_threshold else "warn"


def _primary_issue_rank(issue: DiagnosticIssue) -> int:
    """
    Rank contributor issues for primary-diagnosis selection.
    """
    return _PRIMARY_KIND_PRIORITY.get(issue.kind, 0)


def _top_rank_entries(
    rank_values: Dict[int, float],
    *,
    max_items: int = 3,
) -> list[Dict[str, Any]]:
    """
    Build a compact ranked list of the most affected ranks for one metric.
    """
    if not rank_values:
        return []

    ordered = sorted(
        (
            (int(rank), non_negative_finite(value))
            for rank, value in rank_values.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not ordered:
        return []

    values = sorted(value for _, value in ordered)
    median_value = values[len(values) // 2]

    out: list[Dict[str, Any]] = []
    for rank, value in ordered[: max(1, int(max_items))]:
        excess = max(0.0, value - median_value)
        out.append(
            {
                "rank": rank,
                "value_ms": value,
                "excess_vs_median_ms": excess,
                "pct_vs_median": (
                    (excess / median_value) if median_value > 0.0 else None
                ),
            }
        )
    return out


def _rank_summary_values(
    rank_values: Dict[int, float],
) -> tuple[float, float, Optional[int], float]:
    """
    Return median, worst value, worst rank, and skew for rank values.
    """
    if not rank_values:
        return 0.0, 0.0, None, 0.0
    clean = {
        int(rank): non_negative_finite(value)
        for rank, value in rank_values.items()
    }
    ordered = sorted(clean.values())
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = float(ordered[mid])
    else:
        median = float((ordered[mid - 1] + ordered[mid]) / 2.0)
    worst_rank = max(clean, key=lambda rank: (clean[rank], -int(rank)))
    worst = clean[worst_rank]
    skew = ((worst - median) / median) if median > 0.0 else 0.0
    return median, worst, int(worst_rank), max(0.0, skew)


def _metric_attribution_entry(
    *,
    metric: Optional[StepCombinedTimeMetric],
    metric_key: str,
    rank_values: Dict[int, float],
    step_total: float,
    single_rank: bool,
    phase: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build one machine-readable attribution block for a metric / phase.
    """
    return {
        "metric": metric_key,
        "phase": phase,
        "median_total_ms": metric_median_total(metric),
        "worst_total_ms": metric_worst_total(metric),
        "worst_rank": metric_worst_rank(metric),
        "skew_pct": metric_skew(metric, single_rank=single_rank),
        "share_pct": share(
            metric_total(metric, single_rank=single_rank), step_total
        ),
        "top_ranks": _top_rank_entries(rank_values),
    }


def _select_primary_issue(
    issues: Sequence[DiagnosticIssue],
) -> Optional[DiagnosticIssue]:
    """
    Return the strongest atomic issue candidate.
    """
    if not issues:
        return None
    ranked = sorted(
        issues,
        key=lambda issue: (
            _primary_issue_rank(issue),
            float(issue.score or 0.0),
        ),
        reverse=True,
    )
    return ranked[0]


def _apply_trend_note(
    diagnosis: StepDiagnosis,
    *,
    step_metric: Optional[StepCombinedTimeMetric],
    residual_metric: Optional[StepCombinedTimeMetric],
    dataloader_metric: Optional[StepCombinedTimeMetric],
    single_rank: bool,
    residual_share: float,
    input_bound_share: float,
    thresholds: DiagnosisThresholds,
) -> StepDiagnosis:
    """
    Best-effort trend annotation.
    """
    try:
        trend_note = build_step_trend_note(
            diagnosis_kind=diagnosis.kind,
            steps_used=diagnosis.steps_used,
            single_rank=single_rank,
            step_metric=step_metric,
            residual_metric=residual_metric,
            dataloader_metric=dataloader_metric,
            residual_share=residual_share,
            input_bound_share=input_bound_share,
            residual_warn_threshold=thresholds.residual_share_warn,
            input_warn_threshold=thresholds.input_share_warn,
            cfg=DEFAULT_STEP_TREND_HEURISTICS,
        )
        if not trend_note:
            return diagnosis
        return replace(diagnosis, note=_merge_note(diagnosis.note, trend_note))
    except Exception:
        return diagnosis


def build_step_diagnosis_result(
    metrics: Sequence[StepCombinedTimeMetric],
    thresholds: DiagnosisThresholds = DEFAULT_THRESHOLDS,
    *,
    per_rank_timing: Optional[Dict[int, Dict[str, float]]] = None,
    diagnosis_clock: str = "cpu",
) -> DiagnosticResult[StepDiagnosis]:
    """
    Build a rich step-time diagnosis result from one analyzed window.

    Runtime consumers should typically use `result.primary`. Final-summary and
    dashboard consumers can additionally use:
    - `result.issues`
    - `result.metric_attribution`
    """
    metric_names = [metric.metric for metric in metrics]
    if len(metric_names) != len(set(metric_names)):
        primary = _mk_diag(
            kind="NO_DATA",
            severity="info",
            reason="Duplicate metric keys in diagnosis input.",
            action="Check upstream aggregation.",
            steps_used=0,
        )
        return DiagnosticResult(primary=primary)

    by_key = {metric.metric: metric for metric in metrics}
    step_metric = by_key.get("step_time")
    if step_metric is None:
        primary = _mk_diag(
            kind="NO_DATA",
            severity="info",
            reason="step_time metric is missing.",
            action="Wait for the first complete window.",
            steps_used=0,
        )
        return DiagnosticResult(primary=primary)

    coverage = step_metric.coverage
    single_rank = (coverage.world_size <= 1) or (coverage.ranks_present <= 1)
    steps_used = int(step_metric.summary.steps_used)
    overall_worst_rank = metric_worst_rank(step_metric)
    step_total = metric_total(step_metric, single_rank=single_rank)

    if step_total <= 0.0:
        primary = _mk_diag(
            kind="NO_DATA",
            severity="info",
            reason="No usable step-time data yet.",
            action="Wait for the first complete window.",
            steps_used=steps_used,
            worst_rank=overall_worst_rank,
        )
        return DiagnosticResult(primary=primary)

    if steps_used < thresholds.min_steps_for_confident_diag:
        result = build_step_warmup_diagnosis(
            steps_used=steps_used,
            required_steps=thresholds.min_steps_for_confident_diag,
        )
        return DiagnosticResult(
            primary=replace(result.primary, worst_rank=overall_worst_rank)
        )

    context = build_step_time_context(
        metrics=metrics,
        thresholds=thresholds,
        per_rank_timing=per_rank_timing,
        diagnosis_clock=diagnosis_clock,
    )
    raw_issues = run_step_time_rules(context)
    issue_list = list(raw_issues)

    issues = sort_issues(issue_list)
    primary_issue = _select_primary_issue(issues)

    if primary_issue is not None and primary_issue.kind in {
        "STRAGGLER",
        "INPUT_STRAGGLER",
        "COMPUTE_STRAGGLER",
        "H2D_STRAGGLER",
        "RESIDUAL_STRAGGLER",
    }:
        worst_rank = primary_issue.ranks[0] if primary_issue.ranks else None
        primary = _mk_diag(
            kind=cast(DiagnosisKind, primary_issue.kind),
            severity=primary_issue.severity,
            reason=primary_issue.summary,
            action=primary_issue.action,
            steps_used=context.steps_used,
            worst_rank=worst_rank,
        )
    elif primary_issue is not None and primary_issue.kind == "INPUT_BOUND":
        primary = _mk_diag(
            kind="INPUT_BOUND",
            severity=primary_issue.severity,
            reason=primary_issue.summary,
            action=primary_issue.action,
            steps_used=context.steps_used,
            worst_rank=(
                None if context.single_rank else context.input_bound_worst_rank
            ),
        )
    elif primary_issue is not None and primary_issue.kind == "RESIDUAL_HEAVY":
        primary = _mk_diag(
            kind="RESIDUAL_HEAVY",
            severity=primary_issue.severity,
            reason=primary_issue.summary,
            action=primary_issue.action,
            steps_used=context.steps_used,
            worst_rank=(
                None if context.single_rank else context.overall_worst_rank
            ),
            note=(
                "residual_ms = total_step_ms - dataloader_ms - h2d_ms - "
                "compute_ms."
            ),
        )
    elif primary_issue is not None and primary_issue.kind == "COMPUTE_BOUND":
        primary = _mk_diag(
            kind="COMPUTE_BOUND",
            severity=primary_issue.severity,
            reason=primary_issue.summary,
            action=primary_issue.action,
            steps_used=context.steps_used,
            worst_rank=(
                None if context.single_rank else context.overall_worst_rank
            ),
        )
    else:
        primary = _mk_diag(
            kind="BALANCED",
            severity="info",
            reason="No dominant bottleneck is visible in this window.",
            action="Focus on throughput only if overall speed is still low.",
            steps_used=context.steps_used,
            worst_rank=(
                None if context.single_rank else context.overall_worst_rank
            ),
        )

    primary = _apply_trend_note(
        primary,
        step_metric=context.step_metric,
        residual_metric=context.residual_metric,
        dataloader_metric=context.dataloader_metric,
        single_rank=context.single_rank,
        residual_share=context.residual_share,
        input_bound_share=context.input_bound_share,
        thresholds=thresholds,
    )

    if not issues:
        issues = (
            DiagnosticIssue(
                kind=primary.kind,
                status=primary.status,
                severity=primary.severity,
                summary=primary.reason,
                action=primary.action,
                ranks=(
                    (primary.worst_rank,)
                    if primary.worst_rank is not None
                    else ()
                ),
            ),
        )

    fwd_rank_values = context.rank_values.get("forward", {})
    bwd_rank_values = context.rank_values.get("backward", {})
    opt_rank_values = context.rank_values.get("optimizer_step", {})
    compute_rank_values = context.clean_rank_values.get("clean_compute", {})
    if not compute_rank_values:
        compute_rank_values = {}
        for rank in sorted(
            set(fwd_rank_values) | set(bwd_rank_values) | set(opt_rank_values)
        ):
            compute_rank_values[int(rank)] = (
                non_negative_finite(fwd_rank_values.get(rank, 0.0))
                + non_negative_finite(bwd_rank_values.get(rank, 0.0))
                + non_negative_finite(opt_rank_values.get(rank, 0.0))
            )
    (
        compute_median_ms,
        compute_worst_ms,
        compute_worst_rank,
        compute_skew,
    ) = _rank_summary_values(compute_rank_values)

    metric_attribution = {
        "dataloader_fetch": _metric_attribution_entry(
            metric=context.dataloader_metric,
            metric_key="dataloader_fetch",
            rank_values=context.rank_values.get("dataloader_fetch", {}),
            step_total=context.step_total,
            single_rank=context.single_rank,
            phase="dataloader",
        ),
        "h2d": _metric_attribution_entry(
            metric=context.h2d_metric,
            metric_key="h2d",
            rank_values=context.rank_values.get("h2d", {}),
            step_total=context.step_total,
            single_rank=context.single_rank,
            phase="h2d",
        ),
        "forward": _metric_attribution_entry(
            metric=context.forward_metric,
            metric_key="forward",
            rank_values=fwd_rank_values,
            step_total=context.step_total,
            single_rank=context.single_rank,
            phase="forward",
        ),
        "backward": _metric_attribution_entry(
            metric=context.backward_metric,
            metric_key="backward",
            rank_values=bwd_rank_values,
            step_total=context.step_total,
            single_rank=context.single_rank,
            phase="backward",
        ),
        "optimizer_step": _metric_attribution_entry(
            metric=context.optimizer_metric,
            metric_key="optimizer_step",
            rank_values=opt_rank_values,
            step_total=context.step_total,
            single_rank=context.single_rank,
            phase="optimizer",
        ),
        "residual_proxy": _metric_attribution_entry(
            metric=context.residual_metric,
            metric_key="residual_proxy",
            rank_values=context.rank_values.get("residual_proxy", {}),
            step_total=context.step_total,
            single_rank=context.single_rank,
            phase="residual",
        ),
        "step_time": _metric_attribution_entry(
            metric=context.step_metric,
            metric_key="step_time",
            rank_values=context.rank_values.get("step_time", {}),
            step_total=context.step_total,
            single_rank=context.single_rank,
            phase="step",
        ),
        "compute": {
            "metric": "compute",
            "phase": "clean_compute",
            "median_total_ms": compute_median_ms,
            "worst_total_ms": compute_worst_ms,
            "worst_rank": compute_worst_rank,
            "skew_pct": compute_skew,
            "share_pct": share(compute_median_ms, context.step_total),
            "top_ranks": _top_rank_entries(compute_rank_values),
        },
    }

    return DiagnosticResult(
        primary=primary,
        issues=tuple(issues),
        metric_attribution=metric_attribution,
    )


def build_step_diagnosis(
    metrics: Sequence[StepCombinedTimeMetric],
    thresholds: DiagnosisThresholds = DEFAULT_THRESHOLDS,
    *,
    per_rank_timing: Optional[Dict[int, Dict[str, float]]] = None,
    diagnosis_clock: str = "cpu",
) -> StepDiagnosis:
    """
    Build one primary diagnosis from step-combined metrics.

    This remains the backward-compatible runtime entry point. Richer consumers
    should use `build_step_diagnosis_result(...)`.
    """
    primary = build_step_diagnosis_result(
        metrics,
        thresholds=thresholds,
        per_rank_timing=per_rank_timing,
        diagnosis_clock=diagnosis_clock,
    ).primary
    if not isinstance(primary, StepDiagnosis):
        raise TypeError(
            "build_step_diagnosis_result() must return StepDiagnosis as primary"
        )
    return primary


__all__ = [
    "Severity",
    "DiagnosisKind",
    "DiagnosisThresholds",
    "DEFAULT_THRESHOLDS",
    "StepDiagnosis",
    "build_step_warmup_diagnosis",
    "build_step_diagnosis",
    "build_step_diagnosis_result",
]
