# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Step-time diagnosis shared by live renderers and summaries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Literal, Optional, Sequence

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
    ComputeSignal,
    build_step_time_context,
    compute_median_total,
    compute_worst_total,
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
    "INPUT_BOUND",
    "COMPUTE_BOUND",
    "WAIT_HEAVY",
]

_STATUS_BY_KIND: dict[DiagnosisKind, str] = {
    "NO_DATA": "NO DATA",
    "WARMUP": "WARMUP",
    "BALANCED": "BALANCED",
    "STRAGGLER": "STRAGGLER",
    "INPUT_STRAGGLER": "INPUT STRAGGLER",
    "COMPUTE_STRAGGLER": "COMPUTE STRAGGLER",
    "INPUT_BOUND": "INPUT-BOUND",
    "COMPUTE_BOUND": "COMPUTE-BOUND",
    "WAIT_HEAVY": "WAIT-HEAVY",
}

_PRIMARY_KIND_PRIORITY: dict[str, int] = {
    "STRAGGLER": 50,
    "INPUT_STRAGGLER": 40,
    "COMPUTE_STRAGGLER": 39,
    "INPUT_BOUND": 30,
    "WAIT_HEAVY": 20,
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


def _issue_by_kind(
    issues: Sequence[DiagnosticIssue],
    kind: str,
) -> Optional[DiagnosticIssue]:
    """
    Return the first issue matching one issue kind.
    """
    for issue in issues:
        if issue.kind == kind:
            return issue
    return None


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
    wait_metric: Optional[StepCombinedTimeMetric],
    dataloader_metric: Optional[StepCombinedTimeMetric],
    single_rank: bool,
    wait_share: float,
    dataloader_share: float,
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
            wait_metric=wait_metric,
            dataloader_metric=dataloader_metric,
            wait_share=wait_share,
            dataloader_share=dataloader_share,
            wait_warn_threshold=thresholds.wait_share_warn,
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
    )
    raw_issues = run_step_time_rules(context)
    issue_list = list(raw_issues)

    input_issue = _issue_by_kind(issue_list, "INPUT_STRAGGLER")
    compute_issue = _issue_by_kind(issue_list, "COMPUTE_STRAGGLER")

    if input_issue is not None and compute_issue is not None:
        combined_rank = (
            context.dataloader_worst_rank
            if context.input_straggler_score >= context.compute_straggler_score
            else context.compute_worst_rank
        )
        combined_ranks = tuple(
            sorted(
                {
                    rank
                    for rank in (
                        context.dataloader_worst_rank,
                        context.compute_worst_rank,
                    )
                    if rank is not None
                }
            )
        )
        issue_list.append(
            DiagnosticIssue(
                kind="STRAGGLER",
                status="STRAGGLER",
                severity=_severity(
                    max(
                        context.input_straggler_score,
                        context.compute_straggler_score,
                    ),
                    max(
                        thresholds.input_straggler_score_crit,
                        thresholds.compute_straggler_score_crit,
                    ),
                ),
                summary="Both input and compute are uneven across ranks.",
                action="Inspect the slowest rank and both dominant phases.",
                metric="step_time",
                phase="combined",
                score=max(
                    context.input_straggler_score,
                    context.compute_straggler_score,
                ),
                ranks=combined_ranks
                or ((combined_rank,) if combined_rank is not None else ()),
                evidence={
                    "input_score": context.input_straggler_score,
                    "compute_score": context.compute_straggler_score,
                },
            )
        )

    issues = sort_issues(issue_list)
    primary_issue = _select_primary_issue(issues)

    if input_issue is not None and compute_issue is not None:
        dominant_rank = (
            context.dataloader_worst_rank
            if context.input_straggler_score >= context.compute_straggler_score
            else context.compute_worst_rank
        )
        primary = _mk_diag(
            kind="STRAGGLER",
            severity=_severity(
                max(
                    context.input_straggler_score,
                    context.compute_straggler_score,
                ),
                max(
                    thresholds.input_straggler_score_crit,
                    thresholds.compute_straggler_score_crit,
                ),
            ),
            reason="Both input and compute are uneven across ranks.",
            action="Inspect the slowest rank and both dominant phases.",
            steps_used=context.steps_used,
            worst_rank=dominant_rank,
            note=(
                f"Input score {_pct(context.input_straggler_score)}, "
                f"compute score {_pct(context.compute_straggler_score)}."
            ),
        )
    elif input_issue is not None:
        primary = _mk_diag(
            kind="INPUT_STRAGGLER",
            severity=input_issue.severity,
            reason=input_issue.summary,
            action=input_issue.action,
            steps_used=context.steps_used,
            worst_rank=context.dataloader_worst_rank,
            note=f"Dataloader share is {_pct(context.dataloader_share)}.",
        )
    elif compute_issue is not None:
        primary = _mk_diag(
            kind="COMPUTE_STRAGGLER",
            severity=compute_issue.severity,
            reason=compute_issue.summary,
            action=compute_issue.action,
            steps_used=context.steps_used,
            worst_rank=context.compute_worst_rank,
            note=f"Compute share is {_pct(context.compute_share)}.",
        )
    elif primary_issue is not None and primary_issue.kind == "INPUT_BOUND":
        primary = _mk_diag(
            kind="INPUT_BOUND",
            severity=primary_issue.severity,
            reason=primary_issue.summary,
            action=primary_issue.action,
            steps_used=context.steps_used,
            worst_rank=(
                None if context.single_rank else context.dataloader_worst_rank
            ),
        )
    elif primary_issue is not None and primary_issue.kind == "WAIT_HEAVY":
        primary = _mk_diag(
            kind="WAIT_HEAVY",
            severity=primary_issue.severity,
            reason=primary_issue.summary,
            action=primary_issue.action,
            steps_used=context.steps_used,
            worst_rank=(
                None if context.single_rank else context.overall_worst_rank
            ),
            note=(
                "wait_ms = total_step_ms - dataloader_ms - h2d_ms - "
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
        wait_metric=context.wait_metric,
        dataloader_metric=context.dataloader_metric,
        single_rank=context.single_rank,
        wait_share=context.wait_share,
        dataloader_share=context.dataloader_share,
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
    compute_rank_values: Dict[int, float] = {}
    for rank in sorted(
        set(fwd_rank_values) | set(bwd_rank_values) | set(opt_rank_values)
    ):
        compute_rank_values[int(rank)] = (
            non_negative_finite(fwd_rank_values.get(rank, 0.0))
            + non_negative_finite(bwd_rank_values.get(rank, 0.0))
            + non_negative_finite(opt_rank_values.get(rank, 0.0))
        )

    metric_attribution = {
        "dataloader_fetch": _metric_attribution_entry(
            metric=context.dataloader_metric,
            metric_key="dataloader_fetch",
            rank_values=context.rank_values.get("dataloader_fetch", {}),
            step_total=context.step_total,
            single_rank=context.single_rank,
            phase="dataloader",
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
        "wait_proxy": _metric_attribution_entry(
            metric=context.wait_metric,
            metric_key="wait_proxy",
            rank_values=context.rank_values.get("wait_proxy", {}),
            step_total=context.step_total,
            single_rank=context.single_rank,
            phase="wait",
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
            "phase": (
                context.dominant_compute.label.lower()
                if context.dominant_compute is not None
                else "compute"
            ),
            "median_total_ms": compute_median_total(
                forward=context.forward_metric,
                backward=context.backward_metric,
                optimizer=context.optimizer_metric,
            ),
            "worst_total_ms": compute_worst_total(
                forward=context.forward_metric,
                backward=context.backward_metric,
                optimizer=context.optimizer_metric,
            ),
            "worst_rank": context.compute_worst_rank,
            "skew_pct": context.compute_skew,
            "share_pct": context.compute_share,
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
) -> StepDiagnosis:
    """
    Build one primary diagnosis from step-combined metrics.

    This remains the backward-compatible runtime entry point. Richer consumers
    should use `build_step_diagnosis_result(...)`.
    """
    primary = build_step_diagnosis_result(
        metrics,
        thresholds=thresholds,
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
    "ComputeSignal",
    "build_step_warmup_diagnosis",
    "build_step_diagnosis",
    "build_step_diagnosis_result",
]
