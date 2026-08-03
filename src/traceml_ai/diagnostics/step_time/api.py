# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Step-time diagnosis shared by live renderers and summaries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional, cast

from traceml_ai.step_time.model import StepTimeMetric, StepTimeWindow

from ..common import (
    BaseDiagnosis,
    DiagnosticIssue,
    DiagnosticResult,
    Severity,
    severity_rank,
    validate_confidence,
)
from .context import (
    build_step_time_context,
    metric_total,
    metric_worst_rank,
    non_negative_finite,
)
from .policy import (
    DEFAULT_THRESHOLDS,
    DiagnosisThresholds,
    StepTimeDiagnosisPolicy,
)
from .rules import run_step_time_rules
from .trend import DEFAULT_STEP_TREND_HEURISTICS, build_step_trend_note

DiagnosisKind = Literal[
    "NO_DATA",
    "WARMUP",
    "INCOMPLETE_DATA",
    "BALANCED",
    "STRAGGLER",
    "INPUT_STRAGGLER",
    "COMPUTE_STRAGGLER",
    "H2D_STRAGGLER",
    "INPUT_BOUND",
    "H2D_BOUND",
    "COMPUTE_BOUND",
    "RESIDUAL_HEAVY",
]

_STATUS_BY_KIND: dict[DiagnosisKind, str] = {
    "NO_DATA": "NO DATA",
    "WARMUP": "WARMUP",
    "INCOMPLETE_DATA": "INCOMPLETE DATA",
    "BALANCED": "BALANCED",
    "STRAGGLER": "STRAGGLER",
    "INPUT_STRAGGLER": "INPUT STRAGGLER",
    "COMPUTE_STRAGGLER": "COMPUTE STRAGGLER",
    "H2D_STRAGGLER": "H2D STRAGGLER",
    "INPUT_BOUND": "INPUT-BOUND",
    "H2D_BOUND": "H2D-BOUND",
    "COMPUTE_BOUND": "COMPUTE-BOUND",
    "RESIDUAL_HEAVY": "RESIDUAL-HEAVY",
}

_RANK_STRAGGLER_KINDS = frozenset(
    {
        "STRAGGLER",
        "INPUT_STRAGGLER",
        "COMPUTE_STRAGGLER",
        "H2D_STRAGGLER",
    }
)


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
    diagnosis.
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
            f"Only {available} {suffix} per rank available; diagnosis "
            f"requires {required}."
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


def _step_time_issue_sort_key(
    issue: DiagnosticIssue,
) -> tuple[int, float, int]:
    """Order Step Time findings by severity, impact, then rank-local scope."""
    return (
        severity_rank(issue.severity),
        non_negative_finite(issue.score or 0.0),
        int(issue.kind in _RANK_STRAGGLER_KINDS),
    )


def _cap_issue_severity(
    issue: DiagnosticIssue,
    severity: Severity,
) -> DiagnosticIssue:
    """
    Return an issue whose severity is no stronger than `severity`.
    """
    if severity_rank(issue.severity) <= severity_rank(severity):
        return issue
    return replace(issue, severity=severity)


def _apply_trend_note(
    diagnosis: StepDiagnosis,
    *,
    step_metric: Optional[StepTimeMetric],
    residual_metric: Optional[StepTimeMetric],
    input_wait_metric: Optional[StepTimeMetric],
    single_rank: bool,
    residual_share: Optional[float],
    input_bound_share: Optional[float],
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
            input_wait_metric=input_wait_metric,
            residual_share=residual_share,
            input_bound_share=input_bound_share,
            residual_warn_threshold=thresholds.overhead_share_warn,
            input_warn_threshold=thresholds.overhead_share_warn,
            cfg=DEFAULT_STEP_TREND_HEURISTICS,
        )
        if not trend_note:
            return diagnosis
        return replace(diagnosis, note=_merge_note(diagnosis.note, trend_note))
    except Exception:
        return diagnosis


def diagnose_step_time_window(
    window: StepTimeWindow,
    *,
    policy: StepTimeDiagnosisPolicy,
) -> DiagnosticResult[StepDiagnosis]:
    """Diagnose one canonical window without rebuilding analyzer facts."""
    metrics = window.metrics
    thresholds = policy.thresholds
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

    coverage = window.coverage
    single_rank = (coverage.world_size <= 1) or (coverage.ranks_present <= 1)
    steps_used = int(coverage.steps_used)
    overall_worst_rank = metric_worst_rank(
        by_key.get("total_step") or step_metric
    )
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

    if steps_used < thresholds.min_steps_for_warning_diag:
        result = build_step_warmup_diagnosis(
            steps_used=steps_used,
            required_steps=thresholds.min_steps_for_warning_diag,
        )
        return DiagnosticResult(
            primary=replace(result.primary, worst_rank=overall_worst_rank)
        )

    context = build_step_time_context(
        window=window,
        thresholds=thresholds,
    )
    raw_issues = run_step_time_rules(context)
    issue_list = list(raw_issues)
    if context.training_strategy == "fsdp":
        issue_list = [
            _cap_issue_severity(issue, "warn") for issue in issue_list
        ]
    if steps_used < thresholds.min_steps_for_confident_diag:
        issue_list = [
            _cap_issue_severity(issue, "warn") for issue in issue_list
        ]

    issues = tuple(
        sorted(issue_list, key=_step_time_issue_sort_key, reverse=True)
    )
    primary_issue = issues[0] if issues else None

    if primary_issue is not None and primary_issue.kind in {
        "STRAGGLER",
        "INPUT_STRAGGLER",
        "COMPUTE_STRAGGLER",
        "H2D_STRAGGLER",
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
    elif primary_issue is not None and primary_issue.kind == "H2D_BOUND":
        primary = _mk_diag(
            kind="H2D_BOUND",
            severity=primary_issue.severity,
            reason=primary_issue.summary,
            action=primary_issue.action,
            steps_used=context.steps_used,
            worst_rank=(
                primary_issue.ranks[0] if primary_issue.ranks else None
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
                "residual_ms = selected step_time_ms - h2d_ms - compute_ms."
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
        missing_signals = context.missing_signals
        signal_coverage = context.signal_coverage
        if missing_signals:
            primary = _mk_diag(
                kind="INCOMPLETE_DATA",
                severity="info",
                reason=(
                    "Missing timing signals prevent a reliable diagnosis: "
                    + ", ".join(missing_signals)
                    + "."
                ),
                action=(
                    "Instrument the missing phases (auto mode or the "
                    "matching wrap_* helpers) to restore coverage."
                ),
                steps_used=context.steps_used,
                worst_rank=(
                    None if context.single_rank else context.overall_worst_rank
                ),
            )
            incomplete_evidence = {
                "missing_signals": list(missing_signals),
                "signal_coverage": dict(signal_coverage),
            }
        else:
            primary = _mk_diag(
                kind="BALANCED",
                severity="info",
                reason="No dominant bottleneck is visible in this window.",
                action=(
                    "Focus on throughput only if overall speed is still low."
                ),
                steps_used=context.steps_used,
                worst_rank=(
                    None if context.single_rank else context.overall_worst_rank
                ),
            )

    primary = _apply_trend_note(
        primary,
        step_metric=context.step_metric,
        residual_metric=context.residual_metric,
        input_wait_metric=context.input_wait_metric,
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
                evidence=(
                    incomplete_evidence
                    if primary.kind == "INCOMPLETE_DATA"
                    else {}
                ),
            ),
        )

    return DiagnosticResult(
        primary=primary,
        issues=tuple(issues),
    )


__all__ = [
    "Severity",
    "DiagnosisKind",
    "DiagnosisThresholds",
    "DEFAULT_THRESHOLDS",
    "StepDiagnosis",
    "build_step_warmup_diagnosis",
    "diagnose_step_time_window",
]
