# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Step-time diagnosis shared by live renderers and summaries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence, cast

from traceml_ai.step_time.model import StepTimeMetric, StepTimeWindow

if TYPE_CHECKING:
    from .context import StepTimeAnalysisContext

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
    metric_median_total,
    metric_skew,
    metric_total,
    metric_worst_rank,
    metric_worst_total,
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
) -> tuple[float, float, Optional[int], Optional[float]]:
    """
    Return median, worst value, worst rank, and skew for rank values.
    """
    if not rank_values:
        return 0.0, 0.0, None, None
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
    if len(clean) < 2:
        skew = None
    elif median > 0.0:
        skew = (worst - median) / median
    elif worst <= 0.0:
        skew = 0.0
    else:
        skew = None
    return (
        median,
        worst,
        int(worst_rank),
        max(0.0, skew) if skew is not None else None,
    )


def _metric_attribution_entry(
    *,
    metric: Optional[StepTimeMetric],
    metric_key: str,
    rank_values: Dict[int, float],
    component_share: Optional[float],
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
        "share_pct": component_share,
        "top_ranks": _top_rank_entries(rank_values),
    }


def _rank_values(window: StepTimeWindow, metric: str) -> Dict[int, float]:
    """Return an on-demand rank lookup for optional rich attribution."""
    return {
        facts.global_rank: float(value)
        for facts in window.rank_facts
        if (value := facts.average.value(metric)) is not None
    }


def _component_share(window: StepTimeWindow, metric: str) -> Optional[float]:
    """Calculate a presentation share for optional rich attribution."""
    prepared = {
        "input_wait": window.input_wait_share,
        "h2d": window.h2d_share,
        "compute": window.compute_share,
        "residual_proxy": window.residual_share,
    }
    if metric in prepared:
        return prepared[metric]

    shares: list[float] = []
    for facts in window.rank_facts:
        numerator = facts.average.value(metric)
        input_wait = facts.average.input_wait_ms
        step_time = facts.average.step_time_ms
        if numerator is None or input_wait is None or step_time is None:
            continue
        iteration = input_wait + step_time
        if iteration > 0.0:
            shares.append(float(numerator) / iteration)
    if not shares:
        return None
    ordered = sorted(shares)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _build_metric_attribution(
    context: "StepTimeAnalysisContext",
) -> Dict[str, Any]:
    """Build the optional detailed attribution payload from typed facts."""
    window = context.window
    metrics = {
        "input_wait": context.input_wait_metric,
        "h2d": context.h2d_metric,
        "forward": context.forward_metric,
        "backward": context.backward_metric,
        "optimizer_step": context.optimizer_metric,
        "residual_proxy": context.residual_metric,
        "step_time": context.step_metric,
    }
    phases = {
        "input_wait": "input",
        "h2d": "h2d",
        "forward": "forward",
        "backward": "backward",
        "optimizer_step": "optimizer",
        "residual_proxy": "residual",
        "step_time": "step",
    }
    attribution = {
        metric: _metric_attribution_entry(
            metric=metrics[metric],
            metric_key=metric,
            rank_values=_rank_values(window, metric),
            component_share=_component_share(window, metric),
            single_rank=context.single_rank,
            phase=phases[metric],
        )
        for metric in metrics
    }
    compute_values = _rank_values(window, "compute")
    median, worst, worst_rank, skew = _rank_summary_values(compute_values)
    attribution["compute"] = {
        "metric": "compute",
        "phase": "compute",
        "median_total_ms": median,
        "worst_total_ms": worst,
        "worst_rank": worst_rank,
        "skew_pct": skew,
        "share_pct": context.compute_share,
        "top_ranks": _top_rank_entries(compute_values),
    }
    return attribution


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
    training_strategy: Optional[str] = None,
    include_attribution: bool = False,
) -> DiagnosticResult[StepDiagnosis]:
    """Diagnose one canonical window without rebuilding analyzer facts.

    Detailed per-metric attribution is presentation-only and opt-in. Runtime
    pipeline calls therefore avoid building rank maps they do not consume.
    """
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
    overall_worst_rank = (
        window.worst_rank
        if window.worst_rank is not None
        else metric_worst_rank(step_metric)
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
        training_strategy=training_strategy,
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
        metric_attribution=(
            _build_metric_attribution(context) if include_attribution else {}
        ),
    )


def build_step_diagnosis_result(
    metrics: Sequence[StepTimeMetric],
    thresholds: DiagnosisThresholds = DEFAULT_THRESHOLDS,
    *,
    per_rank_timing: Optional[Dict[int, Dict[str, float]]] = None,
    expected_ranks: Optional[Sequence[int]] = None,
    diagnosis_clock: str = "cpu",
    training_strategy: str = "ddp",
) -> DiagnosticResult[StepDiagnosis]:
    """Preserve the released metric/rank-map diagnosis entry point.

    TODO(PR9): Remove after external callers migrate to window diagnosis.
    """
    from traceml_ai.utils.step_time_window import (
        build_step_time_window_from_rank_averages,
    )

    window = build_step_time_window_from_rank_averages(
        metrics,
        per_rank_timing=per_rank_timing,
        expected_ranks=expected_ranks,
        diagnosis_clock=diagnosis_clock,
        training_strategy=training_strategy,
    )
    return diagnose_step_time_window(
        window,
        policy=StepTimeDiagnosisPolicy(
            name="legacy",
            thresholds=thresholds,
        ),
        training_strategy=training_strategy,
        include_attribution=True,
    )


def build_step_diagnosis(
    metrics: Sequence[StepTimeMetric],
    thresholds: DiagnosisThresholds = DEFAULT_THRESHOLDS,
    *,
    per_rank_timing: Optional[Dict[int, Dict[str, float]]] = None,
    expected_ranks: Optional[Sequence[int]] = None,
    diagnosis_clock: str = "cpu",
    training_strategy: str = "ddp",
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
        expected_ranks=expected_ranks,
        diagnosis_clock=diagnosis_clock,
        training_strategy=training_strategy,
    ).primary
    if not isinstance(primary, StepDiagnosis):
        raise TypeError(
            "build_step_diagnosis_result() must return StepDiagnosis "
            "as primary"
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
    "diagnose_step_time_window",
]
