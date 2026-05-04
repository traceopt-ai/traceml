"""
Summary-oriented step-memory diagnosis rules.

These rules intentionally mirror the existing live diagnosis categories so the
summary issue list stays coherent with the primary diagnosis while preserving
all materially triggered signals in JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from traceml.diagnostics.common import (
    DiagnosticIssue,
    DiagnosticRule,
    severity_rank,
)
from traceml.diagnostics.step_memory import (
    DEFAULT_STEP_MEMORY_THRESHOLDS,
    StepMemoryDiagnosisThresholds,
)

from .adapters import StepMemorySummaryMetricSignals


def _severity(value: float, crit_threshold: float) -> str:
    return "crit" if float(value) >= float(crit_threshold) else "warn"


def _metric_label(metric_name: str) -> str:
    return metric_name.replace("_", " ")


def _format_creep_note(
    metric: StepMemorySummaryMetricSignals,
) -> Optional[str]:
    trend = metric.trend
    parts = []
    if (
        trend.baseline_avg_bytes is not None
        and trend.recent_avg_bytes is not None
    ):
        parts.append(
            f"baseline {trend.baseline_avg_bytes:.0f} B -> "
            f"recent {trend.recent_avg_bytes:.0f} B"
        )
    if trend.overall_abs_delta_bytes is not None:
        sign = "+" if trend.overall_abs_delta_bytes >= 0.0 else "-"
        parts.append(f"{sign}{abs(trend.overall_abs_delta_bytes):.0f} B")
    if trend.overall_worst_growth_pct is not None:
        parts.append(f"(~{trend.overall_worst_growth_pct * 100.0:.0f}%)")
    if not parts:
        return None
    return ", ".join(parts)


@dataclass(frozen=True)
class _BaseStepMemorySummaryRule(
    DiagnosticRule[StepMemorySummaryMetricSignals]
):
    name: str
    thresholds: StepMemoryDiagnosisThresholds = DEFAULT_STEP_MEMORY_THRESHOLDS

    def _issue(
        self,
        *,
        kind: str,
        status: str,
        severity: str,
        summary: str,
        action: str,
        metric: StepMemorySummaryMetricSignals,
        score: Optional[float] = None,
        evidence: Optional[dict] = None,
    ) -> DiagnosticIssue:
        ranks: Tuple[int, ...] = ()
        if metric.worst_rank is not None:
            ranks = (int(metric.worst_rank),)
        return DiagnosticIssue(
            kind=kind,
            status=status,
            severity=severity,
            summary=summary,
            action=action,
            metric=metric.metric,
            phase="memory",
            score=float(score) if score is not None else None,
            skew_pct=metric.skew_pct,
            ranks=ranks,
            evidence=dict(evidence or {}),
        )


@dataclass(frozen=True)
class HighPressureRule(_BaseStepMemorySummaryRule):
    name: str = "high_pressure"

    def evaluate(
        self,
        context: StepMemorySummaryMetricSignals,
    ) -> Optional[DiagnosticIssue]:
        pressure = context.pressure_frac
        if (
            pressure is None
            or context.steps_used < int(self.thresholds.min_steps_for_diag)
            or pressure < float(self.thresholds.pressure_warn_fraction)
        ):
            return None
        sev = _severity(pressure, self.thresholds.pressure_crit_fraction)
        return self._issue(
            kind="HIGH_PRESSURE",
            status="HIGH PRESSURE",
            severity=sev,
            summary=(
                f"{_metric_label(context.metric)} is near device capacity "
                f"(~{pressure * 100.0:.0f}%)."
            ),
            action="Reduce memory load.",
            metric=context,
            score=pressure,
            evidence={"pressure_frac": pressure},
        )


@dataclass(frozen=True)
class ImbalanceRule(_BaseStepMemorySummaryRule):
    name: str = "imbalance"

    def evaluate(
        self,
        context: StepMemorySummaryMetricSignals,
    ) -> Optional[DiagnosticIssue]:
        skew = context.skew_pct
        if context.steps_used < int(
            self.thresholds.min_steps_for_diag
        ) or skew < float(self.thresholds.imbalance_skew_warn):
            return None
        sev = _severity(skew, self.thresholds.imbalance_skew_crit)
        return self._issue(
            kind="IMBALANCE",
            status="IMBALANCE",
            severity=sev,
            summary=(
                f"{_metric_label(context.metric)} shows "
                f"+{skew * 100.0:.1f}% cross-rank skew."
            ),
            action="Inspect per-rank workload.",
            metric=context,
            score=skew,
            evidence={"skew_pct": skew},
        )


@dataclass(frozen=True)
class CreepConfirmedRule(_BaseStepMemorySummaryRule):
    name: str = "creep_confirmed"

    def evaluate(
        self,
        context: StepMemorySummaryMetricSignals,
    ) -> Optional[DiagnosticIssue]:
        if not context.trend.confirmed:
            return None
        return self._issue(
            kind="CREEP_CONFIRMED",
            status="MEMORY CREEP",
            severity="warn",
            summary=f"{_metric_label(context.metric)} is rising across the window.",
            action="Check retained tensors or caches.",
            metric=context,
            score=context.trend.score,
            evidence={
                "overall_abs_delta_bytes": context.trend.overall_abs_delta_bytes,
                "overall_worst_growth_pct": (
                    context.trend.overall_worst_growth_pct
                ),
                "overall_median_growth_pct": (
                    context.trend.overall_median_growth_pct
                ),
                "note": _format_creep_note(context),
            },
        )


@dataclass(frozen=True)
class CreepEarlyRule(_BaseStepMemorySummaryRule):
    name: str = "creep_early"

    def evaluate(
        self,
        context: StepMemorySummaryMetricSignals,
    ) -> Optional[DiagnosticIssue]:
        if not context.trend.early or context.trend.confirmed:
            return None
        return self._issue(
            kind="CREEP_EARLY",
            status="MEMORY RISING",
            severity="info",
            summary=(
                f"{_metric_label(context.metric)} is rising from early "
                "to recent steps."
            ),
            action="Watch the next window.",
            metric=context,
            score=context.trend.score,
            evidence={
                "overall_abs_delta_bytes": context.trend.overall_abs_delta_bytes,
                "overall_worst_growth_pct": (
                    context.trend.overall_worst_growth_pct
                ),
                "overall_median_growth_pct": (
                    context.trend.overall_median_growth_pct
                ),
                "note": _format_creep_note(context),
            },
        )


DEFAULT_STEP_MEMORY_SUMMARY_RULES = (
    HighPressureRule(),
    ImbalanceRule(),
    CreepConfirmedRule(),
    CreepEarlyRule(),
)

_ISSUE_PRIORITY = {
    "HIGH_PRESSURE": 0,
    "IMBALANCE": 1,
    "CREEP_CONFIRMED": 2,
    "CREEP_EARLY": 3,
}


def sort_step_memory_summary_issues(
    issues: Sequence[DiagnosticIssue],
) -> Tuple[DiagnosticIssue, ...]:
    """
    Sort step-memory summary issues by user impact.

    Immediate pressure wins over imbalance, confirmed creep, and early rising.
    Within a kind, severity and score decide which metric becomes primary.
    """
    return tuple(
        sorted(
            issues,
            key=lambda issue: (
                _ISSUE_PRIORITY.get(issue.kind, 100),
                -severity_rank(issue.severity),
                -float(issue.score or 0.0),
                str(issue.metric or ""),
            ),
        )
    )


def run_step_memory_summary_rules(
    metric: StepMemorySummaryMetricSignals,
    *,
    rules: Sequence[
        DiagnosticRule[StepMemorySummaryMetricSignals]
    ] = DEFAULT_STEP_MEMORY_SUMMARY_RULES,
) -> Tuple[DiagnosticIssue, ...]:
    """
    Run all registered summary rules over one step-memory metric context.
    """
    out = []
    for rule in rules:
        issue = rule.evaluate(metric)
        if issue is not None:
            out.append(issue)
    return sort_step_memory_summary_issues(out)


__all__ = [
    "DEFAULT_STEP_MEMORY_SUMMARY_RULES",
    "CreepConfirmedRule",
    "CreepEarlyRule",
    "HighPressureRule",
    "ImbalanceRule",
    "run_step_memory_summary_rules",
    "sort_step_memory_summary_issues",
]
