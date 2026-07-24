"""
Modular rules for step-time diagnosis.

Each rule evaluates the same prepared `StepTimeAnalysisContext` and emits one
atomic issue or no issue. The runner remains responsible for selecting a
primary diagnosis and building richer result payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from ..common import DiagnosticIssue, DiagnosticRule
from .context import (
    StepTimeAnalysisContext,
    metric_skew,
    metric_total,
    metric_worst_rank,
    non_negative_finite,
)


def _severity(value: float, crit_threshold: float) -> str:
    """
    Map a scalar signal to warn or crit severity.
    """
    return "crit" if non_negative_finite(value) >= crit_threshold else "warn"


def _pct(value: float) -> str:
    """
    Format a ratio as a percentage string.
    """
    return f"{non_negative_finite(value) * 100.0:.1f}%"


def _rank_str(rank: Optional[int]) -> str:
    """
    Format a rank identifier for user-facing text.
    """
    return f"r{rank}" if rank is not None else "—"


@dataclass(frozen=True)
class _BaseStepTimeRule(DiagnosticRule[StepTimeAnalysisContext]):
    """
    Small shared base for step-time rules.
    """

    name: str

    def _issue(
        self,
        *,
        kind: str,
        status: str,
        severity: str,
        summary: str,
        action: str,
        metric: Optional[str] = None,
        phase: Optional[str] = None,
        score: Optional[float] = None,
        share_pct: Optional[float] = None,
        skew_pct: Optional[float] = None,
        ranks: Sequence[Optional[int]] = (),
        evidence: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticIssue:
        issue_ranks: Tuple[int, ...] = tuple(
            int(rank) for rank in ranks if rank is not None
        )
        return DiagnosticIssue(
            kind=kind,
            status=status,
            severity=severity,
            summary=summary,
            action=action,
            metric=metric,
            phase=phase,
            score=non_negative_finite(score) if score is not None else None,
            share_pct=(
                non_negative_finite(share_pct)
                if share_pct is not None
                else None
            ),
            skew_pct=(
                non_negative_finite(skew_pct) if skew_pct is not None else None
            ),
            ranks=issue_ranks,
            evidence=evidence or {},
        )


@dataclass(frozen=True)
class RankStragglerRule(_BaseStepTimeRule):
    """
    Detect visible rank stragglers and classify the likely culprit cause.
    """

    name: str = "rank_straggler"

    def evaluate(
        self,
        context: StepTimeAnalysisContext,
    ) -> Optional[DiagnosticIssue]:
        if context.single_rank:
            return None

        evidence = context.rank_straggler
        if evidence is None:
            return None

        rank = evidence.culprit_rank
        component_label = {
            "input": "input wait",
            "compute": "forward compute",
            "h2d": "H2D",
            "sync_or_unattributed": "sync or unattributed work",
        }.get(evidence.component, evidence.component)
        if evidence.kind == "STRAGGLER":
            explanation = (
                "no input or H2D excess explains it"
                if context.training_strategy == "fsdp"
                else "no input, H2D, or DDP-forward excess explains it"
            )
            summary = (
                f"{_rank_str(rank)} appears to be the culprit rank "
                f"(~{_pct(evidence.score)} victim wait cost); "
                f"{explanation}."
            )
            action = (
                "Inspect synchronization, collectives, and unattributed work "
                f"around {_rank_str(rank)}."
            )
        else:
            summary = (
                f"{_rank_str(rank)} has excess {component_label} burden "
                f"explaining ~{_pct(evidence.score)} victim wait cost."
            )
            action = f"Inspect {component_label} on {_rank_str(rank)}."

        return self._issue(
            kind=evidence.kind,
            status=evidence.status,
            severity=_severity(
                evidence.score,
                context.thresholds.straggler_score_crit,
            ),
            summary=summary,
            action=action,
            metric=evidence.metric,
            phase=evidence.phase,
            score=evidence.score,
            skew_pct=evidence.score,
            ranks=(rank,),
            evidence={
                "component": evidence.component,
                "culprit_rank": evidence.culprit_rank,
                "victim_rank": evidence.victim_rank,
                "visible_metric": evidence.visible_metric,
                "visible_culprit_ms": evidence.visible_culprit_ms,
                "visible_victim_ms": evidence.visible_victim_ms,
                "visible_cost_ms": evidence.visible_cost_ms,
                "iteration_time_ms": evidence.iteration_time_ms,
                "component_excesses_ms": evidence.component_excesses_ms,
            },
        )


@dataclass(frozen=True)
class InputBoundRule(_BaseStepTimeRule):
    """
    Detect input-bound behavior from explicit input wait clocks.
    """

    name: str = "input_bound"

    def evaluate(
        self,
        context: StepTimeAnalysisContext,
    ) -> Optional[DiagnosticIssue]:
        if context.input_bound_share < context.thresholds.overhead_share_warn:
            return None

        return self._issue(
            kind="INPUT_BOUND",
            status="INPUT-BOUND",
            severity=_severity(
                context.input_bound_share,
                context.thresholds.overhead_share_crit,
            ),
            summary=(
                f"Input wait is {_pct(context.input_bound_share)} of the "
                f"typical {context.diagnosis_clock} iteration time."
            ),
            action="Increase workers, prefetch, or storage throughput.",
            metric="input_wait",
            phase="input",
            share_pct=context.input_bound_share,
            skew_pct=context.input_bound_skew,
            ranks=(context.input_bound_worst_rank,),
            evidence={
                "input_wait_ms": context.input_wait_total,
                "step_time_ms": context.input_bound_step_total,
                "iteration_time_ms": context.iteration_time_total,
                "input_bound_share": context.input_bound_share,
                "diagnosis_clock": context.diagnosis_clock,
            },
        )


@dataclass(frozen=True)
class H2DBoundRule(_BaseStepTimeRule):
    """Detect broad H2D transfer cost from GPU-selected timing."""

    name: str = "h2d_bound"

    def evaluate(
        self,
        context: StepTimeAnalysisContext,
    ) -> Optional[DiagnosticIssue]:
        if context.diagnosis_clock != "gpu":
            return None
        if context.h2d_share < context.thresholds.overhead_share_warn:
            return None
        worst_rank = metric_worst_rank(context.h2d_metric)

        return self._issue(
            kind="H2D_BOUND",
            status="H2D-BOUND",
            severity=_severity(
                context.h2d_share,
                context.thresholds.overhead_share_crit,
            ),
            summary=(
                f"H2D transfer is {_pct(context.h2d_share)} of the "
                "typical GPU iteration time."
            ),
            action=(
                "Inspect pinned memory, batch transfer, and host-to-device "
                "copies."
            ),
            metric="h2d",
            phase="h2d",
            share_pct=context.h2d_share,
            skew_pct=metric_skew(
                context.h2d_metric,
                single_rank=context.single_rank,
            ),
            ranks=(worst_rank,) if worst_rank is not None else (),
            evidence={
                "h2d_ms": metric_total(
                    context.h2d_metric,
                    single_rank=context.single_rank,
                ),
                "h2d_share": context.h2d_share,
                "diagnosis_clock": context.diagnosis_clock,
            },
        )


@dataclass(frozen=True)
class ResidualHeavyRule(_BaseStepTimeRule):
    """
    Detect windows dominated by residual time rather than traced local work.
    """

    name: str = "residual_heavy"

    def evaluate(
        self,
        context: StepTimeAnalysisContext,
    ) -> Optional[DiagnosticIssue]:
        if context.residual_share < context.thresholds.overhead_share_warn:
            return None

        return self._issue(
            kind="RESIDUAL_HEAVY",
            status="RESIDUAL-HEAVY",
            severity=_severity(
                context.residual_share,
                context.thresholds.overhead_share_crit,
            ),
            summary=(
                f"Residual time is {_pct(context.residual_share)} of the "
                "typical step."
            ),
            action=(
                "Inspect work outside traced phases, CPU stalls, logging, "
                "checkpointing, validation, or unobserved transfers."
            ),
            metric="residual_proxy",
            phase="residual",
            share_pct=context.residual_share,
            ranks=(context.overall_worst_rank,),
        )


@dataclass(frozen=True)
class ComputeBoundRule(_BaseStepTimeRule):
    """
    Report dominant compute as informational context.
    """

    name: str = "compute_bound"

    def evaluate(
        self,
        context: StepTimeAnalysisContext,
    ) -> Optional[DiagnosticIssue]:
        if context.compute_share < context.thresholds.compute_bound_share_warn:
            return None
        if context.input_bound_share >= context.thresholds.overhead_share_warn:
            return None
        if (
            context.diagnosis_clock == "gpu"
            and context.h2d_share >= context.thresholds.overhead_share_warn
        ):
            return None
        if context.residual_share >= context.thresholds.overhead_share_warn:
            return None

        label = (
            context.largest_compute.label
            if context.largest_compute is not None
            else "Compute"
        )
        return self._issue(
            kind="COMPUTE_BOUND",
            status="COMPUTE-BOUND",
            severity="info",
            summary=f"Compute-bound; {label.lower()} is the largest phase.",
            action="Optimize model compute or reduce step cost.",
            metric="compute",
            phase=label.lower(),
            share_pct=context.compute_share,
            skew_pct=context.compute_skew,
            ranks=(context.overall_worst_rank,),
        )


DEFAULT_STEP_TIME_RULES: Tuple[
    DiagnosticRule[StepTimeAnalysisContext], ...
] = (
    RankStragglerRule(),
    InputBoundRule(),
    H2DBoundRule(),
    ResidualHeavyRule(),
    ComputeBoundRule(),
)


def run_step_time_rules(
    context: StepTimeAnalysisContext,
    *,
    rules: Sequence[
        DiagnosticRule[StepTimeAnalysisContext]
    ] = DEFAULT_STEP_TIME_RULES,
) -> Tuple[DiagnosticIssue, ...]:
    """
    Run all registered step-time rules over one shared analysis context.
    """
    out: list[DiagnosticIssue] = []
    for rule in rules:
        issue = rule.evaluate(context)
        if issue is not None:
            out.append(issue)
    return tuple(out)


__all__ = [
    "DEFAULT_STEP_TIME_RULES",
    "ComputeBoundRule",
    "H2DBoundRule",
    "InputBoundRule",
    "RankStragglerRule",
    "ResidualHeavyRule",
    "run_step_time_rules",
]
