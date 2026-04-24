"""
Modular rules for step-time diagnosis.

Each rule evaluates the same prepared `StepTimeAnalysisContext` and emits one
atomic issue or no issue. The runner remains responsible for selecting a
primary diagnosis and building richer result payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ..framework import DiagnosticIssue, DiagnosticRule
from .context import StepTimeAnalysisContext, non_negative_finite


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
    ) -> DiagnosticIssue:
        clean_ranks: Tuple[int, ...] = tuple(
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
            ranks=clean_ranks,
        )


@dataclass(frozen=True)
class InputStragglerRule(_BaseStepTimeRule):
    """
    Detect excess dataloader burden on one rank relative to a typical rank.
    """

    name: str = "input_straggler"

    def evaluate(
        self,
        context: StepTimeAnalysisContext,
    ) -> Optional[DiagnosticIssue]:
        if context.single_rank:
            return None

        score = context.input_straggler_score
        if score < context.thresholds.input_straggler_score_warn:
            return None

        rank = context.dataloader_worst_rank
        return self._issue(
            kind="INPUT_STRAGGLER",
            status="INPUT STRAGGLER",
            severity=_severity(
                score, context.thresholds.input_straggler_score_crit
            ),
            summary=(
                f"{_rank_str(rank)} has excess dataloader burden "
                f"(~{_pct(score)} of a typical local step)."
            ),
            action=f"Inspect input loading on {_rank_str(rank)}.",
            metric="dataloader_fetch",
            phase="dataloader",
            score=score,
            share_pct=context.dataloader_share,
            skew_pct=context.dataloader_skew,
            ranks=(rank,),
        )


@dataclass(frozen=True)
class ComputeStragglerRule(_BaseStepTimeRule):
    """
    Detect excess compute burden on one rank relative to a typical rank.
    """

    name: str = "compute_straggler"

    def evaluate(
        self,
        context: StepTimeAnalysisContext,
    ) -> Optional[DiagnosticIssue]:
        if context.single_rank:
            return None

        score = context.compute_straggler_score
        if score < context.thresholds.compute_straggler_score_warn:
            return None

        rank = context.compute_worst_rank
        label = (
            context.dominant_compute.label
            if context.dominant_compute is not None
            else "Compute"
        )
        return self._issue(
            kind="COMPUTE_STRAGGLER",
            status="COMPUTE STRAGGLER",
            severity=_severity(
                score,
                context.thresholds.compute_straggler_score_crit,
            ),
            summary=(
                f"{_rank_str(rank)} has excess compute burden "
                f"(~{_pct(score)} of a typical local step)."
            ),
            action=f"Inspect {label.lower()} on {_rank_str(rank)}.",
            metric="compute",
            phase=label.lower(),
            score=score,
            share_pct=context.compute_share,
            skew_pct=context.compute_skew,
            ranks=(rank,),
        )


@dataclass(frozen=True)
class InputBoundRule(_BaseStepTimeRule):
    """
    Detect input-bound behavior when dataloader dominates and skew stays low.
    """

    name: str = "input_bound"

    def evaluate(
        self,
        context: StepTimeAnalysisContext,
    ) -> Optional[DiagnosticIssue]:
        if context.dataloader_share < context.thresholds.input_share_warn:
            return None

        if not context.single_rank and (
            context.dataloader_skew > context.thresholds.input_bound_max_skew
        ):
            return None

        return self._issue(
            kind="INPUT_BOUND",
            status="INPUT-BOUND",
            severity=_severity(
                context.dataloader_share,
                context.thresholds.input_share_crit,
            ),
            summary=f"Dataloader is {_pct(context.dataloader_share)} of the typical step.",
            action="Increase workers, prefetch, or storage throughput.",
            metric="dataloader_fetch",
            phase="dataloader",
            share_pct=context.dataloader_share,
            skew_pct=context.dataloader_skew,
            ranks=(context.dataloader_worst_rank,),
        )


@dataclass(frozen=True)
class WaitHeavyRule(_BaseStepTimeRule):
    """
    Detect windows dominated by wait rather than local work.
    """

    name: str = "wait_heavy"

    def evaluate(
        self,
        context: StepTimeAnalysisContext,
    ) -> Optional[DiagnosticIssue]:
        if context.wait_share < context.thresholds.wait_share_warn:
            return None

        return self._issue(
            kind="WAIT_HEAVY",
            status="WAIT-HEAVY",
            severity=_severity(
                context.wait_share,
                context.thresholds.wait_share_crit,
            ),
            summary=f"WAIT* is {_pct(context.wait_share)} of the typical step.",
            action="Inspect sync points, CPU stalls, or H2D copies.",
            metric="wait_proxy",
            phase="wait",
            share_pct=context.wait_share,
            ranks=(context.overall_worst_rank,),
        )


@dataclass(frozen=True)
class ComputeBoundRule(_BaseStepTimeRule):
    """
    Detect windows dominated by compute without a strong cross-rank straggler.
    """

    name: str = "compute_bound"

    def evaluate(
        self,
        context: StepTimeAnalysisContext,
    ) -> Optional[DiagnosticIssue]:
        if context.compute_share < context.thresholds.compute_bound_share_warn:
            return None
        if context.dataloader_share >= context.thresholds.input_share_warn:
            return None
        if context.wait_share >= context.thresholds.wait_share_warn:
            return None
        if not context.single_rank and (
            context.compute_skew > context.thresholds.compute_bound_max_skew
        ):
            return None

        label = (
            context.largest_compute.label
            if context.largest_compute is not None
            else "Compute"
        )
        return self._issue(
            kind="COMPUTE_BOUND",
            status="COMPUTE-BOUND",
            severity=_severity(
                context.compute_share,
                context.thresholds.compute_bound_share_crit,
            ),
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
    InputStragglerRule(),
    ComputeStragglerRule(),
    InputBoundRule(),
    WaitHeavyRule(),
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
    "ComputeStragglerRule",
    "InputBoundRule",
    "InputStragglerRule",
    "WaitHeavyRule",
    "run_step_time_rules",
]
