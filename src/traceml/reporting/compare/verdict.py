"""Rule-based verdict selection for TraceML compare."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from traceml.reporting.compare.model import CompareFinding
from traceml.reporting.compare.policy import (
    DEFAULT_COMPARE_POLICY,
    CompareDecisionPolicy,
    step_memory_status_rank,
    step_time_status_rank,
)


@dataclass(frozen=True)
class CompareVerdictContext:
    """Small view of compare data used by verdict rules."""

    sections: Dict[str, Dict[str, Any]]
    policy: CompareDecisionPolicy = DEFAULT_COMPARE_POLICY

    def section(self, name: str) -> Dict[str, Any]:
        block = self.sections.get(name)
        return block if isinstance(block, dict) else {}

    def metric(self, section: str, metric: str) -> Dict[str, Any]:
        block = self.section(section).get("metrics", {}).get(metric)
        return block if isinstance(block, dict) else {}

    def diagnosis(self, section: str) -> Dict[str, Any]:
        block = self.section(section).get("diagnosis")
        return block if isinstance(block, dict) else {}


class CompareVerdictRule(Protocol):
    """Evaluate one verdict condition."""

    def evaluate(
        self,
        context: CompareVerdictContext,
    ) -> Optional[CompareFinding]: ...


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _metric_has_both(metric: Dict[str, Any]) -> bool:
    return metric.get("lhs") is not None and metric.get("rhs") is not None


def _primary_availability(context: CompareVerdictContext) -> str:
    step = _metric_has_both(context.metric("step_time", "step_avg_ms"))
    memory = _metric_has_both(
        context.metric("step_memory", "peak_reserved_bytes")
    )
    if step and memory:
        return "comparable"
    if step or memory:
        return "partial"
    return "insufficient"


def _metric_state(metric: Dict[str, Any]) -> str:
    lhs = metric.get("lhs") is not None
    rhs = metric.get("rhs") is not None
    if lhs and rhs:
        return "comparable"
    if lhs or rhs:
        return "missing_one_side"
    return "missing_both"


def _signed_pct(metric: Dict[str, Any]) -> Optional[float]:
    return _as_float(metric.get("pct_change"))


def _status(section: Dict[str, Any], side: str) -> Optional[str]:
    value = section.get("diagnosis", {}).get(side)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class MissingPrimarySignalsRule:
    def evaluate(
        self,
        context: CompareVerdictContext,
    ) -> Optional[CompareFinding]:
        state = _primary_availability(context)
        if state != "insufficient":
            return None
        step_state = _metric_state(context.metric("step_time", "step_avg_ms"))
        memory_state = _metric_state(
            context.metric("step_memory", "peak_reserved_bytes")
        )
        if (
            step_state == "missing_one_side"
            or memory_state == "missing_one_side"
        ):
            why = (
                "Primary compare sections are not comparable; signals are "
                "missing on run A or run B."
            )
        else:
            why = "Primary compare sections are not comparable."
        return CompareFinding(
            status="INCONCLUSIVE",
            severity="info",
            priority=10,
            domain="compare",
            metric=None,
            why=why,
        )


class PartialPrimarySignalsRule:
    def evaluate(
        self,
        context: CompareVerdictContext,
    ) -> Optional[CompareFinding]:
        if _primary_availability(context) != "partial":
            return None
        return CompareFinding(
            status="INCONCLUSIVE",
            severity="info",
            priority=15,
            domain="compare",
            metric=None,
            why="Primary compare sections are only partially comparable.",
        )


class PartialDiagnosisChangeRule:
    def evaluate(
        self,
        context: CompareVerdictContext,
    ) -> Optional[CompareFinding]:
        step_diag = context.diagnosis("step_time")
        wait_state = _metric_state(
            context.metric("step_time", "wait_share_pct")
        )
        if step_diag.get("changed") and wait_state != "comparable":
            return CompareFinding(
                status="INCONCLUSIVE",
                severity="info",
                priority=16,
                domain="step_time",
                metric="diagnosis",
                why="Step time diagnosis changed, but supporting metrics are partial.",
            )
        return None


class MixedPrimarySignalsRule:
    def evaluate(
        self,
        context: CompareVerdictContext,
    ) -> Optional[CompareFinding]:
        step = _signed_pct(context.metric("step_time", "step_avg_ms"))
        memory = _signed_pct(
            context.metric("step_memory", "peak_reserved_bytes")
        )
        if step is None or memory is None:
            return None

        step_bad = step >= context.policy.step_avg_pct_moderate
        step_good = step <= -context.policy.step_avg_pct_moderate
        mem_bad = memory >= 10.0
        mem_good = memory <= -10.0

        if step_good and mem_bad:
            why = "Step time improved, but step memory worsened."
        elif step_bad and mem_good:
            why = "Step memory improved, but step time worsened."
        else:
            return None

        return CompareFinding(
            status="MIXED",
            severity="warning",
            priority=20,
            domain="compare",
            metric=None,
            why=why,
        )


class StepTimeRegressionRule:
    def evaluate(
        self,
        context: CompareVerdictContext,
    ) -> Optional[CompareFinding]:
        metric = context.metric("step_time", "step_avg_ms")
        pct = _signed_pct(metric)
        if pct is None or pct < context.policy.step_avg_pct_moderate:
            return None
        return CompareFinding(
            status="REGRESSION",
            severity="warning",
            priority=30,
            domain="step_time",
            metric="step_avg_ms",
            why=f"Step time increased by {pct:.1f}%.",
        )


class StepTimeImprovementRule:
    def evaluate(
        self,
        context: CompareVerdictContext,
    ) -> Optional[CompareFinding]:
        metric = context.metric("step_time", "step_avg_ms")
        pct = _signed_pct(metric)
        if pct is None or pct > -context.policy.step_avg_pct_moderate:
            return None
        return CompareFinding(
            status="IMPROVEMENT",
            severity="info",
            priority=40,
            domain="step_time",
            metric="step_avg_ms",
            why=f"Step time decreased by {abs(pct):.1f}%.",
        )


class StepMemoryRegressionRule:
    def evaluate(
        self,
        context: CompareVerdictContext,
    ) -> Optional[CompareFinding]:
        metric = context.metric("step_memory", "peak_reserved_bytes")
        pct = _signed_pct(metric)
        if pct is None or pct < 10.0:
            return None
        return CompareFinding(
            status="REGRESSION",
            severity="warning",
            priority=50,
            domain="step_memory",
            metric="peak_reserved_bytes",
            why=f"Step memory increased by {pct:.1f}%.",
        )


class StepMemoryImprovementRule:
    def evaluate(
        self,
        context: CompareVerdictContext,
    ) -> Optional[CompareFinding]:
        metric = context.metric("step_memory", "peak_reserved_bytes")
        pct = _signed_pct(metric)
        if pct is None or pct > -10.0:
            return None
        return CompareFinding(
            status="IMPROVEMENT",
            severity="info",
            priority=60,
            domain="step_memory",
            metric="peak_reserved_bytes",
            why=f"Step memory decreased by {abs(pct):.1f}%.",
        )


class DiagnosisRegressionRule:
    def evaluate(
        self,
        context: CompareVerdictContext,
    ) -> Optional[CompareFinding]:
        step = context.section("step_time")
        lhs_step = _status(step, "lhs")
        rhs_step = _status(step, "rhs")
        if (
            lhs_step
            and rhs_step
            and step_time_status_rank(rhs_step)
            > step_time_status_rank(lhs_step)
        ):
            return CompareFinding(
                status="REGRESSION",
                severity="warning",
                priority=70,
                domain="step_time",
                metric="diagnosis",
                why=(
                    "Step time diagnosis worsened from "
                    f"{lhs_step} to {rhs_step}."
                ),
            )

        memory = context.section("step_memory")
        lhs_mem = _status(memory, "lhs")
        rhs_mem = _status(memory, "rhs")
        if (
            lhs_mem
            and rhs_mem
            and step_memory_status_rank(rhs_mem)
            > step_memory_status_rank(lhs_mem)
        ):
            return CompareFinding(
                status="REGRESSION",
                severity="warning",
                priority=80,
                domain="step_memory",
                metric="diagnosis",
                why=(
                    "Step memory diagnosis worsened from "
                    f"{lhs_mem} to {rhs_mem}."
                ),
            )
        return None


class EquivalentPrimarySignalsRule:
    def evaluate(
        self,
        context: CompareVerdictContext,
    ) -> Optional[CompareFinding]:
        if _primary_availability(context) == "insufficient":
            return None
        return CompareFinding(
            status="EQUIVALENT",
            severity="info",
            priority=100,
            domain="compare",
            metric=None,
            why="Primary training signals stayed stable.",
        )


VERDICT_RULES: tuple[CompareVerdictRule, ...] = (
    MissingPrimarySignalsRule(),
    PartialPrimarySignalsRule(),
    PartialDiagnosisChangeRule(),
    MixedPrimarySignalsRule(),
    StepTimeRegressionRule(),
    StepTimeImprovementRule(),
    StepMemoryRegressionRule(),
    StepMemoryImprovementRule(),
    DiagnosisRegressionRule(),
    EquivalentPrimarySignalsRule(),
)


def build_compare_verdict(
    *,
    lhs_payload: Dict[str, Any],
    rhs_payload: Dict[str, Any],
    compare_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Run verdict rules and return the primary compare verdict."""
    del lhs_payload, rhs_payload

    context = CompareVerdictContext(
        sections=compare_payload.get("sections", {}),
    )
    findings = [
        finding
        for rule in VERDICT_RULES
        if (finding := rule.evaluate(context)) is not None
    ]
    findings.sort(key=lambda item: item.priority)
    primary = (
        findings[0]
        if findings
        else CompareFinding(
            status="INCONCLUSIVE",
            severity="info",
            priority=999,
            domain="compare",
            metric=None,
            why="No compare verdict could be derived.",
        )
    )

    status_to_outcome = {
        "REGRESSION": "regression",
        "IMPROVEMENT": "improvement",
        "EQUIVALENT": "equivalent",
        "MIXED": "mixed",
        "INCONCLUSIVE": "unclear",
    }
    action = (
        "Inspect the changed section before drawing strong conclusions."
        if primary.status in {"REGRESSION", "MIXED"}
        else "No immediate action is required from this comparison."
    )
    if primary.status == "INCONCLUSIVE":
        action = (
            "Re-run or compare with matching TraceML summary coverage before "
            "drawing strong conclusions."
        )

    state = _primary_availability(context)
    comparability = {
        "overall": {
            "state": state,
            "reason": primary.why if state != "comparable" else None,
        },
        "step_time": {
            "state": _metric_state(context.metric("step_time", "step_avg_ms"))
        },
        "step_memory": {
            "state": _metric_state(
                context.metric("step_memory", "peak_reserved_bytes")
            )
        },
    }
    if (
        context.diagnosis("step_time").get("changed")
        and _metric_state(context.metric("step_time", "wait_share_pct"))
        != "comparable"
        and comparability["step_time"]["state"] == "comparable"
    ):
        comparability["step_time"]["state"] = "partial"
        comparability["overall"]["state"] = "partial"
        comparability["overall"]["reason"] = primary.why

    summary = primary.status.title().replace("_", " ")
    if primary.status == "INCONCLUSIVE" and "partial" in primary.why.lower():
        summary = "Partially comparable"

    visible_findings = [
        finding
        for finding in findings
        if not (
            finding.status == "EQUIVALENT" and primary.status != "EQUIVALENT"
        )
    ]

    return {
        "status": primary.status,
        "outcome": status_to_outcome.get(primary.status, "unclear"),
        "severity": primary.severity,
        "summary": summary,
        "why": primary.why,
        "action": action,
        "primary_domain": primary.domain,
        "primary_metric": primary.metric,
        "findings": [finding.to_dict() for finding in findings],
        "comparability": comparability,
        "top_changes": [
            {
                "domain": finding.domain,
                "metric": finding.metric,
                "summary": finding.why,
                "significance": (
                    "material"
                    if finding.status in {"REGRESSION", "IMPROVEMENT"}
                    else "context"
                ),
            }
            for finding in visible_findings[:3]
        ],
        "largest_shift": primary.why,
    }


__all__ = [
    "CompareFinding",
    "CompareVerdictContext",
    "CompareVerdictRule",
    "VERDICT_RULES",
    "build_compare_verdict",
]
