"""
Decision-grade interpretation helpers for TraceML run comparison.

This module turns structured run-to-run diffs into a conservative verdict layer
for both human-readable compare output and future automation.

Design goals
------------
- Reuse the existing summary schema instead of inventing new diagnosis truth.
- Stay conservative: prefer "unclear" over a wrong regression/improvement call.
- Keep the output machine-readable for future CI or monitoring use.
- Keep the output semantically ranked instead of mechanically sorted by delta.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from traceml.compare.policy import (
    DEFAULT_COMPARE_POLICY,
    CompareDecisionPolicy,
    classify_memory_bytes,
    classify_memory_skew_pp,
    classify_phase_shift_pp,
    classify_step_avg_pct,
    classify_wait_share_pp,
    significance_rank,
    step_memory_status_rank,
    step_time_status_rank,
)
from traceml.utils.formatting import fmt_mem_new

_PHASE_LABELS = {
    "dataloader": "dataloader",
    "forward": "forward",
    "backward": "backward",
    "optimizer": "optimizer",
}

_COMPUTE_PHASES = {"forward", "backward", "optimizer"}


def _as_float(value: Any) -> Optional[float]:
    """
    Best-effort float conversion for optional metrics.
    """
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _as_str(value: Any) -> Optional[str]:
    """
    Best-effort string conversion for optional text fields.
    """
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text or None


def _nested_get(obj: Dict[str, Any], *keys: str) -> Any:
    """
    Safe nested dictionary access.
    """
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _metric_block(
    compare_payload: Dict[str, Any], *keys: str
) -> Dict[str, Any]:
    """
    Return a compare metric block or an empty mapping if unavailable.
    """
    value = _nested_get(compare_payload, *keys)
    return value if isinstance(value, dict) else {}


def _presented_block(summary: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Return a summary-side presented diagnosis block or an empty mapping.
    """
    block = _nested_get(summary, section, "diagnosis_presented")
    return block if isinstance(block, dict) else {}


def _format_signed_pct(value: Optional[float]) -> str:
    """
    Format a signed percent value.
    """
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.1f}%"


def _format_signed_pp(value: Optional[float]) -> str:
    """
    Format a signed percentage-point value.
    """
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.1f} pp"


def _make_change(
    *,
    importance: int,
    domain: str,
    metric: str,
    significance: str,
    summary: str,
    detail: Optional[str] = None,
    delta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build one structured top-change entry.

    `importance` is semantic rather than purely numeric.
    """
    return {
        "importance": int(importance),
        "domain": domain,
        "metric": metric,
        "significance": significance,
        "summary": summary,
        "detail": detail,
        "delta": delta,
    }


def _sort_changes(changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort change entries by semantic importance first, then significance.
    """
    return sorted(
        changes,
        key=lambda item: (
            -int(item.get("importance", 0)),
            -significance_rank(item.get("significance", "negligible")),
            str(item.get("summary", "")),
        ),
    )


def _phase_deltas(compare_payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Return available step split percentage-point deltas by phase.
    """
    split_pct = _metric_block(compare_payload, "step_time", "split_pct")
    out: Dict[str, float] = {}

    for phase, block in split_pct.items():
        if not isinstance(block, dict):
            continue
        delta = _as_float(block.get("delta"))
        if delta is None:
            continue
        out[str(phase)] = float(delta)

    return out


def _phase_rebalance(
    *,
    phase_deltas: Dict[str, float],
    policy: CompareDecisionPolicy,
) -> Optional[Dict[str, Any]]:
    """
    Summarize a phase rebalance conservatively.

    Returns None when phase changes are too small to be meaningful.
    """
    meaningful = [
        (phase, delta)
        for phase, delta in phase_deltas.items()
        if classify_phase_shift_pp(abs(delta), policy=policy) != "negligible"
    ]
    if not meaningful:
        return None

    meaningful.sort(key=lambda item: abs(item[1]), reverse=True)
    top = meaningful[:2]
    top_sig = max(
        classify_phase_shift_pp(abs(delta), policy=policy) for _, delta in top
    )

    if len(top) == 1:
        phase, delta = top[0]
        direction = "up" if delta > 0 else "down"
        return {
            "summary": (
                f"{_PHASE_LABELS.get(phase, phase).title()} share "
                f"{direction} {abs(delta):.1f} pp"
            ),
            "detail": None,
            "significance": top_sig,
        }

    detail = ", ".join(
        f"{_PHASE_LABELS.get(phase, phase)} "
        f"{'up' if delta > 0 else 'down'} {abs(delta):.1f} pp"
        for phase, delta in top
    )

    return {
        "summary": (
            "Minor compute rebalance"
            if all(phase in _COMPUTE_PHASES for phase, _ in top)
            else "Minor phase rebalance"
        ),
        "detail": detail,
        "significance": top_sig,
    }


def _supported_step_status_shift(
    *,
    rhs_status: Optional[str],
    worsened: bool,
    step_sig: str,
    wait_sig: str,
    phase_sig: str,
) -> bool:
    """
    Return True when a step-time diagnosis worsening has supporting numeric evidence.
    """
    if not worsened or not rhs_status:
        return False

    if rhs_status in {
        "WAIT-HEAVY",
        "INPUT STRAGGLER",
        "COMPUTE STRAGGLER",
        "STRAGGLER",
    }:
        return any(
            sig in {"moderate", "material"}
            for sig in (step_sig, wait_sig, phase_sig)
        )

    if rhs_status in {"INPUT-BOUND", "COMPUTE-BOUND"}:
        return step_sig == "material" or phase_sig == "material"

    return False


def _supported_memory_status_shift(
    *,
    rhs_status: Optional[str],
    worsened: bool,
    peak_sig: str,
    skew_sig: str,
    trend_sig: str,
) -> bool:
    """
    Return True when a memory diagnosis worsening has supporting memory evidence.

    Notes
    -----
    - `MEMORY CREEP (EARLY)` is treated conservatively and does not count as a
      clear regression unless the trend itself is already material.
    """
    if not worsened or not rhs_status:
        return False

    if rhs_status == "MEMORY CREEP (EARLY)":
        return trend_sig == "material"

    if rhs_status in {"HIGH PRESSURE", "IMBALANCE", "MEMORY CREEP"}:
        return any(
            sig in {"moderate", "material"}
            for sig in (peak_sig, skew_sig, trend_sig)
        )

    return False


def _why_equivalent(
    *,
    step_stable: bool,
    wait_stable: bool,
    diagnoses_stable: bool,
    dominant_phase_stable: bool,
) -> str:
    """
    Build one short explanation for an equivalent outcome.
    """
    parts: List[str] = []

    if step_stable:
        parts.append("step time unchanged")
    if wait_stable:
        parts.append("wait share unchanged")
    if diagnoses_stable:
        parts.append("diagnoses unchanged")
    if dominant_phase_stable:
        parts.append("dominant phase unchanged")

    if not parts:
        return "No strong difference in core signals"

    return ", ".join(parts[:3]).capitalize()


def _why_regression(
    *,
    step_avg_pct: Optional[float],
    wait_delta_pp: Optional[float],
    rhs_step_status: Optional[str],
    rhs_mem_status: Optional[str],
    supported_step_status: bool,
    supported_mem_status: bool,
) -> str:
    """
    Build one short explanation for a regression outcome.
    """
    if step_avg_pct is not None and step_avg_pct >= 8.0:
        return f"Step time regressed by {step_avg_pct:.1f}%"

    if wait_delta_pp is not None and wait_delta_pp >= 2.5:
        return f"Wait share increased by {wait_delta_pp:.1f} pp"

    if supported_step_status and rhs_step_status:
        return f"Step-time diagnosis worsened to {rhs_step_status}"

    if supported_mem_status and rhs_mem_status:
        return f"Step-memory diagnosis worsened to {rhs_mem_status}"

    return "Clear regression signal exceeded compare policy thresholds"


def _why_improvement(
    *,
    step_avg_pct: Optional[float],
) -> str:
    """
    Build one short explanation for an improvement outcome.
    """
    if step_avg_pct is not None and step_avg_pct <= -8.0:
        return f"Step time improved by {abs(step_avg_pct):.1f}%"

    return "Clear improvement signal exceeded compare policy thresholds"


def _why_unclear(
    *,
    step_status_changed: bool,
    mem_status_changed: bool,
    supported_step_status: bool,
    supported_mem_status: bool,
    step_sig: str,
    wait_sig: str,
    peak_sig: str,
    skew_sig: str,
    trend_sig: str,
) -> str:
    """
    Build one short explanation for an unclear outcome.
    """
    if step_status_changed and not supported_step_status:
        return "Step-time diagnosis changed without strong supporting timing movement"

    if mem_status_changed and not supported_mem_status:
        return "Step-memory diagnosis changed without strong supporting memory movement"

    if step_sig == "moderate":
        return "Step time changed moderately, but not enough for a clear conclusion"

    if wait_sig == "moderate":
        return (
            "Wait share shifted moderately without a clear regression pattern"
        )

    if any(sig == "moderate" for sig in (peak_sig, skew_sig, trend_sig)):
        return "Memory signals shifted, but not enough for a clear conclusion"

    return "Signals are mixed or incomplete"


def _recommended_action(
    *,
    outcome: str,
    rhs_step_action: Optional[str],
    rhs_mem_action: Optional[str],
    supported_step_status: bool,
    supported_mem_status: bool,
) -> str:
    """
    Choose one concise next-step recommendation.
    """
    if outcome == "regression":
        if supported_step_status and rhs_step_action:
            return rhs_step_action
        if supported_mem_status and rhs_mem_action:
            return rhs_mem_action
        return "Investigate the evidence below before treating the runs as equivalent."

    if outcome == "improvement":
        return "Treat run B as an improvement candidate and confirm with a longer run if needed."

    if outcome == "unclear":
        return (
            "Review the evidence below and treat the compare as inconclusive."
        )

    return "Treat the runs as equivalent unless external metrics disagree."


def build_compare_verdict(
    *,
    lhs_payload: Dict[str, Any],
    rhs_payload: Dict[str, Any],
    compare_payload: Dict[str, Any],
    policy: CompareDecisionPolicy = DEFAULT_COMPARE_POLICY,
) -> Dict[str, Any]:
    """
    Build a conservative decision-grade verdict layer for one compare payload.

    Outcome policy
    --------------
    - `regression`: clear supporting evidence of worse behavior
    - `improvement`: clear supporting evidence of better behavior
    - `equivalent`: core signals are stable
    - `unclear`: anything in between or conflicting

    This abstention-first policy keeps false positives and false negatives low
    for a strict v1.
    """
    lhs_step_status = _as_str(
        _nested_get(lhs_payload, "step_time", "diagnosis", "status")
    )
    rhs_step_status = _as_str(
        _nested_get(rhs_payload, "step_time", "diagnosis", "status")
    )
    lhs_mem_status = _as_str(
        _nested_get(lhs_payload, "step_memory", "diagnosis", "status")
    )
    rhs_mem_status = _as_str(
        _nested_get(rhs_payload, "step_memory", "diagnosis", "status")
    )

    rhs_step_presented = _presented_block(rhs_payload, "step_time")
    rhs_mem_presented = _presented_block(rhs_payload, "step_memory")

    step_avg = _metric_block(compare_payload, "step_time", "step_avg_ms")
    wait_share = _metric_block(compare_payload, "step_time", "wait_share_pct")
    dominant_phase = _metric_block(
        compare_payload, "step_time", "dominant_phase"
    )
    worst_peak = _metric_block(
        compare_payload, "step_memory", "worst_peak_bytes"
    )
    mem_skew = _metric_block(compare_payload, "step_memory", "skew_pct")
    mem_trend = _metric_block(
        compare_payload, "step_memory", "trend_worst_delta_bytes"
    )

    step_avg_pct = _as_float(step_avg.get("pct_change"))
    wait_delta_pp = _as_float(wait_share.get("delta"))
    worst_peak_delta = _as_float(worst_peak.get("delta"))
    mem_skew_delta = _as_float(mem_skew.get("delta"))
    mem_trend_delta = _as_float(mem_trend.get("delta"))

    step_status_changed = lhs_step_status != rhs_step_status
    mem_status_changed = lhs_mem_status != rhs_mem_status

    step_status_worsened = step_time_status_rank(
        rhs_step_status
    ) > step_time_status_rank(lhs_step_status)
    mem_status_worsened = step_memory_status_rank(
        rhs_mem_status
    ) > step_memory_status_rank(lhs_mem_status)

    step_sig = classify_step_avg_pct(
        abs(step_avg_pct) if step_avg_pct is not None else None, policy=policy
    )
    wait_sig = classify_wait_share_pp(
        abs(wait_delta_pp) if wait_delta_pp is not None else None,
        policy=policy,
    )
    peak_sig = classify_memory_bytes(
        abs(worst_peak_delta) if worst_peak_delta is not None else None,
        policy=policy,
    )
    skew_sig = classify_memory_skew_pp(
        abs(mem_skew_delta) if mem_skew_delta is not None else None,
        policy=policy,
    )
    trend_sig = classify_memory_bytes(
        abs(mem_trend_delta) if mem_trend_delta is not None else None,
        policy=policy,
    )

    phase_rebalance = _phase_rebalance(
        phase_deltas=_phase_deltas(compare_payload),
        policy=policy,
    )
    phase_sig = (
        _as_str(phase_rebalance.get("significance"))
        if isinstance(phase_rebalance, dict)
        else "negligible"
    )

    supported_step_status = _supported_step_status_shift(
        rhs_status=rhs_step_status,
        worsened=step_status_worsened,
        step_sig=step_sig,
        wait_sig=wait_sig,
        phase_sig=phase_sig,
    )
    supported_mem_status = _supported_memory_status_shift(
        rhs_status=rhs_mem_status,
        worsened=mem_status_worsened,
        peak_sig=peak_sig,
        skew_sig=skew_sig,
        trend_sig=trend_sig,
    )

    clear_regression = bool(
        (
            step_sig == "material"
            and step_avg_pct is not None
            and step_avg_pct > 0
        )
        or (
            wait_sig == "material"
            and wait_delta_pp is not None
            and wait_delta_pp > 0
        )
        or supported_step_status
        or supported_mem_status
    )

    clear_improvement = bool(
        not clear_regression
        and step_sig == "material"
        and step_avg_pct is not None
        and step_avg_pct < 0
        and not supported_step_status
        and not supported_mem_status
    )

    equivalent = bool(
        not clear_regression
        and not clear_improvement
        and step_sig == "negligible"
        and wait_sig == "negligible"
        and peak_sig == "negligible"
        and skew_sig == "negligible"
        and trend_sig == "negligible"
        and not supported_step_status
        and not supported_mem_status
    )

    if clear_regression:
        outcome = "regression"
    elif clear_improvement:
        outcome = "improvement"
    elif equivalent:
        outcome = "equivalent"
    else:
        outcome = "unclear"

    changes: List[Dict[str, Any]] = []

    if supported_step_status and rhs_step_status:
        changes.append(
            _make_change(
                importance=95,
                domain="step_time",
                metric="diagnosis.status",
                significance="material",
                summary=f"Step-time diagnosis worsened to {rhs_step_status}",
                detail=_as_str(rhs_step_presented.get("reason")),
            )
        )

    if supported_mem_status and rhs_mem_status:
        changes.append(
            _make_change(
                importance=92,
                domain="step_memory",
                metric="diagnosis.status",
                significance="material",
                summary=f"Step-memory diagnosis worsened to {rhs_mem_status}",
                detail=_as_str(rhs_mem_presented.get("reason")),
            )
        )

    if step_sig in {"moderate", "material"} and step_avg_pct is not None:
        direction = "improved" if step_avg_pct < 0 else "regressed"
        changes.append(
            _make_change(
                importance=90,
                domain="step_time",
                metric="timing_primary.step_avg_ms",
                significance=step_sig,
                summary=f"Average step time {direction}: {_format_signed_pct(step_avg_pct)}",
                detail=(
                    (
                        f"A {float(step_avg.get('lhs')):.1f} ms -> "
                        f"B {float(step_avg.get('rhs')):.1f} ms"
                    )
                    if _as_float(step_avg.get("lhs")) is not None
                    and _as_float(step_avg.get("rhs")) is not None
                    else None
                ),
                delta=step_avg,
            )
        )

    if wait_sig in {"moderate", "material"} and wait_delta_pp is not None:
        changes.append(
            _make_change(
                importance=85,
                domain="step_time",
                metric="timing_primary.wait_share_pct",
                significance=wait_sig,
                summary=(
                    "Wait share "
                    + ("increased" if wait_delta_pp > 0 else "decreased")
                    + f" by {_format_signed_pp(wait_delta_pp)}"
                ),
                delta=wait_share,
            )
        )

    if isinstance(phase_rebalance, dict):
        changes.append(
            _make_change(
                importance=70,
                domain="step_time",
                metric="timing_primary.split_pct",
                significance=_as_str(phase_rebalance.get("significance"))
                or "moderate",
                summary=_as_str(phase_rebalance.get("summary"))
                or "Phase rebalance",
                detail=_as_str(phase_rebalance.get("detail")),
            )
        )

    if peak_sig in {"moderate", "material"} and worst_peak_delta is not None:
        changes.append(
            _make_change(
                importance=65,
                domain="step_memory",
                metric="primary_metric.worst_peak_bytes",
                significance=peak_sig,
                summary=(
                    "Worst peak memory "
                    + ("increased" if worst_peak_delta > 0 else "decreased")
                    + f" by {fmt_mem_new(abs(worst_peak_delta))}"
                ),
                delta=worst_peak,
            )
        )

    if skew_sig in {"moderate", "material"} and mem_skew_delta is not None:
        changes.append(
            _make_change(
                importance=63,
                domain="step_memory",
                metric="primary_metric.skew_pct",
                significance=skew_sig,
                summary=(
                    "Memory skew "
                    + ("increased" if mem_skew_delta > 0 else "decreased")
                    + f" by {_format_signed_pp(mem_skew_delta)}"
                ),
                delta=mem_skew,
            )
        )

    if trend_sig in {"moderate", "material"} and mem_trend_delta is not None:
        changes.append(
            _make_change(
                importance=61,
                domain="step_memory",
                metric="primary_metric.trend.worst.delta_bytes",
                significance=trend_sig,
                summary=(
                    "Worst-rank memory trend "
                    + ("rose" if mem_trend_delta > 0 else "softened")
                    + f" by {fmt_mem_new(abs(mem_trend_delta))}"
                ),
                delta=mem_trend,
            )
        )

    if (
        outcome == "unclear"
        and step_status_changed
        and not supported_step_status
        and rhs_step_status
    ):
        changes.append(
            _make_change(
                importance=72,
                domain="step_time",
                metric="diagnosis.status",
                significance="moderate",
                summary="Step-time diagnosis changed without strong supporting timing movement",
                detail=f"{lhs_step_status or 'n/a'} -> {rhs_step_status}",
            )
        )

    if (
        outcome == "unclear"
        and mem_status_changed
        and not supported_mem_status
        and rhs_mem_status
    ):
        changes.append(
            _make_change(
                importance=71,
                domain="step_memory",
                metric="diagnosis.status",
                significance="moderate",
                summary="Step-memory diagnosis changed without strong supporting memory movement",
                detail=f"{lhs_mem_status or 'n/a'} -> {rhs_mem_status}",
            )
        )

    sorted_changes = _sort_changes(changes)
    top_changes = sorted_changes[:4]
    largest_shift = (
        _as_str(top_changes[0].get("summary")) if top_changes else None
    )

    diagnoses_stable = not step_status_changed and not mem_status_changed
    dominant_phase_stable = bool(
        _as_str(dominant_phase.get("lhs"))
        and dominant_phase.get("lhs") == dominant_phase.get("rhs")
    )

    if outcome == "regression":
        summary = "Material training regression detected."
        why = _why_regression(
            step_avg_pct=step_avg_pct,
            wait_delta_pp=wait_delta_pp,
            rhs_step_status=rhs_step_status,
            rhs_mem_status=rhs_mem_status,
            supported_step_status=supported_step_status,
            supported_mem_status=supported_mem_status,
        )
        severity = (
            "high"
            if step_time_status_rank(rhs_step_status) >= 3
            or step_memory_status_rank(rhs_mem_status) >= 4
            or (
                step_avg_pct is not None
                and step_avg_pct >= (policy.step_avg_pct_material * 2.0)
            )
            else "medium"
        )
    elif outcome == "improvement":
        summary = "Meaningful training improvement detected."
        why = _why_improvement(step_avg_pct=step_avg_pct)
        severity = "low"
    elif outcome == "equivalent":
        summary = "No material training regression detected."
        why = _why_equivalent(
            step_stable=(step_sig == "negligible"),
            wait_stable=(wait_sig == "negligible"),
            diagnoses_stable=diagnoses_stable,
            dominant_phase_stable=dominant_phase_stable,
        )
        severity = "info"
    else:
        summary = "No clear comparison outcome."
        why = _why_unclear(
            step_status_changed=step_status_changed,
            mem_status_changed=mem_status_changed,
            supported_step_status=supported_step_status,
            supported_mem_status=supported_mem_status,
            step_sig=step_sig,
            wait_sig=wait_sig,
            peak_sig=peak_sig,
            skew_sig=skew_sig,
            trend_sig=trend_sig,
        )
        severity = "info"

    action = _recommended_action(
        outcome=outcome,
        rhs_step_action=_as_str(rhs_step_presented.get("action")),
        rhs_mem_action=_as_str(rhs_mem_presented.get("action")),
        supported_step_status=supported_step_status,
        supported_mem_status=supported_mem_status,
    )

    return {
        "policy_version": 2,
        "outcome": outcome,
        "summary": summary,
        "why": why,
        "largest_shift": largest_shift,
        "action": action,
        "recommended_action": action,
        "severity": severity,
        "material_regression": outcome == "regression",
        "material_improvement": outcome == "improvement",
        "top_changes": top_changes,
    }
