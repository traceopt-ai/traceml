"""
Unified model-level diagnostics composer.

This module combines domain-specific diagnosis engines into one structured payload
for dashboard presentation. It is intentionally presentation-agnostic.

Current domains
---------------
- step_time
- step_memory

Design goals
------------
- Provides one stable, extendable payload for "Model Diagnostics" UI.
- Adds compact evidence metadata without duplicating renderer logic.
- Never raises to callers; always returns a usable payload.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from traceml.diagnostics.step_memory import build_step_memory_diagnosis
from traceml.diagnostics.step_time import build_step_diagnosis
from traceml.diagnostics.trends import DEFAULT_TREND_CONFIG, compute_trend_pct
from traceml.renderers.step_memory.schema import StepMemoryCombinedMetric
from traceml.renderers.step_time.schema import StepCombinedTimeMetric

Severity = str  # "info" | "warn" | "crit"


@dataclass(frozen=True)
class ModelDiagnosisItem:
    """
    One diagnosis entry in the unified model diagnostics payload.

    Fields
    ------
    evidence:
        Compact, renderer-friendly metadata for quick scanning. Intended for
        short labels such as window size, worst rank, gap, trend, or pressure.
    confidence_label:
        Human-friendly label derived from confidence score.
    """

    source: str
    title: str
    kind: str
    severity: Severity
    status: str
    reason: str
    action: str
    note: Optional[str] = None
    confidence: Optional[float] = None
    confidence_label: Optional[str] = None
    steps_used: Optional[int] = None
    worst_rank: Optional[int] = None
    evidence: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelDiagnosticsPayload:
    """
    Unified payload consumed by the Model Diagnostics dashboard card.
    """

    generated_at_s: float
    overall_severity: Severity
    items: List[ModelDiagnosisItem] = field(default_factory=list)
    status_message: str = "OK"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at_s": float(self.generated_at_s),
            "overall_severity": str(self.overall_severity),
            "status_message": str(self.status_message),
            "items": [
                {
                    "source": item.source,
                    "title": item.title,
                    "kind": item.kind,
                    "severity": item.severity,
                    "status": item.status,
                    "reason": item.reason,
                    "action": item.action,
                    "note": item.note,
                    "confidence": item.confidence,
                    "confidence_label": item.confidence_label,
                    "steps_used": item.steps_used,
                    "worst_rank": item.worst_rank,
                    "evidence": dict(item.evidence),
                }
                for item in self.items
            ],
        }


def build_model_diagnostics_payload(
    *,
    step_time_metrics: Sequence[StepCombinedTimeMetric],
    step_memory_metrics: Sequence[StepMemoryCombinedMetric],
    gpu_total_bytes: Optional[float] = None,
) -> ModelDiagnosticsPayload:
    """
    Build one combined model diagnostics payload from step-time and step-memory inputs.
    """
    items: List[ModelDiagnosisItem] = []

    try:
        step_time_diag = build_step_diagnosis(step_time_metrics)
        items.append(
            ModelDiagnosisItem(
                source="step_time",
                title="Step Time",
                kind=str(step_time_diag.kind),
                severity=str(step_time_diag.severity),
                status=str(step_time_diag.status),
                reason=str(step_time_diag.reason),
                action=str(step_time_diag.action),
                note=getattr(step_time_diag, "note", None),
                confidence=getattr(step_time_diag, "confidence", None),
                confidence_label=_confidence_label(
                    getattr(step_time_diag, "confidence", None)
                ),
                steps_used=getattr(step_time_diag, "steps_used", None),
                worst_rank=getattr(step_time_diag, "worst_rank", None),
                evidence=_build_step_time_evidence(step_time_metrics),
            )
        )
    except Exception:
        items.append(
            ModelDiagnosisItem(
                source="step_time",
                title="Step Time",
                kind="NO_DATA",
                severity="info",
                status="NO DATA",
                reason="Step-time diagnosis is unavailable on this tick.",
                action="Wait for more complete samples.",
            )
        )

    try:
        step_memory_diag = build_step_memory_diagnosis(
            step_memory_metrics,
            gpu_total_bytes=gpu_total_bytes,
        )
        items.append(
            ModelDiagnosisItem(
                source="step_memory",
                title="Step Memory",
                kind=str(step_memory_diag.kind),
                severity=str(step_memory_diag.severity),
                status=str(step_memory_diag.status),
                reason=str(step_memory_diag.reason),
                action=str(step_memory_diag.action),
                note=getattr(step_memory_diag, "note", None),
                confidence=getattr(step_memory_diag, "confidence", None),
                confidence_label=_confidence_label(
                    getattr(step_memory_diag, "confidence", None)
                ),
                steps_used=getattr(step_memory_diag, "steps_used", None),
                worst_rank=getattr(step_memory_diag, "worst_rank", None),
                evidence=_build_step_memory_evidence(
                    step_memory_metrics,
                    gpu_total_bytes=gpu_total_bytes,
                ),
            )
        )
    except Exception:
        items.append(
            ModelDiagnosisItem(
                source="step_memory",
                title="Step Memory",
                kind="NO_DATA",
                severity="info",
                status="NO DATA",
                reason="Step-memory diagnosis is unavailable on this tick.",
                action="Wait for more complete samples.",
            )
        )

    overall = (
        _max_severity([item.severity for item in items]) if items else "info"
    )
    status = "OK" if items else "NO DATA"

    return ModelDiagnosticsPayload(
        generated_at_s=time.time(),
        overall_severity=overall,
        items=items,
        status_message=status,
    )


def _build_step_time_evidence(
    metrics: Sequence[StepCombinedTimeMetric],
) -> Dict[str, str]:
    """
    Build compact evidence fields for the step-time diagnosis card.
    """
    by_key = {metric.metric: metric for metric in metrics}
    step = by_key.get("step_time")
    wait = by_key.get("wait_proxy")

    if step is None:
        return {}

    evidence: Dict[str, str] = {}

    try:
        evidence["window"] = str(int(step.summary.steps_used))
    except Exception:
        pass

    try:
        if step.summary.worst_rank is not None:
            evidence["worst"] = f"r{int(step.summary.worst_rank)}"
    except Exception:
        pass

    try:
        evidence["gap"] = f"{float(step.summary.skew_pct or 0.0) * 100.0:.1f}%"
    except Exception:
        pass

    try:
        median_total = float(step.summary.median_total or 0.0)
        wait_total = (
            float(wait.summary.median_total or 0.0)
            if wait is not None
            else 0.0
        )
        if median_total > 0.0:
            evidence["wait"] = f"{(wait_total / median_total) * 100.0:.1f}%"
    except Exception:
        pass

    dominant = _dominant_step_component(by_key)
    if dominant is not None:
        evidence["dominant"] = dominant

    return evidence


def _build_step_memory_evidence(
    metrics: Sequence[StepMemoryCombinedMetric],
    *,
    gpu_total_bytes: Optional[float] = None,
) -> Dict[str, str]:
    """
    Build compact evidence fields for the step-memory diagnosis card.
    """
    metric = _select_memory_metric(metrics)
    if metric is None:
        return {}

    evidence: Dict[str, str] = {}

    try:
        evidence["window"] = str(int(metric.summary.steps_used))
    except Exception:
        pass

    try:
        if metric.summary.worst_rank is not None:
            evidence["worst"] = f"r{int(metric.summary.worst_rank)}"
    except Exception:
        pass

    try:
        evidence["imb"] = (
            f"{float(metric.summary.skew_pct or 0.0) * 100.0:.1f}%"
        )
    except Exception:
        pass

    try:
        evidence["trend"] = (
            f"{_series_trend_pct(metric.series.worst) * 100.0:.1f}%"
        )
    except Exception:
        pass

    try:
        total = float(gpu_total_bytes) if gpu_total_bytes is not None else 0.0
        peak = float(metric.summary.worst_peak or 0.0)
        if total > 0.0:
            evidence["pressure"] = f"{(peak / total) * 100.0:.1f}%"
        else:
            evidence["pressure"] = "n/a"
    except Exception:
        pass

    return evidence


def _dominant_step_component(
    by_key: Dict[str, StepCombinedTimeMetric],
) -> Optional[str]:
    """
    Return the dominant non-total median split component for step time.
    """
    labels = {
        "dataloader_fetch": "dataloader",
        "forward": "forward",
        "backward": "backward",
        "optimizer_step": "optimizer",
        "wait_proxy": "wait",
    }

    best_label: Optional[str] = None
    best_value = -1.0

    for key, label in labels.items():
        metric = by_key.get(key)
        if metric is None:
            continue
        try:
            value = float(metric.summary.median_total or 0.0)
        except Exception:
            value = 0.0
        if value > best_value:
            best_value = value
            best_label = label

    return best_label


def _select_memory_metric(
    metrics: Sequence[StepMemoryCombinedMetric],
) -> Optional[StepMemoryCombinedMetric]:
    """
    Prefer reserved memory as the operational signal, then allocated memory.
    """
    by_key = {metric.metric: metric for metric in metrics}
    if "peak_reserved" in by_key:
        return by_key["peak_reserved"]
    if "peak_allocated" in by_key:
        return by_key["peak_allocated"]
    if metrics:
        return metrics[0]
    return None


def _series_trend_pct(values: Optional[Sequence[float]]) -> float:
    """
    Compute canonical trend percentage for dashboard metadata.

    This deliberately reuses the shared trend engine so dashboard evidence,
    live diagnosis, and summaries stay consistent.
    """
    if not values:
        return 0.0

    pct = compute_trend_pct(values, config=DEFAULT_TREND_CONFIG)
    return float(pct) if pct is not None else 0.0


def _confidence_label(confidence: Optional[float]) -> Optional[str]:
    """
    Convert a numeric confidence to a short human-readable label.
    """
    if confidence is None:
        return None
    try:
        value = float(confidence)
    except Exception:
        return None

    if value >= 0.85:
        return "high"
    if value >= 0.60:
        return "medium"
    return "low"


def _max_severity(values: Sequence[str]) -> Severity:
    rank = {"info": 0, "warn": 1, "crit": 2}
    best = "info"
    best_rank = -1
    for value in values:
        score = rank.get(str(value), -1)
        if score > best_rank:
            best_rank = score
            best = str(value)
    return best


__all__ = [
    "ModelDiagnosisItem",
    "ModelDiagnosticsPayload",
    "build_model_diagnostics_payload",
]
