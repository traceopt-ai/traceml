"""
Step-memory diagnosis logic shared by live renderers and summaries.

Semantics
---------
- Input values are bytes.
- Diagnosis is conservative (low false positives).
- Creep detection requires strong statistical gates.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from traceml.renderers.step_memory.schema import StepMemoryCombinedMetric

from .common import BaseDiagnosis, Severity, validate_confidence
from .step_memory_trend import (
    DEFAULT_STEP_MEMORY_TREND_HEURISTICS,
    StepMemoryTrendHeuristics,
    evaluate_step_memory_creep,
)

StepMemoryDiagnosisKind = Literal[
    "NO_DATA",
    "BALANCED",
    "HIGH_PRESSURE",
    "IMBALANCE",
    "CREEP_WATCH",
    "CREEP_CONFIRMED",
]

_STATUS_BY_KIND = {
    "NO_DATA": "NO DATA",
    "BALANCED": "BALANCED",
    "HIGH_PRESSURE": "HIGH PRESSURE",
    "IMBALANCE": "IMBALANCE",
    "CREEP_WATCH": "CREEP WATCH",
    "CREEP_CONFIRMED": "CREEP CONFIRMED",
}


@dataclass(frozen=True)
class StepMemoryDiagnosisThresholds:
    """
    Conservative thresholds for step-memory diagnosis.
    """

    min_steps_for_confident_diag: int = 80

    imbalance_skew_warn: float = 0.12
    imbalance_skew_crit: float = 0.20

    pressure_warn_fraction: float = 0.92
    pressure_crit_fraction: float = 0.97

    trend: StepMemoryTrendHeuristics = DEFAULT_STEP_MEMORY_TREND_HEURISTICS


DEFAULT_STEP_MEMORY_THRESHOLDS = StepMemoryDiagnosisThresholds()


@dataclass(frozen=True)
class StepMemoryDiagnosis(BaseDiagnosis):
    """
    Diagnosis payload for step-memory live and summary paths.
    """

    kind: StepMemoryDiagnosisKind
    metric: str
    steps_used: int
    worst_rank: Optional[int] = None
    note: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        validate_confidence(self.confidence)


def build_step_memory_diagnosis(
    metrics: Sequence[StepMemoryCombinedMetric],
    *,
    gpu_total_bytes: Optional[float] = None,
    thresholds: StepMemoryDiagnosisThresholds = DEFAULT_STEP_MEMORY_THRESHOLDS,
) -> StepMemoryDiagnosis:
    """
    Build one primary diagnosis from step-memory combined metrics.

    Priority
    --------
    1) HIGH_PRESSURE
    2) IMBALANCE
    3) CREEP_CONFIRMED
    4) CREEP_WATCH
    5) BALANCED
    """
    metric = _select_primary_metric(metrics)
    if metric is None:
        return _mk_diag(
            kind="NO_DATA",
            severity="info",
            metric="peak_reserved",
            steps_used=0,
            reason="No step-memory metric is available yet.",
            action="Wait for the first complete step window.",
            confidence=0.0,
        )

    steps_used = int(metric.summary.steps_used or 0)
    worst_rank = metric.summary.worst_rank
    skew_pct = _safe_non_negative(metric.summary.skew_pct)
    worst_peak = _safe_non_negative(metric.summary.worst_peak)

    if steps_used < int(thresholds.min_steps_for_confident_diag):
        return _mk_diag(
            kind="NO_DATA",
            severity="info",
            metric=metric.metric,
            steps_used=steps_used,
            worst_rank=worst_rank,
            reason=(
                f"Only {steps_used} aligned steps available; "
                "not enough for a stable memory diagnosis."
            ),
            action="Wait for more completed steps in the current window.",
            confidence=0.0,
        )

    pressure_frac = _pressure_fraction(worst_peak, gpu_total_bytes)
    trend_ev = evaluate_step_memory_creep(
        steps_used=steps_used,
        worst_series_bytes=metric.series.worst,
        median_series_bytes=metric.series.median,
        cfg=thresholds.trend,
    )

    # 1) HIGH PRESSURE
    if (
        pressure_frac is not None
        and pressure_frac >= thresholds.pressure_warn_fraction
    ):
        sev = _severity(pressure_frac, thresholds.pressure_crit_fraction)
        note = _trend_note(metric_name=metric.metric, trend_ev=trend_ev)
        return _mk_diag(
            kind="HIGH_PRESSURE",
            severity=sev,
            metric=metric.metric,
            steps_used=steps_used,
            worst_rank=worst_rank,
            reason=(
                f"Worst {metric.metric.replace('_', ' ')} is "
                f"{pressure_frac * 100.0:.1f}% of device memory."
            ),
            action="Reduce batch/sequence size or activate memory-saving techniques.",
            note=note,
            confidence=0.9 if sev == "crit" else 0.8,
        )

    # 2) IMBALANCE
    if skew_pct >= thresholds.imbalance_skew_warn:
        sev = _severity(skew_pct, thresholds.imbalance_skew_crit)
        note = _trend_note(metric_name=metric.metric, trend_ev=trend_ev)
        return _mk_diag(
            kind="IMBALANCE",
            severity=sev,
            metric=metric.metric,
            steps_used=steps_used,
            worst_rank=worst_rank,
            reason=(
                f"Cross-rank skew is +{skew_pct * 100.0:.1f}% "
                f"for {metric.metric.replace('_', ' ')}."
            ),
            action="Inspect data/rank partitioning and per-rank activation patterns.",
            note=note,
            confidence=0.85 if sev == "crit" else 0.75,
        )

    # 3) CREEP CONFIRMED
    if trend_ev.confirmed:
        return _mk_diag(
            kind="CREEP_CONFIRMED",
            severity="warn",
            metric=metric.metric,
            steps_used=steps_used,
            worst_rank=worst_rank,
            reason=(
                f"{metric.metric.replace('_', ' ')} shows persistent upward drift "
                "with absolute and slope confirmation."
            ),
            action="Check retained tensors/caches and allocator fragmentation patterns.",
            note=_trend_note(metric_name=metric.metric, trend_ev=trend_ev),
            confidence=0.9,
        )

    # 4) CREEP WATCH
    if trend_ev.watch:
        return _mk_diag(
            kind="CREEP_WATCH",
            severity="info",
            metric=metric.metric,
            steps_used=steps_used,
            worst_rank=worst_rank,
            reason=(
                f"{metric.metric.replace('_', ' ')} is trending up, "
                "but confirmation gates are not all satisfied yet."
            ),
            action="Continue monitoring; require sustained growth before remediation.",
            note=_trend_note(metric_name=metric.metric, trend_ev=trend_ev),
            confidence=0.6,
        )

    # 5) BALANCED
    return _mk_diag(
        kind="BALANCED",
        severity="info",
        metric=metric.metric,
        steps_used=steps_used,
        worst_rank=worst_rank,
        reason="No strong memory pressure, imbalance, or confirmed creep signal.",
        action="Keep monitoring memory trends while optimizing throughput.",
        note=_trend_note(metric_name=metric.metric, trend_ev=trend_ev),
        confidence=0.75,
    )


def _select_primary_metric(
    metrics: Sequence[StepMemoryCombinedMetric],
) -> Optional[StepMemoryCombinedMetric]:
    by_name = {}
    for m in metrics:
        key = str(getattr(m, "metric", "") or "")
        if key:
            by_name[key] = m

    if "peak_reserved" in by_name:
        return by_name["peak_reserved"]
    if "peak_allocated" in by_name:
        return by_name["peak_allocated"]
    if metrics:
        return metrics[0]
    return None


def _mk_diag(
    *,
    kind: StepMemoryDiagnosisKind,
    severity: Severity,
    metric: str,
    steps_used: int,
    reason: str,
    action: str,
    worst_rank: Optional[int] = None,
    note: Optional[str] = None,
    confidence: Optional[float] = None,
) -> StepMemoryDiagnosis:
    return StepMemoryDiagnosis(
        kind=kind,
        severity=severity,
        status=_STATUS_BY_KIND[kind],
        reason=reason,
        action=action,
        metric=metric,
        steps_used=int(steps_used),
        worst_rank=worst_rank,
        note=note,
        confidence=confidence,
    )


def _pressure_fraction(
    worst_peak_bytes: float,
    gpu_total_bytes: Optional[float],
) -> Optional[float]:
    try:
        total = float(gpu_total_bytes) if gpu_total_bytes is not None else 0.0
    except Exception:
        total = 0.0
    if total <= 0.0:
        return None
    return max(0.0, float(worst_peak_bytes) / total)


def _safe_non_negative(value: float) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if out < 0.0:
        return 0.0
    return out


def _severity(value: float, crit_threshold: float) -> Severity:
    return (
        "crit"
        if _safe_non_negative(value) >= float(crit_threshold)
        else "warn"
    )


def _trend_note(
    *,
    metric_name: str,
    trend_ev,
) -> Optional[str]:
    if trend_ev is None or not trend_ev.eligible:
        return None

    parts = []

    if trend_ev.abs_delta_bytes is not None:
        parts.append(f"Δ={_fmt_bytes(trend_ev.abs_delta_bytes)}")

    if trend_ev.worst_trend_pct is not None:
        parts.append(f"worst_trend={trend_ev.worst_trend_pct * 100.0:.1f}%")

    if trend_ev.median_trend_pct is not None:
        parts.append(f"median_trend={trend_ev.median_trend_pct * 100.0:.1f}%")

    if trend_ev.worst_slope_pct_per_100 is not None:
        parts.append(
            f"worst_slope={trend_ev.worst_slope_pct_per_100 * 100.0:.2f}%/100"
        )

    if trend_ev.median_slope_pct_per_100 is not None:
        parts.append(
            f"median_slope={trend_ev.median_slope_pct_per_100 * 100.0:.2f}%/100"
        )

    if trend_ev.weak_recovery is not None:
        parts.append(
            f"weak_recovery={'yes' if trend_ev.weak_recovery else 'no'}"
        )

    if not parts:
        return None
    return f"{metric_name}: " + ", ".join(parts)


def _fmt_bytes(v: float) -> str:
    x = abs(float(v))
    kib = 1024.0
    mib = kib * 1024.0
    gib = mib * 1024.0
    if x >= gib:
        return f"{x / gib:.2f} GiB"
    if x >= mib:
        return f"{x / mib:.1f} MiB"
    if x >= kib:
        return f"{x / kib:.1f} KiB"
    return f"{x:.0f} B"


__all__ = [
    "StepMemoryDiagnosisKind",
    "StepMemoryDiagnosisThresholds",
    "DEFAULT_STEP_MEMORY_THRESHOLDS",
    "StepMemoryDiagnosis",
    "build_step_memory_diagnosis",
]
