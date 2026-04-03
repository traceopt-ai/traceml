"""
Step-memory diagnosis logic shared by live renderers and summaries.

Semantics
---------
- Input values are bytes.
- Diagnosis is conservative for confirmed creep.
- Early creep detection is intentionally lighter-weight so short live windows
  can surface meaningful upward drift before long-window confirmation exists.
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
    "CREEP_EARLY",
    "CREEP_CONFIRMED",
]

_STATUS_BY_KIND = {
    "NO_DATA": "NO DATA",
    "BALANCED": "BALANCED",
    "HIGH_PRESSURE": "HIGH PRESSURE",
    "IMBALANCE": "IMBALANCE",
    "CREEP_EARLY": "MEMORY CREEP",
    "CREEP_CONFIRMED": "CREEP CONFIRMED",
}


@dataclass(frozen=True)
class StepMemoryDiagnosisThresholds:
    """
    Thresholds for step-memory diagnosis.

    Notes
    -----
    - Confirmed creep remains strict through StepMemoryTrendHeuristics.
    - Early creep is deliberately lighter so shorter live windows can surface
      obvious upward-drift issues without waiting for long-horizon confirmation.
    """

    min_steps_for_confident_diag: int = 80

    imbalance_skew_warn: float = 0.12
    imbalance_skew_crit: float = 0.20

    pressure_warn_fraction: float = 0.92
    pressure_crit_fraction: float = 0.97

    trend: StepMemoryTrendHeuristics = DEFAULT_STEP_MEMORY_TREND_HEURISTICS

    early_creep_min_steps: int = 80
    early_creep_abs_delta_bytes_min: float = 256.0 * 1024.0 * 1024.0  # 256 MiB
    early_creep_worst_trend_pct_min: float = 0.03
    early_creep_median_trend_pct_min: float = 0.01


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


@dataclass(frozen=True)
class EarlyCreepEvidence:
    """
    Lightweight visible-window evidence for short-horizon memory creep.

    This is separate from the stricter long-window trend engine used for
    confirmed creep detection.
    """

    eligible: bool
    abs_delta_bytes: Optional[float]
    worst_trend_pct: Optional[float]
    median_trend_pct: Optional[float]


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
    4) CREEP_EARLY
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
    early_ev = _evaluate_early_creep(
        steps_used=steps_used,
        worst_series_bytes=metric.series.worst,
        median_series_bytes=metric.series.median,
        min_steps=thresholds.early_creep_min_steps,
    )

    # 1) HIGH PRESSURE
    if (
        pressure_frac is not None
        and pressure_frac >= thresholds.pressure_warn_fraction
    ):
        sev = _severity(pressure_frac, thresholds.pressure_crit_fraction)
        note = _merge_notes(
            _trend_note(metric_name=metric.metric, trend_ev=trend_ev),
            _early_trend_note(metric_name=metric.metric, early_ev=early_ev),
        )
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
        note = _merge_notes(
            _trend_note(metric_name=metric.metric, trend_ev=trend_ev),
            _early_trend_note(metric_name=metric.metric, early_ev=early_ev),
        )
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

    # 4) MEMORY CREEP (early)
    if _is_early_creep_signal(early_ev=early_ev, thresholds=thresholds):
        return _mk_diag(
            kind="CREEP_EARLY",
            severity="info",
            metric=metric.metric,
            steps_used=steps_used,
            worst_rank=worst_rank,
            reason=(
                f"{metric.metric.replace('_', ' ')} is rising in the visible window "
                "and already shows an early creep signal."
            ),
            action=(
                "Monitor continued growth; confirm with longer history before "
                "treating this as a high-confidence leak."
            ),
            note=_early_trend_note(
                metric_name=metric.metric, early_ev=early_ev
            ),
            confidence=0.55,
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
        note=_merge_notes(
            _trend_note(metric_name=metric.metric, trend_ev=trend_ev),
            _early_trend_note(metric_name=metric.metric, early_ev=early_ev),
        ),
        confidence=0.75,
    )


def _select_primary_metric(
    metrics: Sequence[StepMemoryCombinedMetric],
) -> Optional[StepMemoryCombinedMetric]:
    """
    Select the most informative memory metric for diagnosis.

    Policy
    ------
    - If both reserved and allocated exist, choose the one with the larger
      visible upward delta in the worst series.
    - This allows live diagnosis to react to the stronger creep signal instead
      of always defaulting to reserved memory.
    - Fall back to reserved, then allocated, then first available metric.
    """
    by_name = {}
    for metric in metrics:
        key = str(getattr(metric, "metric", "") or "")
        if key:
            by_name[key] = metric

    reserved = by_name.get("peak_reserved")
    allocated = by_name.get("peak_allocated")

    if reserved is not None and allocated is not None:
        reserved_delta = _worst_series_delta(reserved)
        allocated_delta = _worst_series_delta(allocated)
        if allocated_delta > reserved_delta:
            return allocated
        return reserved

    if reserved is not None:
        return reserved
    if allocated is not None:
        return allocated
    if metrics:
        return metrics[0]
    return None


def _worst_series_delta(metric: StepMemoryCombinedMetric) -> float:
    """
    Return the visible worst-series delta over the current window.
    """
    try:
        values = metric.series.worst
        if not values or len(values) < 2:
            return 0.0
        return max(0.0, float(values[-1]) - float(values[0]))
    except Exception:
        return 0.0


def _evaluate_early_creep(
    *,
    steps_used: int,
    worst_series_bytes: Sequence[float],
    median_series_bytes: Sequence[float],
    min_steps: int,
) -> EarlyCreepEvidence:
    """
    Compute lightweight early-creep evidence directly from the visible window.

    This intentionally does not depend on the strict long-window creep engine.
    """
    if int(steps_used) < int(min_steps):
        return EarlyCreepEvidence(
            eligible=False,
            abs_delta_bytes=None,
            worst_trend_pct=None,
            median_trend_pct=None,
        )

    worst = _clean_series(worst_series_bytes)
    median = _clean_series(median_series_bytes)

    if len(worst) < 2 or len(median) < 2:
        return EarlyCreepEvidence(
            eligible=False,
            abs_delta_bytes=None,
            worst_trend_pct=None,
            median_trend_pct=None,
        )

    worst_delta = float(worst[-1] - worst[0])
    worst_baseline = max(1.0, float(sum(worst) / len(worst)))
    median_baseline = max(1.0, float(sum(median) / len(median)))

    return EarlyCreepEvidence(
        eligible=True,
        abs_delta_bytes=worst_delta,
        worst_trend_pct=worst_delta / worst_baseline,
        median_trend_pct=float(median[-1] - median[0]) / median_baseline,
    )


def _clean_series(values: Sequence[float]) -> list[float]:
    out = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            number = 0.0
        out.append(max(0.0, number))
    return out


def _is_early_creep_signal(
    *,
    early_ev: EarlyCreepEvidence,
    thresholds: StepMemoryDiagnosisThresholds,
) -> bool:
    if not early_ev.eligible:
        return False
    if early_ev.abs_delta_bytes is None:
        return False
    if early_ev.worst_trend_pct is None:
        return False
    if early_ev.median_trend_pct is None:
        return False

    return bool(
        float(early_ev.abs_delta_bytes)
        >= float(thresholds.early_creep_abs_delta_bytes_min)
        and float(early_ev.worst_trend_pct)
        >= float(thresholds.early_creep_worst_trend_pct_min)
        and float(early_ev.median_trend_pct)
        >= float(thresholds.early_creep_median_trend_pct_min)
    )


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


def _early_trend_note(
    *,
    metric_name: str,
    early_ev: EarlyCreepEvidence,
) -> Optional[str]:
    if not early_ev.eligible:
        return None

    parts = []

    if early_ev.abs_delta_bytes is not None:
        parts.append(f"window_Δ={_fmt_bytes(early_ev.abs_delta_bytes)}")
    if early_ev.worst_trend_pct is not None:
        parts.append(
            f"worst_window_trend={early_ev.worst_trend_pct * 100.0:.1f}%"
        )
    if early_ev.median_trend_pct is not None:
        parts.append(
            f"median_window_trend={early_ev.median_trend_pct * 100.0:.1f}%"
        )

    if not parts:
        return None
    return f"{metric_name}: " + ", ".join(parts)


def _merge_notes(*notes: Optional[str]) -> Optional[str]:
    parts = [note for note in notes if note]
    if not parts:
        return None
    return " | ".join(parts)


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
