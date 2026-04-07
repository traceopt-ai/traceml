"""
Step-memory diagnosis logic shared by live renderers and summaries.

Design goals
------------
- Work directly from the current aligned renderer window.
- Keep the live policy simple, explainable, and stable.
- Be conservative enough for production, while still surfacing clear drift.
- Avoid GPU-size-specific behavior by combining:
  - absolute growth
  - relative growth
  - optional device-capacity scaling when available

Diagnosis priority
------------------
1. HIGH_PRESSURE
2. IMBALANCE
3. CREEP_CONFIRMED
4. CREEP_EARLY
5. BALANCED
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from traceml.renderers.step_memory.schema import StepMemoryCombinedMetric

from .common import BaseDiagnosis, Severity, validate_confidence

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
    "CREEP_EARLY": "MEMORY CREEP (EARLY)",
    "CREEP_CONFIRMED": "MEMORY CREEP",
}


@dataclass(frozen=True)
class StepMemoryDiagnosisThresholds:
    """
    Thresholds for live step-memory diagnosis.

    Notes
    -----
    - These thresholds are designed for the current visible aligned window.
    - "Early" creep is a strong advisory.
    - "Confirmed" creep requires stronger agreement across multiple slices.
    """

    min_steps_for_diag: int = 48

    pressure_warn_fraction: float = 0.92
    pressure_crit_fraction: float = 0.97

    imbalance_skew_warn: float = 0.12
    imbalance_skew_crit: float = 0.20

    slice_fractions: tuple[float, ...] = (0.10, 0.20, 0.30)
    min_slice_points: int = 4

    early_min_positive_pairs: int = 2
    confirmed_min_positive_pairs: int = 3

    early_pair_worst_growth_min: float = 0.02
    early_pair_median_growth_min: float = 0.01
    confirmed_pair_worst_growth_min: float = 0.04
    confirmed_pair_median_growth_min: float = 0.02

    early_overall_worst_growth_min: float = 0.06
    early_overall_median_growth_min: float = 0.03
    confirmed_overall_worst_growth_min: float = 0.10
    confirmed_overall_median_growth_min: float = 0.05


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
class WindowCreepEvidence:
    """
    Creep evidence computed from the current visible aligned window.

    Fields
    ------
    positive_pairs:
        Number of slice pairs (10/90, 20/80, 30/70 style) where both median
        and worst rise enough.
    overall_*:
        Head-vs-tail growth using a stable 20% slice.
    score:
        Simple ranking score used to pick the stronger metric when both
        allocated and reserved are available.
    """

    eligible: bool
    positive_pairs: int
    total_pairs: int

    overall_abs_delta_bytes: Optional[float]
    overall_worst_growth_pct: Optional[float]
    overall_median_growth_pct: Optional[float]

    early: bool
    confirmed: bool
    score: float


@dataclass(frozen=True)
class MetricAssessment:
    """
    Internal normalized assessment for one step-memory metric.
    """

    metric: StepMemoryCombinedMetric
    steps_used: int
    worst_rank: Optional[int]
    worst_peak_bytes: float
    skew_pct: float
    pressure_frac: Optional[float]
    creep: WindowCreepEvidence


def build_step_memory_diagnosis(
    metrics: Sequence[StepMemoryCombinedMetric],
    *,
    gpu_total_bytes: Optional[float] = None,
    thresholds: StepMemoryDiagnosisThresholds = DEFAULT_STEP_MEMORY_THRESHOLDS,
) -> StepMemoryDiagnosis:
    """
    Build one primary diagnosis from step-memory combined metrics.

    This function evaluates both allocated and reserved memory metrics when
    available, then selects the strongest signal according to the diagnosis
    priority defined above.
    """
    if not metrics:
        return _mk_diag(
            kind="NO_DATA",
            severity="info",
            metric="peak_reserved",
            steps_used=0,
            reason="No step-memory data yet.",
            action="Wait for more completed steps.",
            confidence=0.0,
        )

    assessments = [
        _assess_metric(
            metric=metric,
            gpu_total_bytes=gpu_total_bytes,
            thresholds=thresholds,
        )
        for metric in metrics
    ]

    ready = [
        assessment
        for assessment in assessments
        if assessment.steps_used >= int(thresholds.min_steps_for_diag)
    ]
    if not ready:
        best = max(assessments, key=lambda item: item.steps_used)
        return _mk_diag(
            kind="NO_DATA",
            severity="info",
            metric=best.metric.metric,
            steps_used=best.steps_used,
            worst_rank=best.worst_rank,
            reason="Need more aligned steps.",
            action="Wait for a fuller window.",
            confidence=0.0,
        )

    pressure_hits = [
        item
        for item in ready
        if item.pressure_frac is not None
        and item.pressure_frac >= float(thresholds.pressure_warn_fraction)
    ]
    if pressure_hits:
        best = max(
            pressure_hits,
            key=lambda item: float(item.pressure_frac or 0.0),
        )
        sev = _severity(
            float(best.pressure_frac or 0.0),
            thresholds.pressure_crit_fraction,
        )
        return _mk_diag(
            kind="HIGH_PRESSURE",
            severity=sev,
            metric=best.metric.metric,
            steps_used=best.steps_used,
            worst_rank=best.worst_rank,
            reason=(
                f"{_metric_label(best.metric.metric)} is near device capacity "
                f"(~{float(best.pressure_frac or 0.0) * 100.0:.0f}%)."
            ),
            action="Reduce memory load.",
            confidence=0.9 if sev == "crit" else 0.8,
        )

    imbalance_hits = [
        item
        for item in ready
        if item.skew_pct >= float(thresholds.imbalance_skew_warn)
    ]
    if imbalance_hits:
        best = max(imbalance_hits, key=lambda item: item.skew_pct)
        sev = _severity(best.skew_pct, thresholds.imbalance_skew_crit)
        return _mk_diag(
            kind="IMBALANCE",
            severity=sev,
            metric=best.metric.metric,
            steps_used=best.steps_used,
            worst_rank=best.worst_rank,
            reason=(
                f"{_metric_label(best.metric.metric)} shows "
                f"+{best.skew_pct * 100.0:.1f}% cross-rank skew."
            ),
            action="Inspect per-rank workload.",
            confidence=0.85 if sev == "crit" else 0.75,
        )

    confirmed_hits = [item for item in ready if item.creep.confirmed]
    if confirmed_hits:
        best = max(confirmed_hits, key=lambda item: item.creep.score)
        return _mk_diag(
            kind="CREEP_CONFIRMED",
            severity="warn",
            metric=best.metric.metric,
            steps_used=best.steps_used,
            worst_rank=best.worst_rank,
            reason=f"{_metric_label(best.metric.metric)} is rising across the window.",
            action="Check retained tensors or caches.",
            note=_format_creep_note(best.creep),
            confidence=0.88,
        )

    early_hits = [item for item in ready if item.creep.early]
    if early_hits:
        best = max(early_hits, key=lambda item: item.creep.score)
        return _mk_diag(
            kind="CREEP_EARLY",
            severity="info",
            metric=best.metric.metric,
            steps_used=best.steps_used,
            worst_rank=best.worst_rank,
            reason=f"{_metric_label(best.metric.metric)} is trending upward.",
            action="Watch the next window.",
            note=_format_creep_note(best.creep),
            confidence=0.60,
        )

    baseline = _pick_balanced_metric(ready)
    return _mk_diag(
        kind="BALANCED",
        severity="info",
        metric=baseline.metric.metric,
        steps_used=baseline.steps_used,
        worst_rank=baseline.worst_rank,
        reason="No clear pressure, imbalance, or creep signal.",
        action="Keep monitoring.",
        confidence=0.75,
    )


def _assess_metric(
    *,
    metric: StepMemoryCombinedMetric,
    gpu_total_bytes: Optional[float],
    thresholds: StepMemoryDiagnosisThresholds,
) -> MetricAssessment:
    steps_used = int(metric.summary.steps_used or 0)
    worst_rank = metric.summary.worst_rank
    worst_peak_bytes = _safe_non_negative(metric.summary.worst_peak)
    skew_pct = _safe_non_negative(metric.summary.skew_pct)

    pressure_frac = _pressure_fraction(worst_peak_bytes, gpu_total_bytes)
    creep = _compute_window_creep_evidence(
        worst_series_bytes=metric.series.worst,
        median_series_bytes=metric.series.median,
        steps_used=steps_used,
        gpu_total_bytes=gpu_total_bytes,
        thresholds=thresholds,
    )

    return MetricAssessment(
        metric=metric,
        steps_used=steps_used,
        worst_rank=worst_rank,
        worst_peak_bytes=worst_peak_bytes,
        skew_pct=skew_pct,
        pressure_frac=pressure_frac,
        creep=creep,
    )


def _compute_window_creep_evidence(
    *,
    worst_series_bytes: Sequence[float],
    median_series_bytes: Sequence[float],
    steps_used: int,
    gpu_total_bytes: Optional[float],
    thresholds: StepMemoryDiagnosisThresholds,
) -> WindowCreepEvidence:
    """
    Compute creep evidence from the current visible window.

    The policy is intentionally simple:
    - compare head vs tail averages over multiple slice sizes
    - require both worst and median to rise
    - combine absolute and relative growth
    """
    if int(steps_used) < int(thresholds.min_steps_for_diag):
        return WindowCreepEvidence(
            eligible=False,
            positive_pairs=0,
            total_pairs=0,
            overall_abs_delta_bytes=None,
            overall_worst_growth_pct=None,
            overall_median_growth_pct=None,
            early=False,
            confirmed=False,
            score=0.0,
        )

    worst = _clean_series(worst_series_bytes)
    median = _clean_series(median_series_bytes)

    n = min(len(worst), len(median), int(steps_used))
    if n < int(thresholds.min_steps_for_diag):
        return WindowCreepEvidence(
            eligible=False,
            positive_pairs=0,
            total_pairs=0,
            overall_abs_delta_bytes=None,
            overall_worst_growth_pct=None,
            overall_median_growth_pct=None,
            early=False,
            confirmed=False,
            score=0.0,
        )

    worst = worst[-n:]
    median = median[-n:]

    positive_pairs_early = 0
    positive_pairs_confirmed = 0
    total_pairs = 0

    for frac in thresholds.slice_fractions:
        segment = max(
            int(thresholds.min_slice_points),
            int(round(n * float(frac))),
        )
        segment = min(segment, max(1, n // 2))
        if segment < 1:
            continue

        worst_head = _avg(worst[:segment])
        worst_tail = _avg(worst[-segment:])
        median_head = _avg(median[:segment])
        median_tail = _avg(median[-segment:])

        worst_growth = _growth_pct(worst_head, worst_tail)
        median_growth = _growth_pct(median_head, median_tail)

        total_pairs += 1

        if (
            worst_growth is not None
            and median_growth is not None
            and worst_growth >= float(thresholds.early_pair_worst_growth_min)
            and median_growth >= float(thresholds.early_pair_median_growth_min)
        ):
            positive_pairs_early += 1

        if (
            worst_growth is not None
            and median_growth is not None
            and worst_growth
            >= float(thresholds.confirmed_pair_worst_growth_min)
            and median_growth
            >= float(thresholds.confirmed_pair_median_growth_min)
        ):
            positive_pairs_confirmed += 1

    overall_segment = max(
        int(thresholds.min_slice_points),
        int(round(n * 0.20)),
    )
    overall_segment = min(overall_segment, max(1, n // 2))

    worst_head = _avg(worst[:overall_segment])
    worst_tail = _avg(worst[-overall_segment:])
    median_head = _avg(median[:overall_segment])
    median_tail = _avg(median[-overall_segment:])

    overall_abs_delta = worst_tail - worst_head
    overall_worst_growth = _growth_pct(worst_head, worst_tail)
    overall_median_growth = _growth_pct(median_head, median_tail)

    early_abs_min = _dynamic_abs_delta_min(
        baseline_bytes=worst_head,
        gpu_total_bytes=gpu_total_bytes,
        mode="early",
    )
    confirmed_abs_min = _dynamic_abs_delta_min(
        baseline_bytes=worst_head,
        gpu_total_bytes=gpu_total_bytes,
        mode="confirmed",
    )

    early = bool(
        total_pairs > 0
        and positive_pairs_early >= int(thresholds.early_min_positive_pairs)
        and overall_worst_growth is not None
        and overall_median_growth is not None
        and overall_abs_delta >= early_abs_min
        and overall_worst_growth
        >= float(thresholds.early_overall_worst_growth_min)
        and overall_median_growth
        >= float(thresholds.early_overall_median_growth_min)
    )

    confirmed = bool(
        total_pairs > 0
        and positive_pairs_confirmed
        >= int(thresholds.confirmed_min_positive_pairs)
        and overall_worst_growth is not None
        and overall_median_growth is not None
        and overall_abs_delta >= confirmed_abs_min
        and overall_worst_growth
        >= float(thresholds.confirmed_overall_worst_growth_min)
        and overall_median_growth
        >= float(thresholds.confirmed_overall_median_growth_min)
    )

    score = (
        float(positive_pairs_confirmed) * 2.0
        + float(positive_pairs_early)
        + max(0.0, float(overall_worst_growth or 0.0)) * 10.0
        + max(0.0, float(overall_median_growth or 0.0)) * 6.0
        + max(0.0, float(overall_abs_delta)) / max(1.0, early_abs_min)
    )

    return WindowCreepEvidence(
        eligible=True,
        positive_pairs=positive_pairs_early,
        total_pairs=total_pairs,
        overall_abs_delta_bytes=overall_abs_delta,
        overall_worst_growth_pct=overall_worst_growth,
        overall_median_growth_pct=overall_median_growth,
        early=early,
        confirmed=confirmed,
        score=score,
    )


def _pick_balanced_metric(
    assessments: Sequence[MetricAssessment],
) -> MetricAssessment:
    """
    Prefer reserved for neutral reporting, then allocated, then highest score.
    """
    by_name = {item.metric.metric: item for item in assessments}
    if "peak_reserved" in by_name:
        return by_name["peak_reserved"]
    if "peak_allocated" in by_name:
        return by_name["peak_allocated"]
    return max(assessments, key=lambda item: item.creep.score)


def _dynamic_abs_delta_min(
    *,
    baseline_bytes: float,
    gpu_total_bytes: Optional[float],
    mode: str,
) -> float:
    """
    Compute a scale-aware absolute-delta threshold.

    This makes the policy less brittle across small and large GPUs.
    """
    baseline = max(0.0, float(baseline_bytes))
    total = max(0.0, float(gpu_total_bytes or 0.0))

    if mode == "confirmed":
        candidates = [
            512.0 * 1024.0 * 1024.0,
            baseline * 0.05,
        ]
        if total > 0.0:
            candidates.append(total * 0.015)
    else:
        candidates = [
            256.0 * 1024.0 * 1024.0,
            baseline * 0.03,
        ]
        if total > 0.0:
            candidates.append(total * 0.010)

    return max(candidates)


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


def _avg(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / max(1, len(values)))


def _growth_pct(start: float, end: float) -> Optional[float]:
    base = float(start)
    if base <= 0.0:
        return None
    return (float(end) - base) / base


def _clean_series(values: Sequence[float]) -> list[float]:
    out = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            number = 0.0
        out.append(max(0.0, number))
    return out


def _format_creep_note(evidence: WindowCreepEvidence) -> Optional[str]:
    """
    Format a compact note for CLI and dashboard display.

    Example
    -------
    up ~12%, +1.4 GiB, votes 3/3
    """
    if not evidence.eligible:
        return None

    parts = []

    if evidence.overall_worst_growth_pct is not None:
        parts.append(f"up ~{evidence.overall_worst_growth_pct * 100.0:.0f}%")

    if evidence.overall_abs_delta_bytes is not None:
        parts.append(f"+{_fmt_bytes(evidence.overall_abs_delta_bytes)}")

    if evidence.total_pairs > 0:
        parts.append(
            f"votes {int(evidence.positive_pairs)}/{int(evidence.total_pairs)}"
        )

    if not parts:
        return None
    return ", ".join(parts)


def _metric_label(metric_name: str) -> str:
    return metric_name.replace("_", " ")


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
