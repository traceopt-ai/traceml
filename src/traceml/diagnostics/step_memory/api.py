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

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence

from traceml.analytics.trends import TrendBands
from traceml.diagnostics.common import DiagnosticResult, sort_issues
from traceml.renderers.step_memory.schema import StepMemoryCombinedMetric

from ..common import BaseDiagnosis, Severity, validate_confidence
from ..trends import TrendConfig, compute_trend_evidence

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

    Memory trend / creep uses the shared trend engine and metric-specific byte
    thresholds. This keeps the trend definition centralized while preserving
    memory-specific policy in this module.
    """

    min_steps_for_diag: int = 50

    pressure_warn_fraction: float = 0.92
    pressure_crit_fraction: float = 0.97

    imbalance_skew_warn: float = 0.12
    imbalance_skew_crit: float = 0.20

    creep_watch_delta_bytes: float = 100.0 * 1024.0 * 1024.0
    creep_confirmed_delta_bytes: float = 1024.0 * 1024.0 * 1024.0

    early_overall_worst_growth_min: float = 0.02
    early_overall_median_growth_min: float = 0.01
    confirmed_overall_worst_growth_min: float = 0.05
    confirmed_overall_median_growth_min: float = 0.03

    require_recent_gt_mid: bool = True
    require_mid_ge_baseline: bool = False

    trend: TrendConfig = field(
        default_factory=lambda: TrendConfig(
            min_points=50,
            bands=TrendBands(warmup_frac=0.0),
        )
    )


DEFAULT_STEP_MEMORY_THRESHOLDS = StepMemoryDiagnosisThresholds()


def _log_step_memory_diagnostic_error(message: str, exc: Exception) -> None:
    """
    Log diagnostic enrichment failures without blocking training/reporting.

    Step-memory diagnostics are advisory. A rule or adapter failure should be
    visible to TraceML maintainers through the shared error logger, but it must
    not prevent a final summary from being produced.
    """
    try:
        from traceml.loggers.error_log import get_error_logger

        get_error_logger("StepMemoryDiagnostics").exception(
            "[TraceML] %s", message
        )
    except Exception:
        pass


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
    Memory creep evidence derived from the shared trend engine.
    """

    eligible: bool

    baseline_avg_bytes: Optional[float]
    mid_avg_bytes: Optional[float]
    recent_avg_bytes: Optional[float]

    overall_abs_delta_bytes: Optional[float]
    overall_worst_growth_pct: Optional[float]
    overall_median_growth_pct: Optional[float]
    trend_window_steps: Optional[int]
    avg_growth_bytes_per_step: Optional[float]

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


def build_step_memory_summary_diagnosis_result(
    metrics: Sequence[StepMemoryCombinedMetric],
    *,
    gpu_total_bytes: Optional[float] = None,
    per_rank: Optional[Dict[str, Any]] = None,
    thresholds: StepMemoryDiagnosisThresholds = DEFAULT_STEP_MEMORY_THRESHOLDS,
) -> DiagnosticResult[StepMemoryDiagnosis]:
    """
    Build the summary-oriented step-memory diagnosis result.

    The primary diagnosis intentionally uses the same live diagnosis engine as
    CLI/dashboard step-memory rendering. Summary-specific rules add a richer
    issue list and metric attribution for final-report JSON without changing
    live diagnosis policy.
    """
    from .adapters import build_step_memory_summary_signals
    from .rules import run_step_memory_summary_rules

    primary = build_step_memory_diagnosis(
        metrics,
        gpu_total_bytes=gpu_total_bytes,
        thresholds=thresholds,
    )

    try:
        signals = build_step_memory_summary_signals(
            metrics,
            gpu_total_bytes=gpu_total_bytes,
            thresholds=thresholds,
        )

        issues = []
        metric_attribution: Dict[str, Any] = {}
        for metric in metrics:
            signal = signals.get(metric.metric)
            if signal is None:
                continue
            issues.extend(run_step_memory_summary_rules(signal))
            metric_attribution[metric.metric] = {
                "metric": signal.metric,
                "device": signal.device,
                "steps_used": signal.steps_used,
                "window_size": signal.window_size,
                "completed_step": signal.completed_step,
                "ranks_seen": signal.ranks_seen,
                "worst_rank": signal.worst_rank,
                "worst_peak_bytes": signal.worst_peak_bytes,
                "median_peak_bytes": signal.median_peak_bytes,
                "skew_ratio": signal.skew_ratio,
                "skew_pct": signal.skew_pct,
                "pressure_frac": signal.pressure_frac,
                "trend": {
                    "eligible": signal.trend.eligible,
                    "baseline_avg_bytes": signal.trend.baseline_avg_bytes,
                    "mid_avg_bytes": signal.trend.mid_avg_bytes,
                    "recent_avg_bytes": signal.trend.recent_avg_bytes,
                    "overall_abs_delta_bytes": (
                        signal.trend.overall_abs_delta_bytes
                    ),
                    "overall_worst_growth_pct": (
                        signal.trend.overall_worst_growth_pct
                    ),
                    "overall_median_growth_pct": (
                        signal.trend.overall_median_growth_pct
                    ),
                    "early": signal.trend.early,
                    "confirmed": signal.trend.confirmed,
                    "score": signal.trend.score,
                },
            }
    except Exception as exc:
        _log_step_memory_diagnostic_error(
            "Step-memory summary diagnostic enrichment failed",
            exc,
        )
        issues = []
        metric_attribution = {}

    return DiagnosticResult(
        primary=primary,
        issues=sort_issues(issues),
        metric_attribution=metric_attribution,
        per_rank=dict(per_rank or {}),
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
    Compute memory creep evidence from the shared trend engine.

    The trend definition is common across TraceML. Memory-specific thresholds
    remain local here because memory uses bytes-based alerting semantics.
    """
    if int(steps_used) < int(thresholds.min_steps_for_diag):
        return WindowCreepEvidence(
            eligible=False,
            baseline_avg_bytes=None,
            mid_avg_bytes=None,
            recent_avg_bytes=None,
            overall_abs_delta_bytes=None,
            overall_worst_growth_pct=None,
            overall_median_growth_pct=None,
            trend_window_steps=None,
            avg_growth_bytes_per_step=None,
            early=False,
            confirmed=False,
            score=0.0,
        )

    worst = _clean_series(worst_series_bytes)
    median = _clean_series(median_series_bytes)

    worst_ev = compute_trend_evidence(worst, config=thresholds.trend)
    median_ev = compute_trend_evidence(median, config=thresholds.trend)

    if worst_ev is None or median_ev is None:
        return WindowCreepEvidence(
            eligible=False,
            baseline_avg_bytes=None,
            mid_avg_bytes=None,
            recent_avg_bytes=None,
            overall_abs_delta_bytes=None,
            overall_worst_growth_pct=None,
            overall_median_growth_pct=None,
            trend_window_steps=None,
            avg_growth_bytes_per_step=None,
            early=False,
            confirmed=False,
            score=0.0,
        )

    abs_delta = float(worst_ev.delta_vs_baseline)
    worst_growth = worst_ev.delta_pct_vs_baseline
    median_growth = median_ev.delta_pct_vs_baseline

    direction_recent_mid = (
        worst_ev.delta_vs_mid > 0.0 and median_ev.delta_vs_mid > 0.0
    )
    direction_mid_base = (worst_ev.mid_avg >= worst_ev.baseline_avg) and (
        median_ev.mid_avg >= median_ev.baseline_avg
    )

    direction_ok = True
    if thresholds.require_recent_gt_mid:
        direction_ok = direction_ok and direction_recent_mid
    if thresholds.require_mid_ge_baseline:
        direction_ok = direction_ok and direction_mid_base

    early = bool(
        direction_ok
        and abs_delta >= float(thresholds.creep_watch_delta_bytes)
        and worst_growth is not None
        and median_growth is not None
        and worst_growth >= float(thresholds.early_overall_worst_growth_min)
        and median_growth >= float(thresholds.early_overall_median_growth_min)
    )

    confirmed = bool(
        direction_ok
        and abs_delta >= float(thresholds.creep_confirmed_delta_bytes)
        and worst_growth is not None
        and median_growth is not None
        and worst_growth
        >= float(thresholds.confirmed_overall_worst_growth_min)
        and median_growth
        >= float(thresholds.confirmed_overall_median_growth_min)
    )

    score = (
        max(0.0, abs_delta)
        / max(1.0, float(thresholds.creep_watch_delta_bytes))
        + max(0.0, float(worst_growth or 0.0)) * 10.0
        + max(0.0, float(median_growth or 0.0)) * 6.0
    )
    trend_window_steps = min(len(worst), 1000)
    avg_growth_bytes_per_step = None
    if trend_window_steps >= 2:
        tail = worst[-trend_window_steps:]
        total_delta = float(tail[-1] - tail[0])
        avg_growth_bytes_per_step = total_delta / float(trend_window_steps - 1)

    return WindowCreepEvidence(
        eligible=True,
        baseline_avg_bytes=worst_ev.baseline_avg,
        mid_avg_bytes=worst_ev.mid_avg,
        recent_avg_bytes=worst_ev.recent_avg,
        overall_abs_delta_bytes=abs_delta,
        overall_worst_growth_pct=worst_growth,
        overall_median_growth_pct=median_growth,
        trend_window_steps=trend_window_steps,
        avg_growth_bytes_per_step=avg_growth_bytes_per_step,
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
    Format a compact, action-oriented creep note for CLI and dashboard display.

    Example:
    Memory creep detected: increasing by ~1.2 MiB / step over the last 1000 steps.
    """
    if not evidence.eligible:
        return None

    if (
        evidence.avg_growth_bytes_per_step is not None
        and evidence.trend_window_steps is not None
        and evidence.avg_growth_bytes_per_step > 0.0
        and evidence.trend_window_steps >= 2
    ):
        return (
            "Memory creep detected: increasing by "
            f"~{_fmt_bytes(evidence.avg_growth_bytes_per_step)} / step "
            f"over the last {int(evidence.trend_window_steps)} steps."
        )

    parts = []
    if evidence.overall_abs_delta_bytes is not None:
        parts.append(f"+{_fmt_bytes(abs(evidence.overall_abs_delta_bytes))}")
    if evidence.overall_worst_growth_pct is not None:
        parts.append(f"(~{evidence.overall_worst_growth_pct * 100.0:.0f}%)")
    if not parts:
        return None
    return "Memory creep detected: " + " ".join(parts)


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
    "build_step_memory_summary_diagnosis_result",
]
