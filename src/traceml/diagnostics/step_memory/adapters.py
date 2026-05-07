"""
Prepared analysis context for summary-oriented step-memory diagnostics.

This module centralizes the per-metric signals used by summary rules so
multiple rules can evaluate the same aligned tail window without repeating
aggregation or trend logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from traceml.diagnostics.step_memory.policy import (
    DEFAULT_STEP_MEMORY_THRESHOLDS,
    StepMemoryDiagnosisThresholds,
)
from traceml.diagnostics.trends import compute_trend_evidence
from traceml.renderers.step_memory.schema import StepMemoryCombinedMetric


@dataclass(frozen=True)
class StepMemorySummaryTrendSignals:
    """
    Trend evidence for one summary metric.

    `early` means baseline < middle < recent. `confirmed` means the same
    rising shape has also crossed the configured absolute-growth threshold.
    """

    eligible: bool
    baseline_avg_bytes: Optional[float]
    mid_avg_bytes: Optional[float]
    recent_avg_bytes: Optional[float]
    overall_abs_delta_bytes: Optional[float]
    overall_worst_growth_pct: Optional[float]
    overall_median_growth_pct: Optional[float]
    early: bool
    confirmed: bool
    score: float


@dataclass(frozen=True)
class StepMemorySummaryMetricSignals:
    """
    Summary-oriented normalized signals for one step-memory metric.
    """

    metric: str
    device: Optional[str]
    steps_used: int
    window_size: int
    completed_step: int
    ranks_seen: int
    worst_rank: Optional[int]
    worst_peak_bytes: float
    median_peak_bytes: float
    skew_ratio: float
    skew_pct: float
    pressure_frac: Optional[float]
    trend: StepMemorySummaryTrendSignals


def _safe_non_negative(value: Optional[float]) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return max(0.0, out)


def _clean_series(values: Sequence[float]) -> list[float]:
    out = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            number = 0.0
        out.append(max(0.0, number))
    return out


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


def _build_trend_signals(
    *,
    metric: StepMemoryCombinedMetric,
    thresholds: StepMemoryDiagnosisThresholds,
) -> StepMemorySummaryTrendSignals:
    steps_used = int(metric.summary.steps_used or 0)
    if steps_used < int(thresholds.min_steps_for_diag):
        return StepMemorySummaryTrendSignals(
            eligible=False,
            baseline_avg_bytes=None,
            mid_avg_bytes=None,
            recent_avg_bytes=None,
            overall_abs_delta_bytes=None,
            overall_worst_growth_pct=None,
            overall_median_growth_pct=None,
            early=False,
            confirmed=False,
            score=0.0,
        )

    worst = _clean_series(metric.series.worst)
    median = _clean_series(metric.series.median)

    worst_ev = compute_trend_evidence(worst, config=thresholds.trend)
    median_ev = compute_trend_evidence(median, config=thresholds.trend)
    if worst_ev is None or median_ev is None:
        return StepMemorySummaryTrendSignals(
            eligible=False,
            baseline_avg_bytes=None,
            mid_avg_bytes=None,
            recent_avg_bytes=None,
            overall_abs_delta_bytes=None,
            overall_worst_growth_pct=None,
            overall_median_growth_pct=None,
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
    direction_mid_base = (worst_ev.mid_avg > worst_ev.baseline_avg) and (
        median_ev.mid_avg > median_ev.baseline_avg
    )

    direction_ok = True
    if thresholds.require_recent_gt_mid:
        direction_ok = direction_ok and direction_recent_mid
    if thresholds.require_mid_ge_baseline:
        direction_ok = direction_ok and direction_mid_base

    early = bool(direction_ok and abs_delta > 0.0)
    confirmed = bool(
        direction_ok
        and abs_delta >= float(thresholds.creep_confirmed_delta_bytes)
    )

    score = (
        max(0.0, abs_delta)
        / max(1.0, float(thresholds.creep_score_delta_scale_bytes))
        + max(0.0, float(worst_growth or 0.0)) * 10.0
        + max(0.0, float(median_growth or 0.0)) * 6.0
    )

    return StepMemorySummaryTrendSignals(
        eligible=True,
        baseline_avg_bytes=worst_ev.baseline_avg,
        mid_avg_bytes=worst_ev.mid_avg,
        recent_avg_bytes=worst_ev.recent_avg,
        overall_abs_delta_bytes=abs_delta,
        overall_worst_growth_pct=worst_growth,
        overall_median_growth_pct=median_growth,
        early=early,
        confirmed=confirmed,
        score=score,
    )


def build_step_memory_summary_signals(
    metrics: Sequence[StepMemoryCombinedMetric],
    *,
    gpu_total_bytes: Optional[float] = None,
    thresholds: StepMemoryDiagnosisThresholds = DEFAULT_STEP_MEMORY_THRESHOLDS,
) -> Dict[str, StepMemorySummaryMetricSignals]:
    """
    Build normalized per-metric signals for summary diagnostics.

    This adapter is intentionally separate from the summary rules so
    contributors can add a rule without learning the renderer schema or SQL
    loading code.
    """
    out: Dict[str, StepMemorySummaryMetricSignals] = {}

    for metric in metrics:
        out[metric.metric] = StepMemorySummaryMetricSignals(
            metric=metric.metric,
            device=metric.device,
            steps_used=int(metric.summary.steps_used or 0),
            window_size=int(metric.summary.window_size or 0),
            completed_step=int(metric.coverage.completed_step or 0),
            ranks_seen=int(metric.coverage.ranks_present or 0),
            worst_rank=metric.summary.worst_rank,
            worst_peak_bytes=_safe_non_negative(metric.summary.worst_peak),
            median_peak_bytes=_safe_non_negative(metric.summary.median_peak),
            skew_ratio=_safe_non_negative(metric.summary.skew_ratio),
            skew_pct=_safe_non_negative(metric.summary.skew_pct),
            pressure_frac=_pressure_fraction(
                _safe_non_negative(metric.summary.worst_peak),
                gpu_total_bytes,
            ),
            trend=_build_trend_signals(
                metric=metric,
                thresholds=thresholds,
            ),
        )

    return out


__all__ = [
    "StepMemorySummaryTrendSignals",
    "StepMemorySummaryMetricSignals",
    "build_step_memory_summary_signals",
]
