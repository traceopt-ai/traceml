# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Prepared analysis context for summary-oriented step-memory diagnostics.

This module centralizes the per-metric signals used by summary rules so
multiple rules can evaluate the same aligned tail window without repeating
aggregation or trend logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from traceml.renderers.step_memory.schema import StepMemoryCombinedMetric

from .policy import (
    DEFAULT_STEP_MEMORY_THRESHOLDS,
    StepMemoryDiagnosisThresholds,
)
from .trend import (
    StepMemoryWindowCreepEvidence,
    evaluate_step_memory_window_creep,
)


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
    evidence = evaluate_step_memory_window_creep(
        worst_series_bytes=metric.series.worst,
        median_series_bytes=metric.series.median,
        steps_used=int(metric.summary.steps_used or 0),
        thresholds=thresholds,
    )
    return _trend_evidence_to_summary_signals(evidence)


def _trend_evidence_to_summary_signals(
    evidence: StepMemoryWindowCreepEvidence,
) -> StepMemorySummaryTrendSignals:
    """Convert shared creep evidence into summary-rule signals."""
    return StepMemorySummaryTrendSignals(
        eligible=evidence.eligible,
        baseline_avg_bytes=evidence.baseline_avg_bytes,
        mid_avg_bytes=evidence.mid_avg_bytes,
        recent_avg_bytes=evidence.recent_avg_bytes,
        overall_abs_delta_bytes=evidence.overall_abs_delta_bytes,
        overall_worst_growth_pct=evidence.overall_worst_growth_pct,
        overall_median_growth_pct=evidence.overall_median_growth_pct,
        early=evidence.early,
        confirmed=evidence.confirmed,
        score=evidence.score,
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
