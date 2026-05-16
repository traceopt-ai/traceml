# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Conservative trend heuristics for step-memory diagnostics.

This module focuses on low false positives:
- strong data sufficiency gates
- absolute + relative growth checks
- slope confirmation over long windows
- optional weak-recovery check (for fragmentation-like creep)

All functions never raise to callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from traceml.diagnostics.trends import compute_trend_evidence

from .policy import StepMemoryDiagnosisThresholds


@dataclass(frozen=True)
class StepMemoryTrendHeuristics:
    """
    Conservative gates for creep detection.

    Notes
    -----
    - Thresholds are intentionally strict to reduce false positives.
    - Percent values are ratios (0.06 == 6%).
    """

    short_window: int = 100
    long_window: int = 400

    min_steps_for_creep: int = 800
    abs_delta_bytes_min: float = 512.0 * 1024.0 * 1024.0  # 512 MiB

    worst_trend_pct_min: float = 0.06
    median_trend_pct_min: float = 0.04

    worst_slope_pct_per_100_min: float = 0.015
    median_slope_pct_per_100_min: float = 0.010

    require_weak_recovery: bool = True
    weak_recovery_pullback_max: float = 0.02


DEFAULT_STEP_MEMORY_TREND_HEURISTICS = StepMemoryTrendHeuristics()


@dataclass(frozen=True)
class StepMemoryTrendEvidence:
    """
    Structured evidence for trend-based creep detection.
    """

    eligible: bool
    watch: bool
    confirmed: bool

    worst_trend_pct: Optional[float]
    median_trend_pct: Optional[float]

    worst_slope_pct_per_100: Optional[float]
    median_slope_pct_per_100: Optional[float]

    abs_delta_bytes: Optional[float]
    weak_recovery: Optional[bool]


@dataclass(frozen=True)
class StepMemoryWindowCreepEvidence:
    """
    Shared creep evidence for worst/median memory bands.

    The score is only a ranking signal between already-eligible metrics. The
    actual early/confirmed decisions come from the direction and absolute
    growth gates.
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


def evaluate_step_memory_creep(
    *,
    steps_used: int,
    worst_series_bytes: Sequence[float],
    median_series_bytes: Sequence[float],
    cfg: StepMemoryTrendHeuristics = DEFAULT_STEP_MEMORY_TREND_HEURISTICS,
) -> StepMemoryTrendEvidence:
    """
    Evaluate conservative creep signals from worst/median memory series.
    """
    try:
        worst = _as_non_negative_array(worst_series_bytes)
        median = _as_non_negative_array(median_series_bytes)

        long_n = max(2, int(cfg.long_window))
        short_n = max(1, min(int(cfg.short_window), long_n))

        enough_points = len(worst) >= long_n and len(median) >= long_n
        eligible = bool(
            enough_points and int(steps_used) >= int(cfg.min_steps_for_creep)
        )
        if not eligible:
            return StepMemoryTrendEvidence(
                eligible=False,
                watch=False,
                confirmed=False,
                worst_trend_pct=None,
                median_trend_pct=None,
                worst_slope_pct_per_100=None,
                median_slope_pct_per_100=None,
                abs_delta_bytes=None,
                weak_recovery=None,
            )

        worst_tail = worst[-long_n:]
        median_tail = median[-long_n:]

        worst_trend = _trend_pct(worst_tail, short_n=short_n)
        median_trend = _trend_pct(median_tail, short_n=short_n)

        worst_slope = _slope_pct_per_100(worst_tail)
        median_slope = _slope_pct_per_100(median_tail)

        abs_delta = float(worst_tail[-1] - worst_tail[0])

        weak_recovery = _weak_recovery(
            worst_tail, pullback_max=float(cfg.weak_recovery_pullback_max)
        )

        gate_abs = abs_delta >= float(cfg.abs_delta_bytes_min)
        gate_trend = (
            worst_trend is not None
            and median_trend is not None
            and worst_trend >= float(cfg.worst_trend_pct_min)
            and median_trend >= float(cfg.median_trend_pct_min)
        )
        gate_slope = (
            worst_slope is not None
            and median_slope is not None
            and worst_slope >= float(cfg.worst_slope_pct_per_100_min)
            and median_slope >= float(cfg.median_slope_pct_per_100_min)
        )

        if cfg.require_weak_recovery:
            gate_recovery = weak_recovery is True
        else:
            gate_recovery = True

        watch = bool(gate_abs and (gate_trend or gate_slope))
        confirmed = bool(
            gate_abs and gate_trend and gate_slope and gate_recovery
        )

        return StepMemoryTrendEvidence(
            eligible=True,
            watch=watch,
            confirmed=confirmed,
            worst_trend_pct=worst_trend,
            median_trend_pct=median_trend,
            worst_slope_pct_per_100=worst_slope,
            median_slope_pct_per_100=median_slope,
            abs_delta_bytes=abs_delta,
            weak_recovery=weak_recovery,
        )
    except Exception:
        return StepMemoryTrendEvidence(
            eligible=False,
            watch=False,
            confirmed=False,
            worst_trend_pct=None,
            median_trend_pct=None,
            worst_slope_pct_per_100=None,
            median_slope_pct_per_100=None,
            abs_delta_bytes=None,
            weak_recovery=None,
        )


def evaluate_step_memory_window_creep(
    *,
    worst_series_bytes: Sequence[float],
    median_series_bytes: Sequence[float],
    steps_used: int,
    thresholds: StepMemoryDiagnosisThresholds,
) -> StepMemoryWindowCreepEvidence:
    """
    Evaluate summary-window creep evidence for one memory metric.

    Both live and final-summary diagnosis use this helper so the direction
    gates and score formula cannot drift between paths.
    """
    if int(steps_used) < int(thresholds.min_steps_for_diag):
        return _empty_window_creep_evidence()

    worst = _as_non_negative_list(worst_series_bytes)
    median = _as_non_negative_list(median_series_bytes)

    worst_ev = compute_trend_evidence(worst, config=thresholds.trend)
    median_ev = compute_trend_evidence(median, config=thresholds.trend)
    if worst_ev is None or median_ev is None:
        return _empty_window_creep_evidence()

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

    trend_window_steps = min(len(worst), 1000)
    avg_growth_bytes_per_step = None
    if trend_window_steps >= 2:
        tail = worst[-trend_window_steps:]
        total_delta = float(tail[-1] - tail[0])
        avg_growth_bytes_per_step = total_delta / float(trend_window_steps - 1)

    return StepMemoryWindowCreepEvidence(
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


def _empty_window_creep_evidence() -> StepMemoryWindowCreepEvidence:
    return StepMemoryWindowCreepEvidence(
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


def _as_non_negative_array(values: Sequence[float]) -> np.ndarray:
    out = []
    for v in values:
        try:
            x = float(v)
        except Exception:
            x = 0.0
        if not np.isfinite(x):
            x = 0.0
        out.append(max(0.0, x))
    if not out:
        return np.asarray([], dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def _as_non_negative_list(values: Sequence[float]) -> list[float]:
    out = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            number = 0.0
        if not np.isfinite(number):
            number = 0.0
        out.append(max(0.0, number))
    return out


def _trend_pct(arr: np.ndarray, *, short_n: int) -> Optional[float]:
    if arr.size < 2:
        return None
    n = arr.size
    short_n = max(1, min(int(short_n), n))
    short_avg = float(np.mean(arr[-short_n:]))
    long_avg = float(np.mean(arr))
    if long_avg <= 0.0:
        return None
    return (short_avg - long_avg) / long_avg


def _slope_pct_per_100(arr: np.ndarray) -> Optional[float]:
    if arr.size < 3:
        return None
    baseline = float(np.median(arr))
    if baseline <= 0.0:
        return None

    x = np.arange(arr.size, dtype=np.float64)
    try:
        slope, _ = np.polyfit(x, arr, deg=1)
    except Exception:
        return None
    return float((slope * 100.0) / baseline)


def _weak_recovery(arr: np.ndarray, *, pullback_max: float) -> Optional[bool]:
    if arr.size < 16:
        return None

    peak_idx = int(np.argmax(arr))
    if peak_idx >= arr.size - 4:
        return None

    peak = float(arr[peak_idx])
    if peak <= 0.0:
        return None

    post_min = float(np.min(arr[peak_idx:]))
    pullback = (peak - post_min) / peak
    return bool(pullback <= float(pullback_max))


__all__ = [
    "DEFAULT_STEP_MEMORY_TREND_HEURISTICS",
    "StepMemoryTrendEvidence",
    "StepMemoryTrendHeuristics",
    "StepMemoryWindowCreepEvidence",
    "evaluate_step_memory_creep",
    "evaluate_step_memory_window_creep",
]
