"""
Conservative trend heuristics for step-memory diagnostics.

This module focuses on low false positives:
- strong data sufficiency gates
- absolute + relative growth checks
- slope confirmation over long windows
- optional weak-recovery check (for fragmentation-like creep)

All functions never raise to callers.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


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

    worst_slope_pct_per_100_min: float = 0.015  # 1.5% / 100 steps
    median_slope_pct_per_100_min: float = 0.010  # 1.0% / 100 steps

    require_weak_recovery: bool = True
    weak_recovery_pullback_max: float = 0.02  # <2% pullback = weak recovery


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


def evaluate_step_memory_creep(
    *,
    steps_used: int,
    worst_series_bytes: Sequence[float],
    median_series_bytes: Sequence[float],
    cfg: StepMemoryTrendHeuristics = DEFAULT_STEP_MEMORY_TREND_HEURISTICS,
) -> StepMemoryTrendEvidence:
    """
    Evaluate conservative creep signals from worst/median memory series.

    Returns
    -------
    StepMemoryTrendEvidence
        Watch/confirmed flags plus numeric evidence.
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
