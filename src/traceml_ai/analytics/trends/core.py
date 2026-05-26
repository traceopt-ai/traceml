"""
Shared trend computation for TraceML metrics.

This module intentionally contains only generic time-series logic. Metric-
specific policies such as memory-creep thresholds should stay in the relevant
diagnostics module and consume the structured TrendEvidence returned here.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from .schema import (
    DEFAULT_TREND_CONFIG,
    TrendBand,
    TrendComputationConfig,
    TrendEvidence,
)


def _finite_values(series: Sequence[float]) -> list[float]:
    out: list[float] = []
    for value in series:
        try:
            numeric = float(value)
        except Exception:
            continue
        if math.isfinite(numeric):
            out.append(numeric)
    return out


def _avg(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _band_slice(values: Sequence[float], band: TrendBand) -> list[float]:
    n = len(values)
    if n <= 0:
        return []

    start = int(math.floor(n * float(band.start_frac)))
    end = int(math.ceil(n * float(band.end_frac)))

    start = max(0, min(start, n - 1))
    end = max(start + 1, min(end, n))
    return list(values[start:end])


def compute_trend_evidence(
    series: Sequence[float],
    *,
    config: TrendComputationConfig = DEFAULT_TREND_CONFIG,
) -> Optional[TrendEvidence]:
    """
    Compute canonical trailing-window trend evidence for one numeric series.

    The retained series is:
    1. filtered to finite numeric values
    2. truncated to `history_limit` when configured
    3. warmup-trimmed before baseline/mid/recent bands are measured
    """
    values = _finite_values(series)
    if len(values) < int(config.min_points):
        return None

    truncated = False
    if config.history_limit is not None and len(values) > int(
        config.history_limit
    ):
        values = values[-int(config.history_limit) :]
        truncated = True

    if len(values) < int(config.min_points):
        return None

    warmup_n = int(math.floor(len(values) * float(config.bands.warmup_frac)))
    stable = values[warmup_n:] if warmup_n > 0 else values
    if len(stable) < int(config.min_points):
        return None

    baseline_vals = _band_slice(stable, config.bands.baseline)
    mid_vals = _band_slice(stable, config.bands.mid)
    recent_vals = _band_slice(stable, config.bands.recent)

    if not baseline_vals or not mid_vals or not recent_vals:
        return None

    baseline_avg = _avg(baseline_vals)
    mid_avg = _avg(mid_vals)
    recent_avg = _avg(recent_vals)

    delta_vs_baseline = recent_avg - baseline_avg
    delta_vs_mid = recent_avg - mid_avg

    pct_vs_baseline = (
        None
        if abs(baseline_avg) <= 1e-12
        else delta_vs_baseline / baseline_avg
    )
    pct_vs_mid = None if abs(mid_avg) <= 1e-12 else delta_vs_mid / mid_avg

    return TrendEvidence(
        points_seen=len(_finite_values(series)),
        points_used=len(values),
        truncated=truncated,
        baseline_avg=baseline_avg,
        mid_avg=mid_avg,
        recent_avg=recent_avg,
        delta_vs_baseline=delta_vs_baseline,
        delta_vs_mid=delta_vs_mid,
        delta_pct_vs_baseline=pct_vs_baseline,
        delta_pct_vs_mid=pct_vs_mid,
    )


def compute_trend_pct(
    series: Sequence[float],
    *,
    config: TrendComputationConfig = DEFAULT_TREND_CONFIG,
) -> Optional[float]:
    """
    Compatibility helper returning the canonical recent-vs-baseline percentage.
    """
    evidence = compute_trend_evidence(series, config=config)
    if evidence is None:
        return None
    return evidence.delta_pct_vs_baseline


def format_trend_pct(
    trend_pct: Optional[float],
    *,
    deadband_pct: float = DEFAULT_TREND_CONFIG.deadband_pct,
) -> str:
    """
    Format a trend ratio into a compact label.

    This remains renderer-friendly and backward-compatible with existing use.
    """
    if trend_pct is None:
        return "—"
    if abs(trend_pct) < float(deadband_pct):
        return f"~ {trend_pct * 100:+.1f}%"
    return f"{'↑' if trend_pct > 0 else '↓'} {trend_pct * 100:+.1f}%"
