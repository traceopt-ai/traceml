"""
Generic time-series trend utilities.

These helpers are metric-agnostic and can be reused for step time, GPU util,
memory, and other numeric series.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class TrendConfig:
    """
    Configuration for window-delta trend computation.

    short_window:
        Number of most-recent points for short-term average.
    long_window:
        Number of most-recent points for long-term average.
    min_points:
        Minimum usable samples required before returning a trend.
    deadband_pct:
        Absolute trend threshold below which trend is considered flat.
    """

    short_window: int = 100
    long_window: int = 200
    min_points: int = 50
    deadband_pct: float = 0.02


DEFAULT_TREND_CONFIG = TrendConfig()


def _finite_values(series: Sequence[float]) -> list[float]:
    out: list[float] = []
    for v in series:
        try:
            f = float(v)
        except Exception:
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _avg_tail(values: Sequence[float], n: int) -> float:
    if not values:
        return 0.0
    tail = values[-n:] if len(values) >= n else values
    return sum(tail) / float(len(tail))


def compute_trend_pct(
    series: Sequence[float],
    *,
    config: TrendConfig = DEFAULT_TREND_CONFIG,
) -> Optional[float]:
    """
    Compute trend as percentage delta between short and long trailing averages.

    Returns
    -------
    Optional[float]
        Ratio in [-inf, +inf], e.g. +0.05 means +5%.
        Returns None when data is insufficient or denominator is too small.
    """
    vals = _finite_values(series)
    if len(vals) < max(2, int(config.min_points)):
        return None

    short_n = max(1, int(config.short_window))
    long_n = max(short_n, int(config.long_window))

    short_avg = _avg_tail(vals, short_n)
    long_avg = _avg_tail(vals, min(long_n, len(vals)))

    eps = 1e-12
    if abs(long_avg) <= eps:
        return None

    return (short_avg - long_avg) / long_avg


def format_trend_pct(
    trend_pct: Optional[float],
    *,
    deadband_pct: float = DEFAULT_TREND_CONFIG.deadband_pct,
) -> str:
    """
    Format a trend ratio into a compact terminal-friendly label.
    """
    if trend_pct is None:
        return "—"
    if abs(trend_pct) < float(deadband_pct):
        return f"~ {trend_pct * 100:+.1f}%"
    return f"{'↑' if trend_pct > 0 else '↓'} {trend_pct * 100:+.1f}%"
