"""Small statistics helpers shared by benchmark phases."""

from __future__ import annotations

import math
from typing import Any


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * pct / 100.0
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return data[lower]
    weight = pos - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


def summary_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "std_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "n": len(values),
        "mean_ms": mean,
        "median_ms": percentile(values, 50.0),
        "p95_ms": percentile(values, 95.0),
        "p99_ms": percentile(values, 99.0),
        "std_ms": math.sqrt(variance),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(delta: float | None, base: float | None) -> float | None:
    if delta is None or base in (None, 0.0):
        return None
    return 100.0 * delta / float(base)


def baseline_noise_floor(base_stats: dict[str, Any]) -> float | None:
    median = base_stats.get("median_ms")
    p95 = base_stats.get("p95_ms")
    std = base_stats.get("std_ms")
    if median is None or p95 is None or std is None:
        return None
    return max(float(std), abs(float(p95) - float(median)))


def overhead_label(row: dict[str, Any]) -> str:
    delta = row.get("overhead_median_ms")
    noise = row.get("baseline_noise_floor_ms")
    pct_value = row.get("overhead_median_pct")
    if delta is None:
        return "baseline"
    if row.get("within_baseline_noise"):
        return f"within noise (|delta| <= {fmt(noise)} ms)"
    return f"{fmt(delta)} ms / {fmt(pct_value, 2)}%"
