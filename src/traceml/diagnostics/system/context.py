"""
Prepared analysis context for system diagnostics.

This module centralizes the summary-oriented system signals used by diagnostic
rules so multiple rules can evaluate the same bounded final-summary window
without repeating aggregation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class SystemSummarySignals:
    """
    Aggregated system signals used by summary-oriented system diagnosis rules.
    """

    duration_s: Optional[float]
    samples: int

    cpu_avg_percent: Optional[float]
    cpu_peak_percent: Optional[float]

    ram_avg_bytes: Optional[float]
    ram_peak_bytes: Optional[float]
    ram_total_bytes: Optional[float]

    gpu_available: Optional[bool]
    gpu_count: Optional[int]
    gpu_util_avg_percent: Optional[float]
    gpu_util_peak_percent: Optional[float]
    gpu_mem_avg_bytes: Optional[float]
    gpu_mem_peak_bytes: Optional[float]
    gpu_temp_avg_c: Optional[float]
    gpu_temp_peak_c: Optional[float]
    gpu_power_avg_w: Optional[float]
    gpu_power_peak_w: Optional[float]

    per_gpu: Dict[int, Dict[str, Optional[float]]]

    ram_pressure_frac: Optional[float]
    ram_peak_percent: Optional[float]
    gpu_mem_pressure_frac: Optional[float]
    gpu_mem_peak_percent: Optional[float]
    gpu_power_avg_limit_frac: Optional[float]
    gpu_power_avg_limit_percent: Optional[float]
    gpu_util_imbalance_pct: Optional[float]
    gpu_mem_imbalance_pct: Optional[float]
    lowest_util_gpu_idx: Optional[int]
    highest_util_gpu_idx: Optional[int]
    highest_mem_gpu_idx: Optional[int]
    highest_mem_pressure_gpu_idx: Optional[int]
    highest_temp_gpu_idx: Optional[int]
    highest_power_gpu_idx: Optional[int]


def _safe_float(value: Optional[float]) -> Optional[float]:
    """
    Convert a value to float when possible.
    """
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fraction(
    numerator: Optional[float], denominator: Optional[float]
) -> Optional[float]:
    """
    Return a bounded ratio when both inputs are valid.
    """
    num = _safe_float(numerator)
    den = _safe_float(denominator)
    if num is None or den is None or den <= 0.0:
        return None
    return max(0.0, num / den)


def _best_gpu_idx(
    per_gpu: Dict[int, Dict[str, Optional[float]]],
    key: str,
    *,
    highest: bool,
) -> Optional[int]:
    """
    Return the GPU index with the highest or lowest valid value for one metric.
    """
    best_idx: Optional[int] = None
    best_value: Optional[float] = None

    for gpu_idx, item in per_gpu.items():
        value = _safe_float(item.get(key))
        if value is None:
            continue
        if best_value is None:
            best_idx = int(gpu_idx)
            best_value = value
            continue
        if highest and value > best_value:
            best_idx = int(gpu_idx)
            best_value = value
        if not highest and value < best_value:
            best_idx = int(gpu_idx)
            best_value = value

    return best_idx


def _imbalance_pct(
    per_gpu: Dict[int, Dict[str, Optional[float]]],
    key: str,
) -> Optional[float]:
    """
    Compute simple GPU imbalance as max-vs-min over max.
    """
    values = [
        _safe_float(item.get(key))
        for item in per_gpu.values()
        if _safe_float(item.get(key)) is not None
    ]
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None

    max_value = max(values)
    min_value = min(values)
    if max_value <= 0.0:
        return 0.0
    return max(0.0, (max_value - min_value) / max_value)


def _gpu_memory_pressure_frac(
    per_gpu: Dict[int, Dict[str, Optional[float]]],
) -> Optional[float]:
    """
    Return the highest observed GPU-memory pressure fraction across devices.
    """
    fractions = []
    for item in per_gpu.values():
        pressure = _fraction(
            item.get("mem_peak_bytes"),
            item.get("mem_total_bytes"),
        )
        if pressure is not None:
            fractions.append(pressure)
    if not fractions:
        return None
    return max(fractions)


def _gpu_power_limit_frac(
    per_gpu: Dict[int, Dict[str, Optional[float]]],
) -> Optional[float]:
    fractions = []
    for item in per_gpu.values():
        pressure = _fraction(
            item.get("power_avg_w"),
            item.get("power_limit_w"),
        )
        if pressure is not None:
            fractions.append(pressure)
    if not fractions:
        return None
    return max(fractions)


def _best_gpu_pressure_idx(
    per_gpu: Dict[int, Dict[str, Optional[float]]],
    numerator_key: str,
    denominator_key: str,
) -> Optional[int]:
    best_idx: Optional[int] = None
    best_value: Optional[float] = None

    for gpu_idx, item in per_gpu.items():
        value = _fraction(item.get(numerator_key), item.get(denominator_key))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_idx = int(gpu_idx)
            best_value = value

    return best_idx


def build_system_summary_signals(
    *,
    duration_s: Optional[float],
    samples: int,
    cpu_avg_percent: Optional[float],
    cpu_peak_percent: Optional[float],
    ram_avg_bytes: Optional[float],
    ram_peak_bytes: Optional[float],
    ram_total_bytes: Optional[float],
    gpu_available: Optional[bool],
    gpu_count: Optional[int],
    gpu_util_avg_percent: Optional[float],
    gpu_util_peak_percent: Optional[float],
    gpu_mem_avg_bytes: Optional[float],
    gpu_mem_peak_bytes: Optional[float],
    gpu_temp_avg_c: Optional[float],
    gpu_temp_peak_c: Optional[float],
    gpu_power_avg_w: Optional[float],
    gpu_power_peak_w: Optional[float],
    per_gpu: Dict[int, Dict[str, Optional[float]]],
) -> SystemSummarySignals:
    """
    Build the normalized signal bundle shared by system diagnosis rules.
    """
    ram_pressure_frac = _fraction(ram_peak_bytes, ram_total_bytes)
    gpu_mem_pressure_frac = _gpu_memory_pressure_frac(per_gpu)
    gpu_power_avg_limit_frac = _gpu_power_limit_frac(per_gpu)

    return SystemSummarySignals(
        duration_s=duration_s,
        samples=int(samples),
        cpu_avg_percent=cpu_avg_percent,
        cpu_peak_percent=cpu_peak_percent,
        ram_avg_bytes=ram_avg_bytes,
        ram_peak_bytes=ram_peak_bytes,
        ram_total_bytes=ram_total_bytes,
        gpu_available=gpu_available,
        gpu_count=gpu_count,
        gpu_util_avg_percent=gpu_util_avg_percent,
        gpu_util_peak_percent=gpu_util_peak_percent,
        gpu_mem_avg_bytes=gpu_mem_avg_bytes,
        gpu_mem_peak_bytes=gpu_mem_peak_bytes,
        gpu_temp_avg_c=gpu_temp_avg_c,
        gpu_temp_peak_c=gpu_temp_peak_c,
        gpu_power_avg_w=gpu_power_avg_w,
        gpu_power_peak_w=gpu_power_peak_w,
        per_gpu=per_gpu,
        ram_pressure_frac=ram_pressure_frac,
        ram_peak_percent=(
            ram_pressure_frac * 100.0
            if ram_pressure_frac is not None
            else None
        ),
        gpu_mem_pressure_frac=gpu_mem_pressure_frac,
        gpu_mem_peak_percent=(
            gpu_mem_pressure_frac * 100.0
            if gpu_mem_pressure_frac is not None
            else None
        ),
        gpu_power_avg_limit_frac=gpu_power_avg_limit_frac,
        gpu_power_avg_limit_percent=(
            gpu_power_avg_limit_frac * 100.0
            if gpu_power_avg_limit_frac is not None
            else None
        ),
        gpu_util_imbalance_pct=_imbalance_pct(per_gpu, "util_avg_percent"),
        gpu_mem_imbalance_pct=_imbalance_pct(per_gpu, "mem_peak_bytes"),
        lowest_util_gpu_idx=_best_gpu_idx(
            per_gpu,
            "util_avg_percent",
            highest=False,
        ),
        highest_util_gpu_idx=_best_gpu_idx(
            per_gpu,
            "util_avg_percent",
            highest=True,
        ),
        highest_mem_gpu_idx=_best_gpu_idx(
            per_gpu,
            "mem_peak_bytes",
            highest=True,
        ),
        highest_mem_pressure_gpu_idx=_best_gpu_pressure_idx(
            per_gpu,
            "mem_peak_bytes",
            "mem_total_bytes",
        ),
        highest_temp_gpu_idx=_best_gpu_idx(
            per_gpu,
            "temp_peak_c",
            highest=True,
        ),
        highest_power_gpu_idx=_best_gpu_pressure_idx(
            per_gpu,
            "power_avg_w",
            "power_limit_w",
        ),
    )


__all__ = [
    "SystemSummarySignals",
    "build_system_summary_signals",
]
