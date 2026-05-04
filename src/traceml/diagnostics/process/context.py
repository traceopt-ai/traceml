"""
Prepared analysis context for process diagnostics.

This module centralizes the summary-oriented process signals used by diagnostic
rules so multiple rules can evaluate the same bounded final-summary window
without repeating aggregation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ProcessSummarySignals:
    """
    Aggregated process signals used by summary-oriented diagnosis rules.
    """

    duration_s: Optional[float]
    samples: int

    distinct_ranks: int
    distinct_pids: int

    cpu_avg_percent: Optional[float]
    cpu_peak_percent: Optional[float]
    cpu_logical_core_count: Optional[int]
    cpu_pressure_frac: Optional[float]
    cpu_capacity_percent: Optional[float]

    ram_avg_bytes: Optional[float]
    ram_peak_bytes: Optional[float]
    ram_total_bytes: Optional[float]
    ram_pressure_frac: Optional[float]
    ram_peak_percent: Optional[float]

    gpu_available: Optional[bool]
    gpu_count: Optional[int]
    gpu_device_index: Optional[int]

    gpu_mem_used_avg_bytes: Optional[float]
    gpu_mem_used_peak_bytes: Optional[float]
    gpu_mem_reserved_avg_bytes: Optional[float]
    gpu_mem_reserved_peak_bytes: Optional[float]
    gpu_mem_total_bytes: Optional[float]

    gpu_mem_used_peak_frac: Optional[float]
    gpu_mem_reserved_peak_frac: Optional[float]
    gpu_mem_used_peak_percent: Optional[float]
    gpu_mem_reserved_peak_percent: Optional[float]
    gpu_mem_reserved_overhang_ratio: Optional[float]
    highest_overhang_rank: Optional[int]

    per_rank: Dict[int, Dict[str, Optional[float]]]

    highest_rss_rank: Optional[int]
    highest_used_rank: Optional[int]
    highest_reserved_rank: Optional[int]
    least_headroom_rank: Optional[int]
    least_headroom_bytes: Optional[float]
    rank_rss_imbalance_pct: Optional[float]
    rank_gpu_used_imbalance_pct: Optional[float]
    rank_gpu_reserved_imbalance_pct: Optional[float]
    rank_gpu_used_imbalance_percent: Optional[float]
    rank_gpu_reserved_imbalance_percent: Optional[float]


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
    numerator: Optional[float],
    denominator: Optional[float],
) -> Optional[float]:
    """
    Return a bounded ratio when both inputs are valid.
    """
    num = _safe_float(numerator)
    den = _safe_float(denominator)
    if num is None or den is None or den <= 0.0:
        return None
    return max(0.0, num / den)


def _best_rank_idx(
    per_rank: Dict[int, Dict[str, Optional[float]]],
    key: str,
) -> Optional[int]:
    """
    Return the rank with the largest valid value for one metric.
    """
    best_idx: Optional[int] = None
    best_value: Optional[float] = None

    for rank_id, item in per_rank.items():
        value = _safe_float(item.get(key))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_idx = int(rank_id)
            best_value = value

    return best_idx


def _imbalance_pct(
    per_rank: Dict[int, Dict[str, Optional[float]]],
    key: str,
) -> Optional[float]:
    """
    Compute simple per-rank imbalance as max-vs-min over max.
    """
    values = [
        _safe_float(item.get(key))
        for item in per_rank.values()
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


def _least_headroom(
    per_rank: Dict[int, Dict[str, Optional[float]]],
) -> tuple[Optional[int], Optional[float]]:
    """
    Return the rank with the smallest reserved-memory headroom.
    """
    best_rank: Optional[int] = None
    best_headroom: Optional[float] = None

    for rank_id, item in per_rank.items():
        total_bytes = _safe_float(item.get("gpu_mem_total_bytes"))
        reserved_peak_bytes = _safe_float(
            item.get("gpu_mem_reserved_peak_bytes")
        )
        if total_bytes is None or reserved_peak_bytes is None:
            continue
        headroom = max(total_bytes - reserved_peak_bytes, 0.0)
        if best_headroom is None or headroom < best_headroom:
            best_rank = int(rank_id)
            best_headroom = headroom

    return best_rank, best_headroom


def _peak_reserved_overhang(
    per_rank: Dict[int, Dict[str, Optional[float]]],
) -> tuple[Optional[float], Optional[int]]:
    """Return the largest rank-local reserved/used peak ratio."""
    best_ratio: Optional[float] = None
    best_rank: Optional[int] = None

    for rank_id, item in per_rank.items():
        ratio = _safe_float(item.get("gpu_mem_reserved_overhang_ratio"))
        if ratio is None:
            ratio = _fraction(
                item.get("gpu_mem_reserved_peak_bytes"),
                item.get("gpu_mem_used_peak_bytes"),
            )
        if ratio is None:
            continue
        if best_ratio is None or ratio > best_ratio:
            best_ratio = ratio
            best_rank = int(rank_id)

    return best_ratio, best_rank


def build_process_summary_signals(
    *,
    duration_s: Optional[float],
    samples: int,
    distinct_ranks: int,
    distinct_pids: int,
    cpu_avg_percent: Optional[float],
    cpu_peak_percent: Optional[float],
    cpu_logical_core_count: Optional[int],
    ram_avg_bytes: Optional[float],
    ram_peak_bytes: Optional[float],
    ram_total_bytes: Optional[float],
    gpu_available: Optional[bool],
    gpu_count: Optional[int],
    gpu_device_index: Optional[int],
    gpu_mem_used_avg_bytes: Optional[float],
    gpu_mem_used_peak_bytes: Optional[float],
    gpu_mem_reserved_avg_bytes: Optional[float],
    gpu_mem_reserved_peak_bytes: Optional[float],
    gpu_mem_total_bytes: Optional[float],
    per_rank: Dict[int, Dict[str, Optional[float]]],
) -> ProcessSummarySignals:
    """
    Build the normalized signal bundle shared by process diagnosis rules.
    """
    least_headroom_rank, least_headroom_bytes = _least_headroom(per_rank)
    reserved_overhang_ratio, highest_overhang_rank = _peak_reserved_overhang(
        per_rank
    )

    cpu_pressure_frac = None
    if (
        cpu_avg_percent is not None
        and cpu_logical_core_count is not None
        and cpu_logical_core_count > 0
    ):
        cpu_pressure_frac = max(
            0.0,
            float(cpu_avg_percent) / (100.0 * float(cpu_logical_core_count)),
        )

    ram_pressure_frac = _fraction(ram_peak_bytes, ram_total_bytes)
    used_peak_frac = _fraction(gpu_mem_used_peak_bytes, gpu_mem_total_bytes)
    reserved_peak_frac = _fraction(
        gpu_mem_reserved_peak_bytes,
        gpu_mem_total_bytes,
    )
    used_imbalance = _imbalance_pct(per_rank, "gpu_mem_used_peak_bytes")
    reserved_imbalance = _imbalance_pct(
        per_rank,
        "gpu_mem_reserved_peak_bytes",
    )
    return ProcessSummarySignals(
        duration_s=duration_s,
        samples=int(samples),
        distinct_ranks=int(distinct_ranks),
        distinct_pids=int(distinct_pids),
        cpu_avg_percent=cpu_avg_percent,
        cpu_peak_percent=cpu_peak_percent,
        cpu_logical_core_count=cpu_logical_core_count,
        cpu_pressure_frac=cpu_pressure_frac,
        cpu_capacity_percent=(
            cpu_pressure_frac * 100.0
            if cpu_pressure_frac is not None
            else None
        ),
        ram_avg_bytes=ram_avg_bytes,
        ram_peak_bytes=ram_peak_bytes,
        ram_total_bytes=ram_total_bytes,
        ram_pressure_frac=ram_pressure_frac,
        ram_peak_percent=(
            ram_pressure_frac * 100.0
            if ram_pressure_frac is not None
            else None
        ),
        gpu_available=gpu_available,
        gpu_count=gpu_count,
        gpu_device_index=gpu_device_index,
        gpu_mem_used_avg_bytes=gpu_mem_used_avg_bytes,
        gpu_mem_used_peak_bytes=gpu_mem_used_peak_bytes,
        gpu_mem_reserved_avg_bytes=gpu_mem_reserved_avg_bytes,
        gpu_mem_reserved_peak_bytes=gpu_mem_reserved_peak_bytes,
        gpu_mem_total_bytes=gpu_mem_total_bytes,
        gpu_mem_used_peak_frac=used_peak_frac,
        gpu_mem_reserved_peak_frac=reserved_peak_frac,
        gpu_mem_used_peak_percent=(
            used_peak_frac * 100.0 if used_peak_frac is not None else None
        ),
        gpu_mem_reserved_peak_percent=(
            reserved_peak_frac * 100.0
            if reserved_peak_frac is not None
            else None
        ),
        gpu_mem_reserved_overhang_ratio=reserved_overhang_ratio,
        highest_overhang_rank=highest_overhang_rank,
        per_rank=per_rank,
        highest_rss_rank=_best_rank_idx(per_rank, "ram_peak_bytes"),
        highest_used_rank=_best_rank_idx(per_rank, "gpu_mem_used_peak_bytes"),
        highest_reserved_rank=_best_rank_idx(
            per_rank,
            "gpu_mem_reserved_peak_bytes",
        ),
        least_headroom_rank=least_headroom_rank,
        least_headroom_bytes=least_headroom_bytes,
        rank_rss_imbalance_pct=_imbalance_pct(per_rank, "ram_peak_bytes"),
        rank_gpu_used_imbalance_pct=used_imbalance,
        rank_gpu_reserved_imbalance_pct=reserved_imbalance,
        rank_gpu_used_imbalance_percent=(
            used_imbalance * 100.0 if used_imbalance is not None else None
        ),
        rank_gpu_reserved_imbalance_percent=(
            reserved_imbalance * 100.0
            if reserved_imbalance is not None
            else None
        ),
    )


__all__ = [
    "ProcessSummarySignals",
    "build_process_summary_signals",
]
