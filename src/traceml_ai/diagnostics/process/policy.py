"""Threshold policy for process diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from traceml_ai.diagnostics.bands import BandThresholds


@dataclass(frozen=True)
class ProcessDiagnosisPolicy:
    """Band thresholds used by process diagnosis rules."""

    cpu_capacity_percent: BandThresholds
    rss_peak_percent: BandThresholds
    gpu_memory_peak_percent: BandThresholds
    gpu_reserved_overhang_ratio: BandThresholds
    gpu_reserved_overhang_min_reserved_percent: float
    rank_gpu_memory_imbalance_warn_percent: float
    rank_gpu_memory_imbalance_crit_percent: float
    rank_gpu_memory_imbalance_pressure_warn_percent: float
    rank_gpu_memory_imbalance_pressure_crit_percent: float


DEFAULT_PROCESS_POLICY = ProcessDiagnosisPolicy(
    cpu_capacity_percent=BandThresholds(low_below=30.0, high_at=80.0),
    rss_peak_percent=BandThresholds(low_below=30.0, high_at=80.0),
    gpu_memory_peak_percent=BandThresholds(
        low_below=30.0,
        high_at=80.0,
        very_high_at=90.0,
    ),
    gpu_reserved_overhang_ratio=BandThresholds(high_at=2.0),
    gpu_reserved_overhang_min_reserved_percent=30.0,
    rank_gpu_memory_imbalance_warn_percent=20.0,
    rank_gpu_memory_imbalance_crit_percent=30.0,
    rank_gpu_memory_imbalance_pressure_warn_percent=30.0,
    rank_gpu_memory_imbalance_pressure_crit_percent=50.0,
)


def classify_rank_gpu_memory_imbalance_severity(
    *,
    imbalance_percent: Optional[float],
    pressure_percent: Optional[float],
    policy: ProcessDiagnosisPolicy,
) -> Optional[str]:
    """
    Classify actionable process GPU-memory imbalance across ranks.

    Rank-to-rank memory differences are noisy when each process holds only a
    small fraction of GPU capacity. Require both relative imbalance and
    worst-rank memory pressure before reporting an issue.
    """
    if imbalance_percent is None or pressure_percent is None:
        return None

    imbalance = max(0.0, float(imbalance_percent))
    pressure = max(0.0, float(pressure_percent))

    if imbalance >= float(
        policy.rank_gpu_memory_imbalance_crit_percent
    ) and pressure >= float(
        policy.rank_gpu_memory_imbalance_pressure_crit_percent
    ):
        return "crit"
    if imbalance >= float(
        policy.rank_gpu_memory_imbalance_warn_percent
    ) and pressure >= float(
        policy.rank_gpu_memory_imbalance_pressure_warn_percent
    ):
        return "warn"
    return None


__all__ = [
    "DEFAULT_PROCESS_POLICY",
    "ProcessDiagnosisPolicy",
    "classify_rank_gpu_memory_imbalance_severity",
]
