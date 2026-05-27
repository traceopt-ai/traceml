"""Threshold policy for process diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from traceml_ai.diagnostics.bands import BandThresholds


@dataclass(frozen=True)
class ProcessDiagnosisPolicy:
    """Band thresholds used by process diagnosis rules."""

    cpu_capacity_percent: BandThresholds
    rss_peak_percent: BandThresholds
    gpu_memory_peak_percent: BandThresholds
    gpu_reserved_overhang_ratio: BandThresholds
    gpu_reserved_overhang_min_reserved_percent: float
    rank_gpu_memory_imbalance_percent: BandThresholds


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
    rank_gpu_memory_imbalance_percent=BandThresholds(high_at=30.0),
)


__all__ = [
    "DEFAULT_PROCESS_POLICY",
    "ProcessDiagnosisPolicy",
]
