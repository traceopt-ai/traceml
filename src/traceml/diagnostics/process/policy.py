"""Threshold policy for process diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from traceml.diagnostics.bands import BandThresholds


@dataclass(frozen=True)
class ProcessDiagnosisPolicy:
    """Band thresholds used by process diagnosis rules."""

    cpu_capacity_percent: BandThresholds
    rss_peak_percent: BandThresholds
    gpu_memory_peak_percent: BandThresholds
    gpu_reserved_overhang_ratio: BandThresholds
    rank_gpu_memory_imbalance_percent: BandThresholds


DEFAULT_PROCESS_POLICY = ProcessDiagnosisPolicy(
    cpu_capacity_percent=BandThresholds(low_below=30.0, high_at=80.0),
    rss_peak_percent=BandThresholds(low_below=30.0, high_at=80.0),
    gpu_memory_peak_percent=BandThresholds(
        low_below=30.0,
        high_at=80.0,
        very_high_at=90.0,
    ),
    gpu_reserved_overhang_ratio=BandThresholds(high_at=1.5),
    rank_gpu_memory_imbalance_percent=BandThresholds(high_at=30.0),
)


__all__ = [
    "DEFAULT_PROCESS_POLICY",
    "ProcessDiagnosisPolicy",
]
