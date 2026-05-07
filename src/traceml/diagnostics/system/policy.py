"""Threshold policy for system diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from traceml.diagnostics.bands import BandThresholds


@dataclass(frozen=True)
class SystemDiagnosisPolicy:
    """Band thresholds used by system diagnosis rules and summary stats."""

    cpu_avg_percent: BandThresholds
    ram_peak_percent: BandThresholds
    gpu_util_avg_percent: BandThresholds
    gpu_memory_peak_percent: BandThresholds
    gpu_temp_peak_c: BandThresholds
    gpu_power_avg_limit_percent: BandThresholds


DEFAULT_SYSTEM_POLICY = SystemDiagnosisPolicy(
    cpu_avg_percent=BandThresholds(low_below=30.0, high_at=80.0),
    ram_peak_percent=BandThresholds(low_below=30.0, high_at=80.0),
    gpu_util_avg_percent=BandThresholds(low_below=30.0, high_at=80.0),
    gpu_memory_peak_percent=BandThresholds(
        low_below=30.0,
        high_at=80.0,
        very_high_at=90.0,
    ),
    gpu_temp_peak_c=BandThresholds(high_at=85.0),
    gpu_power_avg_limit_percent=BandThresholds(
        low_below=30.0,
        high_at=80.0,
    ),
)


__all__ = [
    "DEFAULT_SYSTEM_POLICY",
    "SystemDiagnosisPolicy",
]
