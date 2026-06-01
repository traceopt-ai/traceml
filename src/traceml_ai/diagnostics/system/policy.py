"""Threshold policy for system diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Optional

from traceml_ai.diagnostics.bands import BandThresholds

GPUUtilizationBand = Literal["low", "moderate", "healthy"]


@dataclass(frozen=True)
class GPUUtilizationBands:
    """
    Average GPU utilization bands for System diagnosis.

    The public diagnosis bands are intentionally explicit:
    - x < 30.0: low utilization
    - 30.0 <= x <= 70.0: moderate utilization
    - x > 70.0: healthy utilization, so no GPU-utilization issue is emitted
    """

    low_below: float = 30.0
    moderate_at_or_below: float = 70.0

    def classify(self, value: Optional[float]) -> Optional[GPUUtilizationBand]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except Exception:
            return None
        if not isfinite(numeric):
            return None

        if numeric < float(self.low_below):
            return "low"
        if numeric <= float(self.moderate_at_or_below):
            return "moderate"
        return "healthy"


@dataclass(frozen=True)
class SystemDiagnosisPolicy:
    """Band thresholds used by system diagnosis rules and summary stats."""

    cpu_avg_percent: BandThresholds
    ram_peak_percent: BandThresholds
    gpu_util_avg_percent: GPUUtilizationBands
    gpu_memory_peak_percent: BandThresholds
    gpu_temp_peak_c: BandThresholds
    gpu_power_avg_limit_percent: BandThresholds


DEFAULT_SYSTEM_POLICY = SystemDiagnosisPolicy(
    cpu_avg_percent=BandThresholds(low_below=30.0, high_at=80.0),
    ram_peak_percent=BandThresholds(low_below=30.0, high_at=80.0),
    gpu_util_avg_percent=GPUUtilizationBands(),
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
    "GPUUtilizationBand",
    "GPUUtilizationBands",
    "SystemDiagnosisPolicy",
]
