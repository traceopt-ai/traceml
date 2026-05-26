"""Small threshold helpers for diagnosis bands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Band = Literal["low", "normal", "high", "very_high"]


@dataclass(frozen=True)
class BandThresholds:
    """Classify one scalar value into low/normal/high bands."""

    low_below: Optional[float] = None
    high_at: Optional[float] = None
    very_high_at: Optional[float] = None

    def classify(self, value: Optional[float]) -> Optional[Band]:
        if value is None:
            return None

        numeric = float(value)
        if self.very_high_at is not None and numeric >= self.very_high_at:
            return "very_high"
        if self.high_at is not None and numeric >= self.high_at:
            return "high"
        if self.low_below is not None and numeric < self.low_below:
            return "low"
        return "normal"


def format_band_value(
    label: str,
    band: Optional[Band],
    value: Optional[float],
    *,
    unit: str = "%",
    precision: int = 0,
) -> Optional[str]:
    """Return compact text such as ``CPU normal 42%``."""
    if band is None or value is None:
        return None

    rendered = f"{float(value):.{precision}f}{unit}"
    return f"{label} {band.replace('_', ' ')} {rendered}"


__all__ = [
    "Band",
    "BandThresholds",
    "format_band_value",
]
