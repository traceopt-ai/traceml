from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TrendBand:
    """
    One normalized band inside a trailing analysis window.

    Fractions are expressed in [0, 1], where 0.0 is the start of the retained
    trailing window and 1.0 is the end of that window.
    """

    start_frac: float
    end_frac: float

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.start_frac) < float(self.end_frac) <= 1.0):
            raise ValueError(
                "TrendBand requires 0.0 <= start_frac < end_frac <= 1.0"
            )


@dataclass(frozen=True)
class TrendBands:
    """
    Canonical band layout for one trend computation.

    warmup_frac:
        Fraction of the trailing window ignored before computing trend bands.
    baseline:
        Stable baseline band after warmup.
    mid:
        Middle reference band used to confirm direction.
    recent:
        Most recent band representing current behavior.
    """

    warmup_frac: float = 0.10
    baseline: TrendBand = field(default_factory=lambda: TrendBand(0.15, 0.25))
    mid: TrendBand = field(default_factory=lambda: TrendBand(0.45, 0.55))
    recent: TrendBand = field(default_factory=lambda: TrendBand(0.90, 1.00))

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.warmup_frac) < 1.0):
            raise ValueError("warmup_frac must be in [0.0, 1.0)")


@dataclass(frozen=True)
class TrendComputationConfig:
    """
    Shared trend computation policy.

    This object is designed to be easy to override programmatically later
    without changing call sites. A future config layer can populate this
    dataclass directly.
    """

    history_limit: Optional[int] = 10_000
    min_points: int = 200
    deadband_pct: float = 0.02
    bands: TrendBands = field(default_factory=TrendBands)

    def __post_init__(self) -> None:
        if self.history_limit is not None and int(self.history_limit) <= 0:
            raise ValueError("history_limit must be positive when provided")
        if int(self.min_points) < 3:
            raise ValueError("min_points must be at least 3")
        if float(self.deadband_pct) < 0.0:
            raise ValueError("deadband_pct must be non-negative")


DEFAULT_TREND_CONFIG = TrendComputationConfig()


@dataclass(frozen=True)
class TrendEvidence:
    """
    Structured trend evidence computed from one numeric series.

    All averages are computed on the trailing analysis window after warmup.
    Deltas are recent minus baseline / mid.
    """

    points_seen: int
    points_used: int
    truncated: bool

    baseline_avg: float
    mid_avg: float
    recent_avg: float

    delta_vs_baseline: float
    delta_vs_mid: float

    delta_pct_vs_baseline: Optional[float]
    delta_pct_vs_mid: Optional[float]
