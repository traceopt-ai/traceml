"""Tunable policies for step-memory diagnosis."""

from __future__ import annotations

from dataclasses import dataclass, field

from traceml.analytics.trends import TrendBands
from traceml.diagnostics.trends import TrendConfig


@dataclass(frozen=True)
class StepMemoryDiagnosisThresholds:
    """Thresholds used by step-memory live and summary diagnosis."""

    min_steps_for_diag: int = 50

    pressure_warn_fraction: float = 0.92
    pressure_crit_fraction: float = 0.97

    imbalance_skew_warn: float = 0.12
    imbalance_skew_crit: float = 0.20

    creep_score_delta_scale_bytes: float = 100.0 * 1024.0 * 1024.0
    creep_confirmed_delta_bytes: float = 1024.0 * 1024.0 * 1024.0

    require_recent_gt_mid: bool = True
    require_mid_ge_baseline: bool = True

    trend: TrendConfig = field(
        default_factory=lambda: TrendConfig(
            min_points=50,
            bands=TrendBands(warmup_frac=0.0),
        )
    )


@dataclass(frozen=True)
class StepMemoryDiagnosisPolicy:
    """Named threshold bundle for a step-memory diagnosis path."""

    name: str
    thresholds: StepMemoryDiagnosisThresholds = field(
        default_factory=StepMemoryDiagnosisThresholds
    )


LIVE_STEP_MEMORY_POLICY = StepMemoryDiagnosisPolicy(name="live")
SUMMARY_STEP_MEMORY_POLICY = StepMemoryDiagnosisPolicy(name="summary")

DEFAULT_STEP_MEMORY_THRESHOLDS = LIVE_STEP_MEMORY_POLICY.thresholds


__all__ = [
    "StepMemoryDiagnosisPolicy",
    "StepMemoryDiagnosisThresholds",
    "LIVE_STEP_MEMORY_POLICY",
    "SUMMARY_STEP_MEMORY_POLICY",
    "DEFAULT_STEP_MEMORY_THRESHOLDS",
]
