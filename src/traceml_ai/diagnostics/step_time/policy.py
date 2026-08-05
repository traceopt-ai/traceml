"""Policies for live and summary step-time diagnosis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiagnosisThresholds:
    """
    Thresholds used by the shared step-time rules.

    Built-in live and summary policies share these values and differ only by
    their selected timing windows. Explicit callers may still supply a custom
    policy.

    Typical overhead diagnoses use selected-clock per-rank Step Time shares.
    The context takes the median of those shares across ranks, where
    ``step_time_ms = input_wait_ms + traced_step_time_ms``. The shared overhead
    thresholds are the future configuration surface for input, residual, and
    H2D policies. Compute-bound uses the same denominator and a separate
    informational dominance threshold.

    ``min_steps_for_warning_diag`` is the minimum window size for warning-only
    bottleneck diagnoses. ``min_steps_for_confident_diag`` is the minimum window
    size for critical diagnoses. ``straggler_cause_coverage_min`` is the future
    configuration surface for naming a rank-straggler cause.
    """

    straggler_score_warn: float = 0.10
    straggler_score_crit: float = 0.20
    straggler_cause_coverage_min: float = 0.80

    overhead_share_warn: float = 0.10
    overhead_share_crit: float = 0.20

    compute_bound_share_warn: float = 0.90

    min_steps_for_warning_diag: int = 2
    min_steps_for_confident_diag: int = 20


@dataclass(frozen=True)
class StepTimeDiagnosisPolicy:
    """Named threshold set for shared Step Time diagnosis."""

    name: str
    thresholds: DiagnosisThresholds = field(
        default_factory=DiagnosisThresholds
    )


DEFAULT_THRESHOLDS = DiagnosisThresholds()

LIVE_STEP_TIME_POLICY = StepTimeDiagnosisPolicy(
    name="live",
    thresholds=DEFAULT_THRESHOLDS,
)

SUMMARY_STEP_TIME_POLICY = StepTimeDiagnosisPolicy(
    name="summary",
    thresholds=DEFAULT_THRESHOLDS,
)


__all__ = [
    "DEFAULT_THRESHOLDS",
    "DiagnosisThresholds",
    "LIVE_STEP_TIME_POLICY",
    "SUMMARY_STEP_TIME_POLICY",
    "StepTimeDiagnosisPolicy",
]
