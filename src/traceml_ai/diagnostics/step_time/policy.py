"""Policies for live and summary step-time diagnosis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiagnosisThresholds:
    """
    Thresholds used by the shared step-time rules.

    Live and summary policies may choose different values, but they still run
    the same rules and produce the same diagnosis vocabulary.

    INPUT_BOUND uses selected-clock ``input_wait_ms / step_time_ms``. This is
    a pre-step wait compared with the traced step envelope, not an end-to-end
    wall-time share, so values can exceed 1.0. Summary thresholds are slightly
    more conservative than live thresholds because the final report should call
    only durable bottlenecks across a larger window.
    """

    straggler_score_warn: float = 0.10
    straggler_score_crit: float = 0.20
    straggler_dominance_tolerance: float = 1.25

    input_share_warn: float = 0.25
    input_share_crit: float = 0.35

    residual_share_warn: float = 0.15
    residual_share_crit: float = 0.25

    input_bound_max_skew: float = 0.06
    compute_bound_max_skew: float = 0.06

    compute_bound_share_warn: float = 0.85
    compute_bound_share_crit: float = 0.92

    min_steps_for_confident_diag: int = 20


@dataclass(frozen=True)
class StepTimeDiagnosisPolicy:
    """Named threshold set for one step-time diagnosis window type."""

    name: str
    thresholds: DiagnosisThresholds = field(
        default_factory=DiagnosisThresholds
    )
    min_steps_for_diag: int = 20


LIVE_STEP_TIME_POLICY = StepTimeDiagnosisPolicy(
    name="live",
    thresholds=DiagnosisThresholds(),
    min_steps_for_diag=20,
)

SUMMARY_STEP_TIME_POLICY = StepTimeDiagnosisPolicy(
    name="summary",
    thresholds=DiagnosisThresholds(
        straggler_score_warn=0.10,
        straggler_score_crit=0.18,
        input_share_warn=0.30,
        input_share_crit=0.40,
        residual_share_warn=0.18,
        residual_share_crit=0.28,
        input_bound_max_skew=0.05,
        compute_bound_max_skew=0.05,
        compute_bound_share_warn=0.88,
        compute_bound_share_crit=0.94,
        min_steps_for_confident_diag=20,
    ),
    min_steps_for_diag=50,
)

DEFAULT_THRESHOLDS = LIVE_STEP_TIME_POLICY.thresholds


__all__ = [
    "DEFAULT_THRESHOLDS",
    "DiagnosisThresholds",
    "LIVE_STEP_TIME_POLICY",
    "SUMMARY_STEP_TIME_POLICY",
    "StepTimeDiagnosisPolicy",
]
