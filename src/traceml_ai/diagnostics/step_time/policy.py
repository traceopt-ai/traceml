"""Policies for live and summary step-time diagnosis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiagnosisThresholds:
    """
    Thresholds used by the shared step-time rules.

    Live and summary policies may choose different values, but they still run
    the same rules and produce the same diagnosis vocabulary.
    """

    input_straggler_score_warn: float = 0.10
    input_straggler_score_crit: float = 0.20

    compute_straggler_score_warn: float = 0.10
    compute_straggler_score_crit: float = 0.20

    input_share_warn: float = 0.25
    input_share_crit: float = 0.35

    wait_share_warn: float = 0.15
    wait_share_crit: float = 0.25

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
        input_straggler_score_warn=0.10,
        input_straggler_score_crit=0.18,
        compute_straggler_score_warn=0.10,
        compute_straggler_score_crit=0.18,
        input_share_warn=0.30,
        input_share_crit=0.40,
        wait_share_warn=0.18,
        wait_share_crit=0.28,
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
