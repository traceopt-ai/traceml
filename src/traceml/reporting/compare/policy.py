"""
Policy helpers for decision-grade TraceML run comparison.

This module centralizes the thresholds and rankings used by the compare verdict
layer so that:

- compare interpretation stays stable across outputs
- rendering code does not own product policy
- future monitor or CI integrations can reuse the same logic

The thresholds are intentionally conservative. The goal is to avoid overstating
small run-to-run noise as a material regression or improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_SIGNIFICANCE_ORDER = {
    "negligible": 0,
    "moderate": 1,
    "material": 2,
}

_STEP_TIME_STATUS_RANK = {
    "NO DATA": 0,
    "BALANCED": 1,
    "INPUT-BOUND": 2,
    "COMPUTE-BOUND": 2,
    "WAIT-HEAVY": 3,
    "INPUT STRAGGLER": 3,
    "COMPUTE STRAGGLER": 3,
    "STRAGGLER": 4,
}

_STEP_MEMORY_STATUS_RANK = {
    "NO DATA": 0,
    "BALANCED": 1,
    "MEMORY RISING": 2,
    "IMBALANCE": 3,
    "HIGH PRESSURE": 4,
    "MEMORY CREEP": 4,
}


@dataclass(frozen=True)
class CompareDecisionPolicy:
    """
    Conservative policy thresholds for TraceML compare interpretation.

    Notes
    -----
    - `step_avg_pct_*` gates primary performance regression or improvement.
    - `wait_share_pp_*` and `phase_shift_pp_*` are supporting timing signals.
    - Memory thresholds are supporting signals unless reinforced by a stronger
      memory diagnosis change.
    - The policy is intentionally biased toward abstaining rather than
      overstating a conclusion.
    """

    step_avg_pct_moderate: float = 3.0
    step_avg_pct_material: float = 8.0

    wait_share_pp_moderate: float = 0.75
    wait_share_pp_material: float = 2.5

    phase_shift_pp_moderate: float = 0.75
    phase_shift_pp_material: float = 2.0

    memory_bytes_moderate: float = 256.0 * 1024.0 * 1024.0
    memory_bytes_material: float = 1.0 * 1024.0 * 1024.0 * 1024.0

    memory_skew_pp_moderate: float = 0.75
    memory_skew_pp_material: float = 2.5


DEFAULT_COMPARE_POLICY = CompareDecisionPolicy()


def significance_rank(name: str) -> int:
    """
    Return a stable rank for one significance label.
    """
    return _SIGNIFICANCE_ORDER.get(str(name or "").strip(), 0)


def classify_step_avg_pct(
    abs_pct: Optional[float],
    *,
    policy: CompareDecisionPolicy = DEFAULT_COMPARE_POLICY,
) -> str:
    """
    Classify an average step-time percent change.
    """
    if abs_pct is None:
        return "negligible"
    if abs_pct >= float(policy.step_avg_pct_material):
        return "material"
    if abs_pct >= float(policy.step_avg_pct_moderate):
        return "moderate"
    return "negligible"


def classify_wait_share_pp(
    abs_pp: Optional[float],
    *,
    policy: CompareDecisionPolicy = DEFAULT_COMPARE_POLICY,
) -> str:
    """
    Classify a wait-share delta in percentage points.
    """
    if abs_pp is None:
        return "negligible"
    if abs_pp >= float(policy.wait_share_pp_material):
        return "material"
    if abs_pp >= float(policy.wait_share_pp_moderate):
        return "moderate"
    return "negligible"


def classify_phase_shift_pp(
    abs_pp: Optional[float],
    *,
    policy: CompareDecisionPolicy = DEFAULT_COMPARE_POLICY,
) -> str:
    """
    Classify a phase split shift in percentage points.
    """
    if abs_pp is None:
        return "negligible"
    if abs_pp >= float(policy.phase_shift_pp_material):
        return "material"
    if abs_pp >= float(policy.phase_shift_pp_moderate):
        return "moderate"
    return "negligible"


def classify_memory_bytes(
    abs_bytes: Optional[float],
    *,
    policy: CompareDecisionPolicy = DEFAULT_COMPARE_POLICY,
) -> str:
    """
    Classify a memory delta in bytes.
    """
    if abs_bytes is None:
        return "negligible"
    if abs_bytes >= float(policy.memory_bytes_material):
        return "material"
    if abs_bytes >= float(policy.memory_bytes_moderate):
        return "moderate"
    return "negligible"


def classify_memory_skew_pp(
    abs_pp: Optional[float],
    *,
    policy: CompareDecisionPolicy = DEFAULT_COMPARE_POLICY,
) -> str:
    """
    Classify a memory skew delta in percentage points.
    """
    if abs_pp is None:
        return "negligible"
    if abs_pp >= float(policy.memory_skew_pp_material):
        return "material"
    if abs_pp >= float(policy.memory_skew_pp_moderate):
        return "moderate"
    return "negligible"


def step_time_status_rank(status: Optional[str]) -> int:
    """
    Return a conservative severity rank for one step-time status.
    """
    return _STEP_TIME_STATUS_RANK.get(str(status or "").strip(), 0)


def step_memory_status_rank(status: Optional[str]) -> int:
    """
    Return a conservative severity rank for one step-memory status.
    """
    return _STEP_MEMORY_STATUS_RANK.get(str(status or "").strip(), 0)


# Backward-compatible generic helpers, kept to avoid breaking imports if other
# compare code still references them.
def classify_pct(
    abs_pct: Optional[float],
    *,
    policy: CompareDecisionPolicy = DEFAULT_COMPARE_POLICY,
) -> str:
    """
    Backward-compatible alias for step-time percent classification.
    """
    return classify_step_avg_pct(abs_pct, policy=policy)


def classify_pp(
    abs_pp: Optional[float],
    *,
    moderate: float,
    material: float,
) -> str:
    """
    Backward-compatible generic percentage-point classifier.
    """
    if abs_pp is None:
        return "negligible"
    if abs_pp >= float(material):
        return "material"
    if abs_pp >= float(moderate):
        return "moderate"
    return "negligible"


def classify_bytes(
    abs_bytes: Optional[float],
    *,
    policy: CompareDecisionPolicy = DEFAULT_COMPARE_POLICY,
) -> str:
    """
    Backward-compatible alias for memory-byte classification.
    """
    return classify_memory_bytes(abs_bytes, policy=policy)
