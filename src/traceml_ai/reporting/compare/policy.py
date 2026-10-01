# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

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

import math
from dataclasses import dataclass
from typing import Any, Optional

from traceml_ai.regression.contract import (
    ContractValidationError,
    parse_guard_contract,
)
from traceml_ai.regression.outcome import contract_digest

_SIGNIFICANCE_ORDER = {
    "negligible": 0,
    "moderate": 1,
    "material": 2,
}

_STEP_TIME_STATUS_RANK = {
    "NO DATA": 0,
    "WARMUP": 0,
    "INCOMPLETE DATA": 0,
    "BALANCED": 1,
    "INPUT-BOUND": 2,
    "H2D-BOUND": 2,
    "COMPUTE-BOUND": 2,
    "RESIDUAL-HEAVY": 3,
    "INPUT STRAGGLER": 3,
    "COMPUTE STRAGGLER": 3,
    "H2D STRAGGLER": 3,
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
    - `phase_shift_pp_*` thresholds are supporting timing signals.
    - Memory thresholds are supporting signals unless reinforced by a stronger
      memory diagnosis change.
    - The policy is intentionally biased toward abstaining rather than
      overstating a conclusion.
    """

    step_avg_pct_moderate: float = 3.0
    step_avg_pct_material: float = 8.0

    phase_shift_pp_moderate: float = 0.75
    phase_shift_pp_material: float = 2.0

    memory_bytes_moderate: float = 256.0 * 1024.0 * 1024.0
    memory_bytes_material: float = 1.0 * 1024.0 * 1024.0 * 1024.0

    memory_skew_pp_moderate: float = 0.75
    memory_skew_pp_material: float = 2.5


DEFAULT_COMPARE_POLICY = CompareDecisionPolicy()


def parse_step_time_regression_threshold(value: Any) -> float:
    """Return one finite, nonnegative Step Time regression threshold."""
    try:
        threshold = float(value)
    except (OverflowError, TypeError, ValueError):
        threshold = math.nan
    if (
        isinstance(value, bool)
        or not math.isfinite(threshold)
        or threshold < 0
    ):
        raise RuntimeError(
            "maximum Step Time regression percentage must be a finite, "
            "nonnegative number"
        )
    return threshold


def evaluate_step_time_ci_result(
    *, delta_pct: float, threshold_pct: float
) -> str:
    """Classify one existing Step Time percentage difference for CI."""
    if isinstance(delta_pct, bool) or not math.isfinite(delta_pct):
        raise ValueError("Step Time percentage difference must be finite")
    threshold = parse_step_time_regression_threshold(threshold_pct)
    # Percentage division can leave a representational remainder at an
    # inclusive boundary (for example, 0.3 -> 0.315 yields slightly over 5).
    # Keep full precision for real policy decisions and tolerate only that
    # floating-point noise at the declared boundaries.
    if math.isclose(delta_pct, threshold) or math.isclose(
        delta_pct, -threshold
    ):
        return "WITHIN_THRESHOLD_IN_THIS_PAIR"
    if delta_pct > threshold:
        return "SLOWER_IN_THIS_PAIR"
    if delta_pct < -threshold:
        return "FASTER_IN_THIS_PAIR"
    return "WITHIN_THRESHOLD_IN_THIS_PAIR"


def _nested_dict(value: Any, *keys: str) -> dict[str, Any]:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _declaration_digest(payload: dict[str, Any]) -> Optional[str]:
    declaration = _nested_dict(payload, "run_context").get("declaration")
    try:
        # Use the same canonical identity as capture so scalar types remain
        # significant and key ordering does not affect compatibility.
        return contract_digest(parse_guard_contract(declaration))
    except ContractValidationError:
        return None


def _expected_topology(
    payload: dict[str, Any],
) -> Optional[tuple[int, int, int]]:
    execution = _nested_dict(payload, "run_context", "execution")
    topology = tuple(
        execution.get(key)
        for key in (
            "expected_nodes",
            "processes_per_node",
            "expected_world_size",
        )
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in topology
    ):
        return None
    nodes, processes_per_node, world_size = topology
    if world_size != nodes * processes_per_node:
        return None
    return nodes, processes_per_node, world_size


def _positive_finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0.0
    )


def _steps_analyzed(payload: dict[str, Any]) -> Optional[int]:
    value = _nested_dict(payload, "step_time", "global", "window").get(
        "steps_analyzed"
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def extract_step_time_ci_evidence(
    *,
    lhs_payload: dict[str, Any],
    rhs_payload: dict[str, Any],
    compare_payload: dict[str, Any],
) -> dict[str, Any]:
    """Read CI evidence from the existing common-clock comparison."""
    step_time = _nested_dict(compare_payload, "sections", "step_time")
    metric = _nested_dict(step_time, "metrics", "step_time_ms")
    return {
        "selected_clock": step_time.get("comparison_clock"),
        "reference_step_time_ms": metric.get("lhs"),
        "candidate_step_time_ms": metric.get("rhs"),
        "pct_change": metric.get("pct_change"),
        "reference_steps_analyzed": _steps_analyzed(lhs_payload),
        "candidate_steps_analyzed": _steps_analyzed(rhs_payload),
    }


def ci_policy_ineligibility_reason(
    *,
    lhs_payload: dict[str, Any],
    rhs_payload: dict[str, Any],
    compare_payload: dict[str, Any],
) -> Optional[str]:
    """Explain why a summary pair cannot support the v0.1 CI decision.

    This checks only portable final-summary context and the Step Time evidence
    already selected by compare. A ``None`` result means the pair is eligible.
    """
    lhs_declaration = _declaration_digest(lhs_payload)
    rhs_declaration = _declaration_digest(rhs_payload)
    if lhs_declaration is None or rhs_declaration is None:
        return "Both summaries must contain a valid guard declaration."
    if lhs_declaration != rhs_declaration:
        return "The guard declarations do not match."

    lhs_topology = _expected_topology(lhs_payload)
    rhs_topology = _expected_topology(rhs_payload)
    if lhs_topology is None or rhs_topology is None:
        return "Both summaries must contain a valid expected topology."
    if lhs_topology != rhs_topology:
        return "The expected topologies do not match."

    if any(
        _nested_dict(payload, "run_context", "run").get("status")
        != "completed"
        for payload in (lhs_payload, rhs_payload)
    ):
        return "Training must be completed in both summaries."

    if any(
        _nested_dict(
            payload,
            "run_context",
            "execution",
            "launcher_completion",
        ).get("status")
        != "completed"
        for payload in (lhs_payload, rhs_payload)
    ):
        return "Launcher completion must be recorded for both summaries."

    evidence = extract_step_time_ci_evidence(
        lhs_payload=lhs_payload,
        rhs_payload=rhs_payload,
        compare_payload=compare_payload,
    )
    if evidence["selected_clock"] not in {"cpu", "gpu"} or not all(
        _positive_finite(evidence[key])
        for key in ("reference_step_time_ms", "candidate_step_time_ms")
    ):
        return "Positive Step Time is required on a common CPU or GPU clock."
    return None


def build_step_time_ci_policy(
    *,
    lhs_payload: dict[str, Any],
    rhs_payload: dict[str, Any],
    compare_payload: dict[str, Any],
    threshold: Any,
) -> dict[str, Any]:
    """Build the optional v0.1 CI block from existing compare evidence."""
    threshold_pct = parse_step_time_regression_threshold(threshold)
    evidence = extract_step_time_ci_evidence(
        lhs_payload=lhs_payload,
        rhs_payload=rhs_payload,
        compare_payload=compare_payload,
    )
    reason = ci_policy_ineligibility_reason(
        lhs_payload=lhs_payload,
        rhs_payload=rhs_payload,
        compare_payload=compare_payload,
    )
    pct_change = evidence["pct_change"]
    if reason is None and (
        isinstance(pct_change, bool)
        or not isinstance(pct_change, (int, float))
        or not math.isfinite(pct_change)
    ):
        reason = "Step Time percentage difference is unavailable."

    policy = {"threshold_pct": threshold_pct, **evidence}
    if reason is not None:
        policy.update(result="INCONCLUSIVE", reason=reason)
        return policy

    policy["result"] = evaluate_step_time_ci_result(
        delta_pct=pct_change,
        threshold_pct=threshold_pct,
    )
    return policy


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
