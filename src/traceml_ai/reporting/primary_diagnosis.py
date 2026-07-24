# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Primary performance diagnosis for final run summaries.

The final summary already contains section-local diagnoses:

- ``system.diagnosis``
- ``process.diagnosis``
- ``step_time.diagnosis``
- ``step_memory.diagnosis``

Those section diagnoses remain the detailed source of truth for their domain.
This module adds one run-level finding, ``primary_diagnosis``, for the first
question most users ask after a run: "why was training slow?"

The implementation is intentionally a promotion policy over existing final
summary JSON. It does not read telemetry databases, query SQLite tables, or
recompute diagnostics. Keeping the logic here pure makes the policy easy to
test and easy for contributors to evolve.

Primary diagnosis policy
------------------------
1. Step-time rank-skew findings become primary performance findings:
   ``INPUT_STRAGGLER``, ``COMPUTE_STRAGGLER``, ``H2D_STRAGGLER``,
   ``STRAGGLER``.
2. Step-time phase-share findings become primary performance findings:
   ``RESIDUAL_HEAVY``, ``INPUT_BOUND``, ``H2D_BOUND``, ``COMPUTE_BOUND``.
3. If Step Time is ``BALANCED`` and System reports low or moderate GPU
   utilization, the primary becomes ``LOW_GPU_UTILIZATION_UNEXPLAINED``.
   GPU utilization is treated as a symptom or fallback, not root-cause proof.
4. If Step Time is ``BALANCED`` and GPU utilization is not low/moderate, the
   primary becomes ``NO_CLEAR_PERFORMANCE_BOTTLENECK``.
5. If Step Time is ``NO_DATA`` or ``WARMUP``, the primary becomes
   ``INSUFFICIENT_STEP_TIME_DATA``.

The v1 policy deliberately does not promote System, Process, or Step Memory
health/resource findings such as high GPU temperature, memory pressure, memory
creep, high RSS, or high CPU. Those findings can matter, but they do not by
themselves prove why step time was slow. They stay visible in their section
diagnoses. If TraceML needs top-level health surfacing later, add a separate
``run_health_warnings`` style field instead of overloading the performance
primary.

Evidence policy
---------------
``phase_share``
    Used for ``INPUT_BOUND``, ``H2D_BOUND``, ``RESIDUAL_HEAVY``, and
    ``COMPUTE_BOUND``. Values come from ``step_time.global.average`` because
    the diagnosis describes where the average step time went.

``rank_comparison``
    Used for ``INPUT_STRAGGLER``, ``COMPUTE_STRAGGLER``,
    ``H2D_STRAGGLER``, and ``STRAGGLER``. Values come from step-time rank
    summaries because the diagnosis compares ranks.

``utilization_fallback``
    Used only when Step Time is balanced and System GPU utilization is low or
    moderate. This says utilization is unexplained by Step Time, not that GPU
    utilization itself is the root cause.

``no_clear_bottleneck``
    Used when Step Time has enough data and no material performance bottleneck
    is found.

``insufficient_data``
    Used when Step Time cannot make a stable diagnosis.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

JsonDict = Dict[str, Any]

STEP_TIME_SECTION = "step_time"
PERFORMANCE_SCOPE = "performance"

PHASE_SHARE_KINDS = {
    "INPUT_BOUND",
    "H2D_BOUND",
    "RESIDUAL_HEAVY",
    "COMPUTE_BOUND",
}
STRAGGLER_KINDS = {
    "INPUT_STRAGGLER",
    "COMPUTE_STRAGGLER",
    "H2D_STRAGGLER",
    "STRAGGLER",
}
INSUFFICIENT_STEP_TIME_KINDS = {"NO_DATA", "WARMUP"}
LOW_GPU_UTIL_KINDS = {
    "LOW_GPU_UTILIZATION",
    "MODERATE_GPU_UTILIZATION",
}

PHASE_METRICS = (
    "input_wait_ms",
    "h2d_ms",
    "compute_ms",
    "residual_ms",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping or an empty mapping for malformed payload blocks."""
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    """Return a sequence or an empty tuple for malformed payload blocks."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    return ()


def _float_or_none(value: Any) -> Optional[float]:
    """Return a finite-enough float for summary evidence, else ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _int_or_none(value: Any) -> Optional[int]:
    """Return an integer rank/index value when one is available."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _round(value: Optional[float], ndigits: int = 3) -> Optional[float]:
    """Round numeric evidence without hiding missing values."""
    if value is None:
        return None
    return round(float(value), ndigits)


def _diagnosis(section: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a section's primary diagnosis block."""
    return _mapping(section.get("diagnosis"))


def _global(section: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a section's global rollup block."""
    return _mapping(section.get("global"))


def _global_average(section: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a section's global average metric block."""
    return _mapping(_global(section).get("average"))


def _global_window(section: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a section's global analysis window block."""
    return _mapping(_global(section).get("window"))


def _point(
    section: Mapping[str, Any],
    *,
    block: str,
    metric: str,
) -> Mapping[str, Any]:
    """Return one ``global.median/worst[metric]`` point."""
    return _mapping(_mapping(_global(section).get(block)).get(metric))


def _gpu_util_avg(system_summary: Mapping[str, Any]) -> Optional[float]:
    """Return average system GPU utilization when available."""
    return _float_or_none(
        _global_average(system_summary).get("gpu_util_percent")
    )


def _steps_analyzed(step_time_summary: Mapping[str, Any]) -> Optional[int]:
    """Return the number of aligned steps behind the step-time summary."""
    return _int_or_none(
        _global_window(step_time_summary).get("steps_analyzed")
    )


def _diag_field(
    diagnosis: Mapping[str, Any],
    key: str,
    default: str = "",
) -> str:
    """Read a string diagnosis field."""
    value = diagnosis.get(key)
    return str(value) if value is not None else default


def _primary_payload(
    *,
    kind: str,
    status: str,
    severity: str,
    section: str,
    summary: str,
    action: str,
    evidence: Mapping[str, Any],
) -> JsonDict:
    """Build the stable top-level primary diagnosis payload."""
    return {
        "kind": str(kind),
        "status": str(status),
        "severity": str(severity),
        "section": str(section),
        "scope": PERFORMANCE_SCOPE,
        "summary": str(summary),
        "action": str(action),
        "evidence": dict(evidence),
    }


def _phase_share_evidence(
    *,
    step_time_summary: Mapping[str, Any],
    system_summary: Mapping[str, Any],
) -> JsonDict:
    """Build evidence with selected-clock iteration-time phase shares."""
    average = _global_average(step_time_summary)
    total_ms = _float_or_none(average.get("total_step_ms"))
    step_time_ms = _float_or_none(average.get("step_time_ms"))
    input_wait_ms = _float_or_none(average.get("input_wait_ms"))
    iteration_time_ms = (
        input_wait_ms + step_time_ms
        if input_wait_ms is not None and step_time_ms is not None
        else None
    )
    window = _global_window(step_time_summary)
    evidence: JsonDict = {
        "type": "phase_share",
        "basis": "average",
        "steps_analyzed": _steps_analyzed(step_time_summary),
        "total_step_ms": _round(total_ms),
        "step_time_ms": _round(step_time_ms),
        "diagnosis_clock": window.get("diagnosis_clock"),
        "dataloader_ms": _round(_float_or_none(average.get("dataloader_ms"))),
    }
    evidence["iteration_time_ms"] = _round(iteration_time_ms)

    for metric in PHASE_METRICS:
        evidence[metric] = _round(_float_or_none(average.get(metric)))

    shares: JsonDict = {}
    for metric in PHASE_METRICS:
        value = _float_or_none(average.get(metric))
        key = metric.replace("_ms", "_pct")
        shares[key] = (
            _round(100.0 * value / iteration_time_ms)
            if (
                value is not None
                and iteration_time_ms
                and iteration_time_ms > 0.0
            )
            else None
        )
    evidence["shares"] = shares
    evidence["gpu_util_avg_percent"] = _round(_gpu_util_avg(system_summary))
    return evidence


def _metric_for_step_time_issue(issue: Mapping[str, Any]) -> Optional[str]:
    """Map one step-time issue to the public metric used for comparison."""
    kind = str(issue.get("kind") or "")
    if kind == "INPUT_STRAGGLER":
        return "input_wait_ms"
    if kind == "COMPUTE_STRAGGLER":
        phase = str(issue.get("phase") or "").lower()
        if phase in {"forward", "backward", "optimizer"}:
            return f"{phase}_ms"
        return "compute_ms"
    if kind == "H2D_STRAGGLER":
        return "h2d_ms"
    return None


def _metric_for_primary_diagnosis(
    diagnosis: Mapping[str, Any],
) -> Optional[str]:
    """Map a primary step-time diagnosis to the public comparison metric."""
    kind = str(diagnosis.get("kind") or "")
    if kind == "INPUT_STRAGGLER":
        return "input_wait_ms"
    if kind == "COMPUTE_STRAGGLER":
        phase = str(diagnosis.get("phase") or "").lower()
        if phase in {"forward", "backward", "optimizer"}:
            return f"{phase}_ms"
        return "compute_ms"
    if kind == "H2D_STRAGGLER":
        return "h2d_ms"
    return None


def _rank_point_json(point: Mapping[str, Any]) -> JsonDict:
    """Convert a global point into the primary evidence rank/value shape."""
    return {
        "rank": _int_or_none(point.get("idx")),
        "value_ms": _round(_float_or_none(point.get("value"))),
    }


def _comparison(
    *,
    step_time_summary: Mapping[str, Any],
    metric: str,
    phase: Optional[str] = None,
) -> JsonDict:
    """Build a median-vs-worst comparison for one step-time metric."""
    median = _rank_point_json(
        _point(step_time_summary, block="median", metric=metric)
    )
    worst = _rank_point_json(
        _point(step_time_summary, block="worst", metric=metric)
    )
    median_value = _float_or_none(median.get("value_ms"))
    worst_value = _float_or_none(worst.get("value_ms"))
    delta_ms = (
        worst_value - median_value
        if worst_value is not None and median_value is not None
        else None
    )
    ratio = (
        worst_value / median_value
        if worst_value is not None
        and median_value is not None
        and median_value > 0.0
        else None
    )
    return {
        "metric": metric,
        "phase": phase or metric.replace("_ms", ""),
        "median": median,
        "worst": worst,
        "delta_ms": _round(delta_ms),
        "ratio": _round(ratio),
    }


def _rank_comparison_evidence(
    *,
    step_time_summary: Mapping[str, Any],
    system_summary: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
) -> JsonDict:
    """Build evidence for diagnoses based on cross-rank comparisons."""
    kind = str(diagnosis.get("kind") or "")
    evidence: JsonDict = {
        "type": "rank_comparison",
        "steps_analyzed": _steps_analyzed(step_time_summary),
        "gpu_util_avg_percent": _round(_gpu_util_avg(system_summary)),
    }

    if kind == "STRAGGLER":
        comparisons = _straggler_comparisons(
            step_time_summary=step_time_summary,
            issues=_sequence(step_time_summary.get("issues")),
        )
        evidence["comparisons"] = comparisons
        return evidence

    metric = _metric_for_primary_diagnosis(diagnosis)
    if metric is None:
        metric = "total_step_ms"
    comparison = _comparison(
        step_time_summary=step_time_summary,
        metric=metric,
        phase=str(diagnosis.get("phase") or metric.replace("_ms", "")),
    )
    evidence.update(comparison)
    return evidence


def _straggler_comparisons(
    *,
    step_time_summary: Mapping[str, Any],
    issues: Sequence[Any],
) -> list[JsonDict]:
    """Return comparisons for the atomic issues behind a STRAGGLER primary."""
    out: list[JsonDict] = []
    seen: set[str] = set()
    for raw_issue in issues:
        issue = _mapping(raw_issue)
        metric = _metric_for_step_time_issue(issue)
        if metric is None or metric in seen:
            continue
        seen.add(metric)
        out.append(
            _comparison(
                step_time_summary=step_time_summary,
                metric=metric,
                phase=str(issue.get("phase") or metric.replace("_ms", "")),
            )
        )

    if out:
        return out

    return [
        _comparison(
            step_time_summary=step_time_summary,
            metric="input_wait_ms",
            phase="input",
        ),
        _comparison(
            step_time_summary=step_time_summary,
            metric="compute_ms",
            phase="compute",
        ),
        _comparison(
            step_time_summary=step_time_summary,
            metric="h2d_ms",
            phase="h2d",
        ),
        _comparison(
            step_time_summary=step_time_summary,
            metric="residual_ms",
            phase="residual",
        ),
    ]


def _phase_share_summary(kind: str, evidence: Mapping[str, Any]) -> str:
    """Return a concise primary summary for phase-share diagnoses."""
    total = _float_or_none(evidence.get("step_time_ms")) or _float_or_none(
        evidence.get("total_step_ms")
    )
    if kind == "INPUT_BOUND":
        value = _float_or_none(evidence.get("input_wait_ms"))
        iteration_time = _float_or_none(evidence.get("iteration_time_ms"))
        if value is not None and iteration_time is not None:
            return (
                f"Input wait was {value:.1f}ms of "
                f"{iteration_time:.1f}ms iteration time."
            )
        return "Input wait took a large share of iteration time."
    if kind == "H2D_BOUND":
        value = _float_or_none(evidence.get("h2d_ms"))
        iteration_time = _float_or_none(evidence.get("iteration_time_ms"))
        if value is not None and iteration_time is not None:
            return (
                f"H2D transfer took {value:.1f}ms of "
                f"{iteration_time:.1f}ms iteration time."
            )
        return "H2D transfer took a large share of iteration time."
    if kind == "RESIDUAL_HEAVY":
        value = _float_or_none(evidence.get("residual_ms"))
        if value is not None and total is not None:
            return (
                f"Residual time took {value:.1f}ms of a "
                f"{total:.1f}ms average step."
            )
        return "Residual time took a large share of step time."
    if kind == "COMPUTE_BOUND":
        value = _float_or_none(evidence.get("compute_ms"))
        if value is not None and total is not None:
            return (
                f"Model compute took {value:.1f}ms of a "
                f"{total:.1f}ms average step."
            )
        return "Most step time was model compute."
    return "Step time was dominated by one phase."


def _rank_comparison_summary(
    kind: str,
    evidence: Mapping[str, Any],
) -> str:
    """Return a concise primary summary for rank-comparison diagnoses."""
    if kind == "STRAGGLER":
        return "Visible rank skew was sync-bound or unattributed."

    metric = str(evidence.get("metric") or "step_time")
    phase = str(evidence.get("phase") or metric.replace("_ms", ""))
    phase_label = {
        "dataloader": "input wait",
        "input": "input wait",
        "residual": "residual time",
    }.get(phase, phase)
    median = _mapping(evidence.get("median"))
    worst = _mapping(evidence.get("worst"))
    worst_rank = _int_or_none(worst.get("rank"))
    median_rank = _int_or_none(median.get("rank"))
    worst_value = _float_or_none(worst.get("value_ms"))
    median_value = _float_or_none(median.get("value_ms"))

    if (
        worst_rank is not None
        and median_rank is not None
        and worst_value is not None
        and median_value is not None
    ):
        return (
            f"Rank r{worst_rank} {phase_label} was {worst_value:.1f}ms "
            f"vs median rank r{median_rank} at {median_value:.1f}ms."
        )
    return "One rank was materially slower than its peers."


def _promote_step_time_primary(
    *,
    kind: str,
    system_summary: Mapping[str, Any],
    step_time_summary: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
) -> JsonDict:
    """Promote an actionable Step Time diagnosis to primary diagnosis."""
    if kind in PHASE_SHARE_KINDS:
        evidence = _phase_share_evidence(
            step_time_summary=step_time_summary,
            system_summary=system_summary,
        )
        summary = _phase_share_summary(kind, evidence)
    else:
        evidence = _rank_comparison_evidence(
            step_time_summary=step_time_summary,
            system_summary=system_summary,
            diagnosis=diagnosis,
        )
        summary = _rank_comparison_summary(kind, evidence)

    return _primary_payload(
        kind=kind,
        status=_diag_field(diagnosis, "status", kind),
        severity=_diag_field(diagnosis, "severity", "info"),
        section=STEP_TIME_SECTION,
        summary=summary or _diag_field(diagnosis, "summary", ""),
        action=_diag_field(diagnosis, "action", ""),
        evidence=evidence,
    )


def _insufficient_step_time_primary(
    diagnosis: Mapping[str, Any],
    *,
    step_time_summary: Mapping[str, Any],
    system_summary: Mapping[str, Any],
) -> JsonDict:
    """Build the primary fallback for missing or unstable timing data."""
    evidence = {
        "type": "insufficient_data",
        "step_time_status": _diag_field(diagnosis, "status", "NO DATA"),
        "steps_analyzed": _steps_analyzed(step_time_summary),
        "gpu_util_avg_percent": _round(_gpu_util_avg(system_summary)),
    }
    return _primary_payload(
        kind="INSUFFICIENT_STEP_TIME_DATA",
        status="INSUFFICIENT STEP-TIME DATA",
        severity="info",
        section=STEP_TIME_SECTION,
        summary=(
            "Not enough completed step-time samples were available for a "
            "stable performance diagnosis."
        ),
        action="Run for more steps or ensure step timing is recorded.",
        evidence=evidence,
    )


def _low_gpu_util_unexplained_primary(
    *,
    system_summary: Mapping[str, Any],
    step_time_summary: Mapping[str, Any],
    step_time_diagnosis: Mapping[str, Any],
) -> JsonDict:
    """Build the primary fallback for low GPU utilization without a cause."""
    evidence = {
        "type": "utilization_fallback",
        "gpu_util_avg_percent": _round(_gpu_util_avg(system_summary)),
        "step_time_status": _diag_field(
            step_time_diagnosis,
            "status",
            "BALANCED",
        ),
        "steps_analyzed": _steps_analyzed(step_time_summary),
    }
    return _primary_payload(
        kind="LOW_GPU_UTILIZATION_UNEXPLAINED",
        status="LOW GPU UTILIZATION",
        severity="info",
        section="system",
        summary=(
            "GPU utilization was low, but step timing did not identify input, "
            "residual time, compute-skew, or rank-skew as the cause."
        ),
        action=(
            "Inspect untraced work, validation/checkpointing, kernel "
            "efficiency, or missing instrumentation."
        ),
        evidence=evidence,
    )


def _no_clear_bottleneck_primary(
    *,
    system_summary: Mapping[str, Any],
    step_time_summary: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
) -> JsonDict:
    """Build the primary fallback when step timing is healthy enough."""
    evidence = _phase_share_evidence(
        step_time_summary=step_time_summary,
        system_summary=system_summary,
    )
    evidence["type"] = "no_clear_bottleneck"
    evidence["step_time_status"] = _diag_field(diagnosis, "status", "BALANCED")
    return _primary_payload(
        kind="NO_CLEAR_PERFORMANCE_BOTTLENECK",
        status="NO CLEAR PERFORMANCE BOTTLENECK",
        severity="info",
        section=STEP_TIME_SECTION,
        summary=(
            "Step timing did not show material input, residual time, "
            "compute-skew, or rank-skew bottlenecks."
        ),
        action=(
            "No data-pipeline or rank-skew bottleneck was detected; use "
            "model/kernel-level profiling if more speed is needed."
        ),
        evidence=evidence,
    )


def build_primary_diagnosis(
    *,
    system_summary: Mapping[str, Any],
    process_summary: Mapping[str, Any],
    step_time_summary: Mapping[str, Any],
    step_memory_summary: Mapping[str, Any],
) -> JsonDict:
    """
    Build the top-level primary performance diagnosis.

    Parameters are already-built section payloads from the final report. The
    process and step-memory summaries are accepted now to keep the public
    function signature aligned with the full report, even though v1 does not
    promote their health/resource findings into the primary performance
    diagnosis.
    """
    del process_summary, step_memory_summary

    step_diag = _diagnosis(step_time_summary)
    kind = _diag_field(step_diag, "kind", "NO_DATA")

    # The primary diagnosis is performance-first. Step Time is the only direct
    # source of root-cause candidates in v1; System GPU utilization is a
    # fallback symptom when Step Time is balanced.
    if kind in STRAGGLER_KINDS or kind in PHASE_SHARE_KINDS:
        return _promote_step_time_primary(
            kind=kind,
            system_summary=system_summary,
            step_time_summary=step_time_summary,
            diagnosis=step_diag,
        )

    if kind in INSUFFICIENT_STEP_TIME_KINDS:
        return _insufficient_step_time_primary(
            step_diag,
            step_time_summary=step_time_summary,
            system_summary=system_summary,
        )

    system_kind = _diag_field(_diagnosis(system_summary), "kind", "")
    if kind == "BALANCED" and system_kind in LOW_GPU_UTIL_KINDS:
        return _low_gpu_util_unexplained_primary(
            system_summary=system_summary,
            step_time_summary=step_time_summary,
            step_time_diagnosis=step_diag,
        )

    return _no_clear_bottleneck_primary(
        system_summary=system_summary,
        step_time_summary=step_time_summary,
        diagnosis=step_diag,
    )


__all__ = [
    "build_primary_diagnosis",
]
