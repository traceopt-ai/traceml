# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Payload builder for the final-report step-time section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from traceml.diagnostics.step_time.adapters import (
    build_summary_step_diagnosis_result,
)
from traceml.reporting.sections.step_time.alignment import AlignedStepWindow
from traceml.reporting.sections.step_time.loader import StepTimeSectionData
from traceml.reporting.sections.step_time.model import (
    RankStepSummary,
    _build_global_rollup,
    _build_overview,
    _closest_rank_to_median,
    _compute_wait_avg_ms,
    _finite_float,
    _global_rank_entry_to_json,
    _to_rank_signals,
)
from traceml.reporting.summaries.diagnosis_presentation import (
    diagnosis_presentation_to_json,
    present_step_time_summary_diagnosis,
)
from traceml.reporting.summaries.issue_summary import (
    issues_by_rank_json,
    issues_to_json,
)
from traceml.reporting.summaries.summary_formatting import format_ms


@dataclass(frozen=True)
class StepTimeMetricPair:
    """Median and worst global-rank values for one timing bucket."""

    median_ms: Optional[float]
    worst_ms: Optional[float]
    median_global_rank: Optional[int]
    worst_global_rank: Optional[int]


@dataclass(frozen=True)
class StepTimeCardStats:
    """Compact values used by the human final-summary card."""

    global_rank_count: int
    step: StepTimeMetricPair
    compute: StepTimeMetricPair
    wait: StepTimeMetricPair
    input: StepTimeMetricPair

    @property
    def is_multi_rank(self) -> bool:
        return self.global_rank_count > 1


def _global_rank_label(global_rank: Optional[int]) -> str:
    """Format an optional global-rank id for summary-card text."""
    return f"r{int(global_rank)}" if global_rank is not None else "n/a"


def _format_ms_pair(
    left: Optional[float],
    right: Optional[float],
) -> str:
    """Format a compact `median/worst` millisecond pair."""
    if left is None or right is None:
        return "n/a"
    return f"{float(left):.1f}/{float(right):.1f}ms"


def _format_rank_pair(
    left: Optional[int],
    right: Optional[int],
) -> str:
    """Format a compact `median/worst` global-rank pair."""
    return f"{_global_rank_label(left)}/{_global_rank_label(right)}"


def _metric_pair_from_rank_values(
    rank_to_value: Dict[int, float],
) -> StepTimeMetricPair:
    """Return median/worst values and the ranks that best represent them."""
    if not rank_to_value:
        return StepTimeMetricPair(None, None, None, None)

    values = np.asarray(
        [_finite_float(value) for value in rank_to_value.values()],
        dtype=np.float64,
    )
    if values.size == 0:
        return StepTimeMetricPair(None, None, None, None)

    median_value = float(np.median(values))
    median_rank = _closest_rank_to_median(rank_to_value)
    worst_rank = max(
        rank_to_value,
        key=lambda rank: (_finite_float(rank_to_value[rank]), -int(rank)),
    )
    return StepTimeMetricPair(
        median_ms=median_value,
        worst_ms=_finite_float(rank_to_value[worst_rank]),
        median_global_rank=median_rank,
        worst_global_rank=int(worst_rank),
    )


def _build_card_stats(
    per_rank_summary: Dict[int, RankStepSummary],
) -> Optional[StepTimeCardStats]:
    """Build the timing values rendered in the final summary card."""
    if not per_rank_summary:
        return None

    return StepTimeCardStats(
        global_rank_count=len(per_rank_summary),
        step=_metric_pair_from_rank_values(
            {
                int(rank): _finite_float(summary.avg_total_step_ms)
                for rank, summary in per_rank_summary.items()
            }
        ),
        compute=_metric_pair_from_rank_values(
            {
                int(rank): _finite_float(summary.avg_gpu_compute_ms)
                for rank, summary in per_rank_summary.items()
            }
        ),
        wait=_metric_pair_from_rank_values(
            {
                int(rank): _compute_wait_avg_ms(summary)
                for rank, summary in per_rank_summary.items()
            }
        ),
        input=_metric_pair_from_rank_values(
            {
                int(rank): _finite_float(summary.avg_dataloader_ms)
                for rank, summary in per_rank_summary.items()
            }
        ),
    )


def _format_card_stats(stats: StepTimeCardStats) -> str:
    """Render the compact Step Time `Stats` line."""
    if stats.is_multi_rank:
        step = _format_ms_pair(stats.step.median_ms, stats.step.worst_ms)
        compute = _format_ms_pair(
            stats.compute.median_ms,
            stats.compute.worst_ms,
        )
        wait = _format_ms_pair(stats.wait.median_ms, stats.wait.worst_ms)
        input_ms = _format_ms_pair(
            stats.input.median_ms,
            stats.input.worst_ms,
        )
        return (
            "- Stats: median/worst | "
            f"step {step} | compute {compute} | "
            f"wait {wait} | input {input_ms}"
        )

    return (
        "- Stats: "
        f"step {format_ms(stats.step.worst_ms)} | "
        f"compute {format_ms(stats.compute.worst_ms)} | "
        f"wait {format_ms(stats.wait.worst_ms)} | "
        f"input {format_ms(stats.input.worst_ms)}"
    )


def _format_card_ranks(stats: StepTimeCardStats) -> Optional[str]:
    """Render the compact Step Time `Ranks` line for distributed runs."""
    if not stats.is_multi_rank:
        return None
    step = _format_rank_pair(
        stats.step.median_global_rank,
        stats.step.worst_global_rank,
    )
    compute = _format_rank_pair(
        stats.compute.median_global_rank,
        stats.compute.worst_global_rank,
    )
    wait = _format_rank_pair(
        stats.wait.median_global_rank,
        stats.wait.worst_global_rank,
    )
    input_rank = _format_rank_pair(
        stats.input.median_global_rank,
        stats.input.worst_global_rank,
    )
    return (
        "- Ranks: median/worst | "
        f"step {step} | compute {compute} | "
        f"wait {wait} | input {input_rank}"
    )


def _largest_compute_phase(
    summary: Optional[RankStepSummary],
) -> Optional[str]:
    """Return the largest compute bucket for one rank summary."""
    if summary is None:
        return None
    values = {
        "forward": _finite_float(summary.avg_forward_ms),
        "backward": _finite_float(summary.avg_backward_ms),
        "optimizer": _finite_float(summary.avg_optimizer_ms),
    }
    return max(values, key=values.get) if values else None


def _step_time_card_reason(
    diagnosis: Optional[Any],
    *,
    stats: Optional[StepTimeCardStats],
    per_rank_summary: Dict[int, RankStepSummary],
) -> str:
    """Build the short `Why` line used only by the human card."""
    kind = str(getattr(diagnosis, "kind", "") or "")
    if diagnosis is None or kind == "NO_DATA":
        return "Need more step-time samples."
    if kind == "WARMUP":
        return str(getattr(diagnosis, "reason", "") or "").strip()
    if stats is None:
        return str(getattr(diagnosis, "reason", "") or "n/a")

    if kind == "BALANCED":
        return "No clear timing bottleneck."
    if kind == "INPUT_STRAGGLER":
        evidence = _format_ms_pair(stats.input.worst_ms, stats.input.median_ms)
        return (
            f"{_global_rank_label(stats.input.worst_global_rank)} input was "
            f"slower than median global rank ({evidence})."
        )
    if kind == "COMPUTE_STRAGGLER":
        evidence = _format_ms_pair(
            stats.compute.worst_ms,
            stats.compute.median_ms,
        )
        return (
            f"{_global_rank_label(stats.compute.worst_global_rank)} compute "
            f"was slower than median global rank ({evidence})."
        )
    if kind == "STRAGGLER":
        return "Input and compute varied across ranks."
    if kind == "INPUT_BOUND":
        evidence = (
            f"{format_ms(stats.input.worst_ms)}/"
            f"{format_ms(stats.step.worst_ms)}"
        )
        return f"Input loading took a large share ({evidence})."
    if kind == "WAIT_HEAVY":
        evidence = (
            f"{format_ms(stats.wait.worst_ms)}/"
            f"{format_ms(stats.step.worst_ms)}"
        )
        return f"Wait time was high ({evidence})."
    if kind == "COMPUTE_BOUND":
        summary = per_rank_summary.get(stats.compute.worst_global_rank)
        phase = _largest_compute_phase(summary)
        suffix = f"; {phase} was largest" if phase else ""
        evidence = (
            f"{format_ms(stats.compute.worst_ms)}/"
            f"{format_ms(stats.step.worst_ms)}"
        )
        return f"Compute dominated ({evidence}){suffix}."

    reason = str(getattr(diagnosis, "reason", "") or "").strip()
    return reason or "No clear timing bottleneck."


def _metric_rollup_to_json(metric: StepTimeMetricPair) -> Dict[str, Any]:
    """Serialize median/worst/skew for one timing metric."""
    skew_percent = None
    if metric.median_ms is not None and metric.median_ms > 0.0:
        skew_percent = (
            100.0
            * (_finite_float(metric.worst_ms) - metric.median_ms)
            / metric.median_ms
        )
    return {
        "median": metric.median_ms,
        "worst": metric.worst_ms,
        "median_global_rank": metric.median_global_rank,
        "worst_global_rank": metric.worst_global_rank,
        "skew_percent": skew_percent,
    }


def _build_global_rank_rollup(
    per_global_rank_summary: Dict[int, RankStepSummary],
) -> Dict[str, Dict[str, Any]]:
    """Build metric-by-metric median/worst/skew across global ranks."""
    stats = _build_card_stats(per_global_rank_summary)
    if stats is None:
        empty = _metric_rollup_to_json(
            StepTimeMetricPair(None, None, None, None)
        )
        return {
            "step_time_ms": dict(empty),
            "compute_ms": dict(empty),
            "wait_ms": dict(empty),
            "input_ms": dict(empty),
        }
    return {
        "step_time_ms": _metric_rollup_to_json(stats.step),
        "compute_ms": _metric_rollup_to_json(stats.compute),
        "wait_ms": _metric_rollup_to_json(stats.wait),
        "input_ms": _metric_rollup_to_json(stats.input),
    }


def _default_aligned_window(
    aligned_summary: Dict[int, RankStepSummary],
) -> AlignedStepWindow:
    """Return a conservative aligned window for direct builder callers."""
    steps = [
        int(summary.steps_analyzed) for summary in aligned_summary.values()
    ]
    analyzed = min(steps) if steps else 0
    return AlignedStepWindow(
        alignment="common_steps",
        steps_analyzed=analyzed,
        start_step=None,
        end_step=None,
        global_ranks_used=len(aligned_summary),
        global_ranks_observed=len(aligned_summary),
    )


def build_step_time_card(
    *,
    training_steps: int,
    latest_step_observed: Optional[int],
    aligned_summary: Dict[int, RankStepSummary],
    aligned_step_metrics: Dict[int, Dict[int, Dict[str, float]]],
    per_global_rank_summary: Optional[Dict[int, RankStepSummary]] = None,
    aligned_window: Optional[AlignedStepWindow] = None,
    identities: Optional[Dict[int, Any]] = None,
    max_rows: int,
) -> tuple[str, Dict[str, Any]]:
    """Build the Step Time section payload and compact card text."""
    per_rank_summary = per_global_rank_summary or aligned_summary
    aligned_window = aligned_window or _default_aligned_window(aligned_summary)
    identities = identities or {}

    global_ranks_present = sorted(aligned_summary.keys())
    all_global_ranks = sorted(per_rank_summary.keys())
    overview = _build_overview(per_global_rank_summary=aligned_summary)

    median_global_rank = overview["median_global_rank"]
    worst_global_rank = overview["worst_global_rank"]
    median_summary = (
        aligned_summary.get(median_global_rank)
        if median_global_rank is not None
        else None
    )
    worst_summary = (
        aligned_summary.get(worst_global_rank)
        if worst_global_rank is not None
        else None
    )
    primary_summary = median_summary or worst_summary

    summary_diag_result = build_summary_step_diagnosis_result(
        rank_signals=_to_rank_signals(aligned_summary),
        max_rows=max_rows,
        per_rank_step_metrics=aligned_step_metrics,
    )
    summary_diag = (
        summary_diag_result.primary
        if summary_diag_result is not None
        else None
    )
    summary_diag_presented = present_step_time_summary_diagnosis(summary_diag)
    issues = summary_diag_result.issues if summary_diag_result else ()
    issues_by_global_rank, _unassigned_issues = issues_by_rank_json(
        issues,
        rank_keys=global_ranks_present,
    )

    global_rollup = _build_global_rollup(
        per_global_rank_summary=aligned_summary,
        median_global_rank=median_global_rank,
        worst_global_rank=worst_global_rank,
        analysis_window=aligned_window,
    )
    global_rank_rollup = _build_global_rank_rollup(aligned_summary)
    card_stats = _build_card_stats(aligned_summary)

    title = (
        f"TraceML Step Timing Summary | steps {training_steps} | "
        f"global ranks {len(all_global_ranks)}"
    )
    lines = [title, "Step Time"]
    diagnosis_status = (
        summary_diag_presented.status
        if summary_diag_presented is not None
        else "NO DATA"
    )
    diagnosis_why = _step_time_card_reason(
        summary_diag,
        stats=card_stats,
        per_rank_summary=per_rank_summary,
    )

    if not aligned_summary:
        latest_step_text = latest_step_observed or "n/a"
        lines.extend(
            [
                f"- Diagnosis: {diagnosis_status}",
                f"- Scope: latest step {latest_step_text}",
                "- Stats: n/a",
                f"- Why: {diagnosis_why}",
            ]
        )
    elif len(global_ranks_present) == 1 and primary_summary is not None:
        only_rank = global_ranks_present[0]
        lines.extend(
            [
                f"- Diagnosis: {diagnosis_status}",
                (
                    f"- Scope: last {primary_summary.steps_analyzed} "
                    f"aligned steps on global rank r{only_rank}"
                ),
            ]
        )
        if card_stats is not None:
            lines.append(_format_card_stats(card_stats))
        lines.append(f"- Why: {diagnosis_why}")
    else:
        lines.append(f"- Diagnosis: {diagnosis_status}")
        lines.append(
            "- Scope: compared over "
            f"last {aligned_window.steps_analyzed} aligned steps "
            f"across {aligned_window.global_ranks_used} global ranks"
        )
        if card_stats is not None:
            lines.append(_format_card_stats(card_stats))
            ranks_line = _format_card_ranks(card_stats)
            if ranks_line:
                lines.append(ranks_line)
        lines.append(f"- Why: {diagnosis_why}")

    card = "\n".join(lines)
    per_global_rank_json = {
        str(rank): {
            **_global_rank_entry_to_json(
                rank,
                s,
                identities.get(rank),
            ),
            "issues": issues_by_global_rank.get(str(rank), []),
        }
        for rank, s in sorted(per_rank_summary.items())
    }

    summary = {
        "overview": {
            "mode": overview["mode"],
            "training_steps": training_steps,
            "latest_step_observed": latest_step_observed,
            "global_ranks_seen": len(all_global_ranks),
            "global_ranks_used": len(global_ranks_present),
            "aligned_steps_analyzed": int(aligned_window.steps_analyzed),
        },
        "primary_diagnosis": diagnosis_presentation_to_json(
            summary_diag_presented,
            include_action=False,
        ),
        "issues": issues_to_json(issues),
        "issues_by_global_rank": issues_by_global_rank,
        "global": global_rollup,
        "global_rank_rollup": global_rank_rollup,
        "per_global_rank": per_global_rank_json,
        "units": {"time": "ms", "skew": "%"},
        "card": card,
    }
    return card, summary


def build_step_time_section_payload(
    data: StepTimeSectionData,
) -> Dict[str, Any]:
    """Build the JSON-safe step-time section payload from loaded data."""
    _, payload = build_step_time_card(
        training_steps=data.training_steps,
        latest_step_observed=data.latest_step_observed,
        aligned_summary=data.aligned_summary,
        aligned_step_metrics=data.aligned_step_metrics,
        aligned_window=data.aligned_window,
        per_global_rank_summary=data.per_global_rank_summary,
        identities=data.identities,
        max_rows=data.max_rows,
    )
    return payload


__all__ = [
    "StepTimeCardStats",
    "StepTimeMetricPair",
    "build_step_time_card",
    "build_step_time_section_payload",
]
