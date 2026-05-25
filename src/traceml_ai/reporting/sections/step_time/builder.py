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

from traceml_ai.diagnostics.common import DiagnosticResult
from traceml_ai.diagnostics.step_time.api import StepDiagnosis
from traceml_ai.reporting.schema import (
    BaseGroups,
    BaseSectionPayload,
    GroupRow,
    StepMetadata,
)
from traceml_ai.reporting.sections.step_time.loader import StepTimeSectionData
from traceml_ai.reporting.sections.step_time.model import (
    STEP_TIME_METRIC_NAMES,
    GlobalRankIdentity,
    RankStepSummary,
    build_global_rollup,
    build_overview,
    closest_rank_to_median,
    compute_wait_avg_ms,
    finite_float,
    summary_metric_values,
)
from traceml_ai.reporting.summaries.diagnosis_presentation import (
    diagnosis_presentation_to_json,
    present_step_time_summary_diagnosis,
)
from traceml_ai.reporting.summaries.issue_summary import issues_to_json
from traceml_ai.reporting.summaries.summary_formatting import format_ms
from traceml_ai.reporting.topology import topology_mode_from_identities


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
    total_step: StepTimeMetricPair
    h2d: StepTimeMetricPair
    compute: StepTimeMetricPair
    wait: StepTimeMetricPair
    input: StepTimeMetricPair

    @property
    def is_multi_rank(self) -> bool:
        """Whether the card compares more than one global rank."""
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
        [finite_float(value) for value in rank_to_value.values()],
        dtype=np.float64,
    )
    if values.size == 0:
        return StepTimeMetricPair(None, None, None, None)

    median_value = float(np.median(values))
    median_rank = closest_rank_to_median(rank_to_value)
    worst_global_rank = max(
        rank_to_value,
        key=lambda rank: (finite_float(rank_to_value[rank]), -int(rank)),
    )
    return StepTimeMetricPair(
        median_ms=median_value,
        worst_ms=finite_float(rank_to_value[worst_global_rank]),
        median_global_rank=median_rank,
        worst_global_rank=int(worst_global_rank),
    )


def _build_card_stats(
    per_global_rank_summary: Dict[int, RankStepSummary],
) -> Optional[StepTimeCardStats]:
    """Build the timing values rendered in the final summary card."""
    if not per_global_rank_summary:
        return None

    return StepTimeCardStats(
        global_rank_count=len(per_global_rank_summary),
        total_step=_metric_pair_from_rank_values(
            {
                int(rank): finite_float(summary.avg_total_step_ms)
                for rank, summary in per_global_rank_summary.items()
            }
        ),
        compute=_metric_pair_from_rank_values(
            {
                int(rank): finite_float(summary.avg_gpu_compute_ms)
                for rank, summary in per_global_rank_summary.items()
            }
        ),
        wait=_metric_pair_from_rank_values(
            {
                int(rank): compute_wait_avg_ms(summary)
                for rank, summary in per_global_rank_summary.items()
            }
        ),
        input=_metric_pair_from_rank_values(
            {
                int(rank): finite_float(summary.avg_dataloader_ms)
                for rank, summary in per_global_rank_summary.items()
            }
        ),
        h2d=_metric_pair_from_rank_values(
            {
                int(rank): finite_float(summary.avg_h2d_ms)
                for rank, summary in per_global_rank_summary.items()
            }
        ),
    )


def _format_card_stats(stats: StepTimeCardStats) -> str:
    """Render the compact Step Time `Stats` line."""
    if stats.is_multi_rank:
        total = _format_ms_pair(
            stats.total_step.median_ms,
            stats.total_step.worst_ms,
        )
        compute = _format_ms_pair(
            stats.compute.median_ms,
            stats.compute.worst_ms,
        )
        wait = _format_ms_pair(stats.wait.median_ms, stats.wait.worst_ms)
        input_ms = _format_ms_pair(
            stats.input.median_ms,
            stats.input.worst_ms,
        )
        h2d_ms = _format_ms_pair(
            stats.h2d.median_ms,
            stats.h2d.worst_ms,
        )
        return (
            "- Stats: median/worst | "
            f"total {total} | input {input_ms} | H2D {h2d_ms} | compute {compute} | "
            f"wait {wait}"
        )

    return (
        "- Stats: "
        f"total {format_ms(stats.total_step.worst_ms)} | "
        f"input {format_ms(stats.input.worst_ms)} | "
        f"H2D {format_ms(stats.h2d.worst_ms)} | "
        f"compute {format_ms(stats.compute.worst_ms)} | "
        f"wait {format_ms(stats.wait.worst_ms)}"
    )


def _format_card_ranks(stats: StepTimeCardStats) -> Optional[str]:
    """Render the compact Step Time `Ranks` line for distributed runs."""
    if not stats.is_multi_rank:
        return None
    total = _format_rank_pair(
        stats.total_step.median_global_rank,
        stats.total_step.worst_global_rank,
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
    h2d_rank = _format_rank_pair(
        stats.h2d.median_global_rank,
        stats.h2d.worst_global_rank,
    )
    return (
        "- Ranks: median/worst | "
        f"total {total} | input {input_rank} | H2D {h2d_rank} | compute {compute} | "
        f"wait {wait}"
    )


def _largest_compute_phase(
    summary: Optional[RankStepSummary],
) -> Optional[str]:
    """Return the largest compute bucket for one rank summary."""
    if summary is None:
        return None
    values = {
        "forward": finite_float(summary.avg_forward_ms),
        "backward": finite_float(summary.avg_backward_ms),
        "optimizer": finite_float(summary.avg_optimizer_ms),
    }
    return max(values, key=values.get) if values else None


def _step_time_card_reason(
    diagnosis: Optional[Any],
    *,
    stats: Optional[StepTimeCardStats],
    per_global_rank_summary: Dict[int, RankStepSummary],
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
            f"{format_ms(stats.total_step.worst_ms)}"
        )
        return f"Input loading took a large share ({evidence})."
    if kind == "WAIT_HEAVY":
        evidence = (
            f"{format_ms(stats.wait.worst_ms)}/"
            f"{format_ms(stats.total_step.worst_ms)}"
        )
        return f"Wait was high inside the total step ({evidence})."
    if kind == "COMPUTE_BOUND":
        summary = per_global_rank_summary.get(stats.compute.worst_global_rank)
        phase = _largest_compute_phase(summary)
        suffix = f"; {phase} was largest" if phase else ""
        evidence = (
            f"{format_ms(stats.compute.worst_ms)}/"
            f"{format_ms(stats.total_step.worst_ms)}"
        )
        return f"Compute dominated ({evidence}){suffix}."

    reason = str(getattr(diagnosis, "reason", "") or "").strip()
    return reason or "No clear timing bottleneck."


def _global_rank_entry_to_json(
    global_rank: int,
    summary: RankStepSummary,
    identity: Optional[GlobalRankIdentity] = None,
) -> Dict[str, Any]:
    """Serialize one global-rank row for the Step Time summary."""
    return GroupRow(
        identity={
            "global_rank": int(global_rank),
            "local_rank": identity.local_rank if identity else None,
            "node_rank": identity.node_rank if identity else None,
            "hostname": identity.hostname if identity else None,
            "local_world_size": (
                identity.local_world_size if identity else None
            ),
            "world_size": identity.world_size if identity else None,
        },
        metrics=summary_metric_values(summary),
    ).to_json()


def build_step_time_payload(
    data: StepTimeSectionData,
    diagnosis_result: Optional[DiagnosticResult[StepDiagnosis]],
) -> Dict[str, Any]:
    """
    Build the Step Time payload and compact card text.

    Loading and diagnosis are handled by the section lifecycle. This builder
    only formats the aligned summaries, global comparisons, issues, and card.
    """
    aligned_summary = data.aligned_summary
    all_rank_summary = data.per_global_rank_summary or aligned_summary
    row_rank_summary = aligned_summary
    aligned_window = data.aligned_window
    identities = data.identities

    global_ranks_present = sorted(aligned_summary.keys())
    all_global_ranks = sorted(all_rank_summary.keys())
    # Step Time rows are limited to ranks with data in the common step window.
    overview = build_overview(per_global_rank_summary=aligned_summary)

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

    summary_diag = (
        diagnosis_result.primary if diagnosis_result is not None else None
    )
    summary_diag_presented = present_step_time_summary_diagnosis(summary_diag)
    issues = diagnosis_result.issues if diagnosis_result else ()

    global_rollup = build_global_rollup(
        per_global_rank_summary=aligned_summary,
        median_global_rank=median_global_rank,
        worst_global_rank=worst_global_rank,
        analysis_window=aligned_window,
    )
    card_stats = _build_card_stats(aligned_summary)

    title = (
        f"TraceML Step Timing Summary | steps {data.training_steps} | "
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
        per_global_rank_summary=row_rank_summary,
    )

    if not aligned_summary:
        latest_step_text = data.latest_step_observed or "n/a"
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
        str(rank): _global_rank_entry_to_json(
            rank,
            s,
            identities.get(rank),
        )
        for rank, s in sorted(row_rank_summary.items())
    }

    metadata = StepMetadata(
        mode=topology_mode_from_identities(
            (identities.get(rank) for rank in global_ranks_present),
            has_data=bool(global_ranks_present),
        ),
        global_ranks_seen=len(all_global_ranks),
        global_ranks_used=len(global_ranks_present),
        training_total_steps=data.training_steps,
        training_latest_step=data.latest_step_observed,
        section_metric_names=STEP_TIME_METRIC_NAMES,
    )
    summary = BaseSectionPayload(
        metadata=metadata.to_json(),
        diagnosis=diagnosis_presentation_to_json(
            summary_diag_presented,
            include_action=False,
        ),
        issues=issues_to_json(issues),
        global_summary=global_rollup,
        groups=BaseGroups(
            by="global_rank",
            rows=per_global_rank_json,
        ).to_json(),
        units={"time": "ms"},
        card=card,
    ).to_json()
    return summary


__all__ = [
    "StepTimeCardStats",
    "StepTimeMetricPair",
    "build_step_time_payload",
]
