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
    finite_float,
    finite_float_or_none,
    summary_metric_values,
)
from traceml_ai.reporting.summaries.issue_summary import (
    diagnostic_result_to_json,
)
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
    residual: StepTimeMetricPair
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

    def _measured(field: str) -> Dict[int, float]:
        # Ranks whose signal was never measured stay out of the card
        # statistics; a measured zero stays in.
        out: Dict[int, float] = {}
        for rank, summary in per_global_rank_summary.items():
            value = finite_float_or_none(getattr(summary, field))
            if value is not None:
                out[int(rank)] = value
        return out

    return StepTimeCardStats(
        global_rank_count=len(per_global_rank_summary),
        total_step=_metric_pair_from_rank_values(
            _measured("avg_total_step_ms")
        ),
        compute=_metric_pair_from_rank_values(_measured("avg_compute_ms")),
        residual=_metric_pair_from_rank_values(_measured("avg_residual_ms")),
        input=_metric_pair_from_rank_values(_measured("avg_input_wait_ms")),
        h2d=_metric_pair_from_rank_values(_measured("avg_h2d_ms")),
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
        residual = _format_ms_pair(
            stats.residual.median_ms,
            stats.residual.worst_ms,
        )
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
            f"total {total} | input {input_ms} | H2D {h2d_ms} | "
            f"compute {compute}\n"
            f"- Residual: median/worst {residual}"
        )

    return (
        "- Stats: "
        f"total {format_ms(stats.total_step.worst_ms)} | "
        f"input {format_ms(stats.input.worst_ms)} | "
        f"H2D {format_ms(stats.h2d.worst_ms)} | "
        f"compute {format_ms(stats.compute.worst_ms)}\n"
        f"- Residual: {format_ms(stats.residual.worst_ms)}"
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
    residual = _format_rank_pair(
        stats.residual.median_global_rank,
        stats.residual.worst_global_rank,
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
        f"total {total} | input {input_rank} | H2D {h2d_rank} | "
        f"compute {compute}\n"
        f"- Residual ranks: median/worst {residual}"
    )


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
    diagnosis_result: DiagnosticResult[StepDiagnosis],
) -> Dict[str, Any]:
    """
    Build the Step Time payload and compact card text.

    Loading and diagnosis are handled by the section lifecycle. This builder
    only formats the aligned summaries, global comparisons, issues, and card.
    """
    rank_summary = data.per_global_rank_summary
    step_time_window = data.step_time_window
    identities = data.identities

    global_ranks_used = sorted(rank_summary.keys())
    global_ranks_seen = sorted(set(identities.keys()) | set(global_ranks_used))
    # Step Time rows are limited to ranks with data in the common step window.
    overview = build_overview(per_global_rank_summary=rank_summary)

    median_global_rank = overview["median_global_rank"]
    worst_global_rank = overview["worst_global_rank"]

    summary_diag = diagnosis_result.primary
    issues = diagnosis_result.issues
    diagnosis_json, issues_json = diagnostic_result_to_json(diagnosis_result)
    primary_issue = issues[0]

    global_rollup = build_global_rollup(
        per_global_rank_summary=rank_summary,
        median_global_rank=median_global_rank,
        worst_global_rank=worst_global_rank,
        analysis_window=step_time_window,
    )
    card_stats = _build_card_stats(rank_summary)

    title = (
        f"TraceML Step Timing Summary | steps {data.training_steps} | "
        f"global ranks {len(global_ranks_seen)}"
    )
    lines = [title, "Step Time"]
    diagnosis_status = summary_diag.status
    diagnosis_why = str(
        primary_issue.summary
        or summary_diag.reason
        or "No clear timing bottleneck."
    ).strip()

    if not rank_summary:
        latest_step_text = (
            data.latest_step_observed
            if data.latest_step_observed is not None
            else "n/a"
        )
        lines.extend(
            [
                f"- Diagnosis: {diagnosis_status}",
                f"- Scope: latest step {latest_step_text}",
                "- Stats: n/a",
                f"- Why: {diagnosis_why}",
            ]
        )
    elif len(global_ranks_used) == 1:
        # The single-rank wording must not depend on the rank winning a
        # median/worst pick: a rank whose total step is unmeasured (for
        # example H2D-only) is excluded from those picks but is still the
        # run's only rank.
        only_rank = global_ranks_used[0]
        steps_analyzed = rank_summary[only_rank].steps_analyzed
        lines.extend(
            [
                f"- Diagnosis: {diagnosis_status}",
                (
                    f"- Scope: last {steps_analyzed} "
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
            f"last {step_time_window.coverage.steps_used} aligned steps "
            f"across {step_time_window.coverage.ranks_present} global ranks"
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
        for rank, s in sorted(rank_summary.items())
    }

    metadata = StepMetadata(
        mode=topology_mode_from_identities(
            (identities.get(rank) for rank in global_ranks_used),
            has_data=bool(global_ranks_used),
        ),
        global_ranks_seen=len(global_ranks_seen),
        global_ranks_used=len(global_ranks_used),
        training_total_steps=data.training_steps,
        training_latest_step=data.latest_step_observed,
        section_metric_names=STEP_TIME_METRIC_NAMES,
    )
    summary = BaseSectionPayload(
        metadata=metadata.to_json(),
        diagnosis=diagnosis_json,
        issues=issues_json,
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
