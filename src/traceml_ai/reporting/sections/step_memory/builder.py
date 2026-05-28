# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Payload builder for the final-report step-memory section.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from traceml_ai.diagnostics.common import DiagnosticResult
from traceml_ai.diagnostics.step_memory import StepMemoryDiagnosis
from traceml_ai.renderers.step_memory.schema import StepMemoryCombinedMetric
from traceml_ai.reporting.schema import (
    BaseGroups,
    BaseSectionPayload,
    GroupRow,
    StepMetadata,
)
from traceml_ai.reporting.sections.step_memory.loader import (
    StepMemorySectionData,
)
from traceml_ai.reporting.sections.step_memory.model import (
    STEP_MEMORY_METRIC_NAMES,
    StepMemoryGlobalRankIdentity,
    StepMemoryGlobalRankSummary,
    build_global_rollup,
    empty_global_rollup,
    metric_label,
    metric_sort_key,
    primary_metric,
    topology_mode,
)
from traceml_ai.reporting.summaries.issue_summary import (
    diagnostic_result_to_json,
)
from traceml_ai.reporting.summaries.summary_formatting import (
    format_ratio_percent,
)
from traceml_ai.utils.formatting import fmt_mem_new


def _identity_to_json(
    identity: StepMemoryGlobalRankIdentity,
) -> Dict[str, Any]:
    """Serialize the distributed runtime identity for one memory rank."""
    return {
        "global_rank": identity.global_rank,
        "local_rank": identity.local_rank,
        "node_rank": identity.node_rank,
        "hostname": identity.hostname,
        "local_world_size": identity.local_world_size,
        "world_size": identity.world_size,
    }


def _group_rows_to_json(
    per_global_rank: Dict[str, StepMemoryGlobalRankSummary],
) -> Dict[str, Dict[str, Any]]:
    """Build schema rows from typed per-rank memory summaries."""
    return {
        rank_key: GroupRow(
            identity=_identity_to_json(summary.identity),
            metrics=dict(summary.metrics),
        ).to_json()
        for rank_key, summary in per_global_rank.items()
    }


def _build_step_memory_payload(
    *,
    training_steps: int,
    latest_step_observed: Optional[int],
    metrics: list[StepMemoryCombinedMetric],
    diagnosis_result: DiagnosticResult[StepMemoryDiagnosis],
    per_global_rank: Dict[str, StepMemoryGlobalRankSummary],
    global_ranks_seen_fallback: int = 0,
) -> Dict[str, Any]:
    """Build the end-of-run step-memory summary payload and text card."""
    sorted_metrics = sorted(metrics, key=metric_sort_key)
    diagnosis = diagnosis_result.primary
    primary = primary_metric(sorted_metrics, diagnosis)
    diagnosis_json, issues_json = diagnostic_result_to_json(diagnosis_result)
    group_rows = _group_rows_to_json(per_global_rank)

    if not sorted_metrics or primary is None:
        latest_step_text = (
            latest_step_observed if latest_step_observed is not None else "n/a"
        )
        card = "\n".join(
            [
                "TraceML Step Memory Summary | "
                f"steps {training_steps} | "
                f"global ranks {global_ranks_seen_fallback}",
                "Step Memory",
                f"- Diagnosis: {diagnosis.status}",
                f"- Scope: latest step {latest_step_text}",
                "- Stats: n/a",
                f"- Why: {diagnosis.reason}",
            ]
        )

        metadata = StepMetadata(
            mode="no_data",
            global_ranks_seen=int(global_ranks_seen_fallback),
            global_ranks_used=0,
            training_total_steps=training_steps,
            training_latest_step=latest_step_observed,
            section_metric_names=STEP_MEMORY_METRIC_NAMES,
        )
        summary = BaseSectionPayload(
            metadata=metadata.to_json(),
            diagnosis=diagnosis_json,
            issues=issues_json,
            global_summary=empty_global_rollup(),
            groups=BaseGroups(
                by="global_rank",
                rows={},
            ).to_json(),
            units={"memory": "bytes"},
            card=card,
        ).to_json()
        return summary

    steps_used = int(primary.summary.steps_used)
    global_ranks_seen = int(primary.coverage.world_size)
    global_ranks_used = int(primary.coverage.ranks_present)
    single_rank = global_ranks_used <= 1

    lines = [
        (
            f"TraceML Step Memory Summary | steps {training_steps} | "
            f"global ranks {global_ranks_seen}"
        ),
        "Step Memory",
        f"- Diagnosis: {diagnosis.status}",
        f"- Scope: last {steps_used} aligned steps",
    ]
    if single_rank:
        stats_text = (
            f"{metric_label(primary.metric)} peak "
            f"{fmt_mem_new(primary.summary.worst_peak)}"
        )
    else:
        worst_global_rank = (
            f"r{primary.summary.worst_rank}"
            if primary.summary.worst_rank is not None
            else "rn/a"
        )
        stats_text = (
            f"{metric_label(primary.metric)} worst "
            f"{fmt_mem_new(primary.summary.worst_peak)} on "
            f"{worst_global_rank} | "
            f"skew {format_ratio_percent(primary.summary.skew_pct)}"
        )
    lines.extend([f"- Stats: {stats_text}", f"- Why: {diagnosis.reason}"])
    card = "\n".join(lines)

    aggregate = build_global_rollup(
        metrics=sorted_metrics,
        per_global_rank=per_global_rank,
    )
    metadata = StepMetadata(
        mode=topology_mode(
            global_ranks_used=global_ranks_used,
            per_global_rank=per_global_rank,
        ),
        global_ranks_seen=global_ranks_seen,
        global_ranks_used=global_ranks_used,
        training_total_steps=training_steps,
        training_latest_step=latest_step_observed,
        section_metric_names=STEP_MEMORY_METRIC_NAMES,
    )
    summary = BaseSectionPayload(
        metadata=metadata.to_json(),
        diagnosis=diagnosis_json,
        issues=issues_json,
        global_summary=aggregate,
        groups=BaseGroups(
            by="global_rank",
            rows=group_rows,
        ).to_json(),
        units={"memory": "bytes"},
        card=card,
    ).to_json()
    return summary


def build_step_memory_section_payload(
    data: StepMemorySectionData,
    diagnosis_result: DiagnosticResult[StepMemoryDiagnosis],
) -> Dict[str, Any]:
    """
    Build the JSON-safe step-memory section payload from loaded data.
    """
    return _build_step_memory_payload(
        training_steps=data.training_steps,
        latest_step_observed=data.latest_step_observed,
        metrics=data.metrics,
        diagnosis_result=diagnosis_result,
        per_global_rank=data.per_global_rank,
        global_ranks_seen_fallback=data.aligned_window.global_ranks_seen,
    )


__all__ = [
    "build_step_memory_section_payload",
]
