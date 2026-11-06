# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Payload builder for the final-report step-memory section.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from traceml_ai.diagnostics.common import DiagnosticResult
from traceml_ai.diagnostics.step_memory import StepMemoryDiagnosis
from traceml_ai.measurements.step_memory import (
    StepMemoryCombinedMetric,
    StepMemoryGlobalRankIdentity,
    StepMemoryGlobalRankSummary,
    StepMemoryMeasurements,
    metric_sort_key,
)
from traceml_ai.reporting.schema import (
    BaseGlobal,
    BaseGroups,
    BaseSectionPayload,
    GlobalWindow,
    GroupRow,
    StepMetadata,
)
from traceml_ai.reporting.summaries.issue_summary import (
    diagnostic_result_to_json,
)
from traceml_ai.reporting.summaries.summary_formatting import (
    format_ratio_percent,
)
from traceml_ai.reporting.topology import topology_mode_from_identities
from traceml_ai.utils.formatting import fmt_mem_new

STEP_MEMORY_METRIC_NAMES = [
    "peak_allocated_bytes",
    "peak_reserved_bytes",
]


def _metric_label(metric_name: str) -> str:
    """Return the display label for one step-memory metric."""
    if metric_name == "peak_allocated":
        return "peak allocated"
    if metric_name == "peak_reserved":
        return "peak reserved"
    return metric_name.replace("_", " ")


def _primary_metric(
    metrics: list[StepMemoryCombinedMetric],
    diagnosis: Optional[StepMemoryDiagnosis],
) -> Optional[StepMemoryCombinedMetric]:
    """Select the metric surfaced in the compact summary card."""
    if not metrics:
        return None

    by_name = {metric.metric: metric for metric in metrics}
    if diagnosis is not None and diagnosis.metric in by_name:
        return by_name[diagnosis.metric]
    if "peak_reserved" in by_name:
        return by_name["peak_reserved"]
    if "peak_allocated" in by_name:
        return by_name["peak_allocated"]
    return metrics[0]


def _empty_rank_points() -> Dict[str, Dict[str, Any]]:
    """Return null rank points for every public step-memory metric."""
    return {
        metric_name: {"value": None, "idx": None}
        for metric_name in STEP_MEMORY_METRIC_NAMES
    }


def _empty_global_rollup() -> Dict[str, Any]:
    """Return the stable empty rollup used when measurements are absent."""
    rank_points = _empty_rank_points()
    return BaseGlobal(
        index_by="global_rank",
        window=GlobalWindow(
            kind="step_window",
            alignment="common_steps",
            steps_analyzed=0,
        ).to_json(),
        average={metric: None for metric in STEP_MEMORY_METRIC_NAMES},
        median=rank_points,
        worst=rank_points,
    ).to_json()


def _row_metric_values(
    per_global_rank: Mapping[str, StepMemoryGlobalRankSummary],
) -> Dict[str, Dict[str, float]]:
    """Return finite public metric values keyed by metric and row id."""
    values: Dict[str, Dict[str, float]] = {
        metric_name: {} for metric_name in STEP_MEMORY_METRIC_NAMES
    }
    for rank_key, entry in per_global_rank.items():
        for metric_name in STEP_MEMORY_METRIC_NAMES:
            value = entry.metrics.get(metric_name)
            if value is None:
                continue
            try:
                metric_value = float(value)
            except Exception:
                continue
            if math.isfinite(metric_value):
                values[metric_name][str(rank_key)] = metric_value
    return values


def _average_memory_by_metric(
    per_global_rank: Mapping[str, StepMemoryGlobalRankSummary],
) -> Dict[str, Optional[float]]:
    """Average each public memory metric across global-rank rows."""
    values = _row_metric_values(per_global_rank)
    return {
        metric_name: (
            sum(metric_values.values()) / len(metric_values)
            if metric_values
            else None
        )
        for metric_name, metric_values in sorted(values.items())
    }


def _rank_sort_value(rank_key: str) -> int:
    """Sort numeric rank keys predictably while tolerating other strings."""
    try:
        return int(rank_key)
    except Exception:
        return 0


def _closest_rank_to_median(values: Dict[str, float]) -> Optional[str]:
    """Return the row id whose value is closest to the metric median."""
    if not values:
        return None

    ordered_values = sorted(values.values())
    mid = len(ordered_values) // 2
    if len(ordered_values) % 2:
        median_value = ordered_values[mid]
    else:
        median_value = (ordered_values[mid - 1] + ordered_values[mid]) / 2.0

    return min(
        values,
        key=lambda rank_key: (
            abs(values[rank_key] - median_value),
            values[rank_key],
            _rank_sort_value(rank_key),
        ),
    )


def _rank_points_from_rows(
    per_global_rank: Mapping[str, StepMemoryGlobalRankSummary],
    *,
    kind: str,
) -> Dict[str, Dict[str, Any]]:
    """Build public median or worst points from per-rank rows."""
    values_by_metric = _row_metric_values(per_global_rank)
    points: Dict[str, Dict[str, Any]] = {}
    for metric_name, values in sorted(values_by_metric.items()):
        if not values:
            points[metric_name] = {"value": None, "idx": None}
            continue
        if kind == "median":
            rank_key = _closest_rank_to_median(values)
        elif kind == "worst":
            rank_key = max(
                values,
                key=lambda item: (values[item], -_rank_sort_value(item)),
            )
        else:
            raise ValueError(f"Unsupported Step Memory point kind: {kind}")

        points[metric_name] = {
            "value": values.get(rank_key) if rank_key is not None else None,
            "idx": rank_key,
        }
    return points


def _build_global_rollup(
    *,
    metrics: list[StepMemoryCombinedMetric],
    per_global_rank: Mapping[str, StepMemoryGlobalRankSummary],
) -> Dict[str, Any]:
    """Build the public global memory rollup for the analyzed window."""
    if not metrics:
        return _empty_global_rollup()

    primary = metrics[0]
    return BaseGlobal(
        index_by="global_rank",
        window=GlobalWindow(
            kind="step_window",
            alignment="common_steps",
            steps_analyzed=primary.summary.steps_used,
            end_step=primary.coverage.completed_step,
            completed_step=primary.coverage.completed_step,
        ).to_json(),
        average=_average_memory_by_metric(per_global_rank),
        median=_rank_points_from_rows(per_global_rank, kind="median")
        or _empty_rank_points(),
        worst=_rank_points_from_rows(per_global_rank, kind="worst")
        or _empty_rank_points(),
    ).to_json()


def _topology_mode(
    *,
    global_ranks_used: int,
    per_global_rank: Mapping[str, StepMemoryGlobalRankSummary],
) -> str:
    """Return run topology from observed Step Memory runtime identity."""
    return topology_mode_from_identities(
        (
            {
                "global_rank": entry.identity.global_rank,
                "local_rank": entry.identity.local_rank,
                "node_rank": entry.identity.node_rank,
                "hostname": entry.identity.hostname,
                "local_world_size": entry.identity.local_world_size,
                "world_size": entry.identity.world_size,
            }
            for entry in per_global_rank.values()
        ),
        has_data=global_ranks_used > 0,
    )


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
    primary = _primary_metric(sorted_metrics, diagnosis)
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
            samples=0,
            section_metric_names=STEP_MEMORY_METRIC_NAMES,
        )
        summary = BaseSectionPayload(
            metadata=metadata.to_json(),
            diagnosis=diagnosis_json,
            issues=issues_json,
            global_summary=_empty_global_rollup(),
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
            f"{_metric_label(primary.metric)} peak "
            f"{fmt_mem_new(primary.summary.worst_peak)}"
        )
    else:
        worst_global_rank = (
            f"r{primary.summary.worst_rank}"
            if primary.summary.worst_rank is not None
            else "rn/a"
        )
        stats_text = (
            f"{_metric_label(primary.metric)} worst "
            f"{fmt_mem_new(primary.summary.worst_peak)} on "
            f"{worst_global_rank} | "
            f"skew {format_ratio_percent(primary.summary.skew_pct)}"
        )
    lines.extend([f"- Stats: {stats_text}", f"- Why: {diagnosis.reason}"])
    card = "\n".join(lines)

    aggregate = _build_global_rollup(
        metrics=sorted_metrics,
        per_global_rank=per_global_rank,
    )
    metadata = StepMetadata(
        mode=_topology_mode(
            global_ranks_used=global_ranks_used,
            per_global_rank=per_global_rank,
        ),
        global_ranks_seen=global_ranks_seen,
        global_ranks_used=global_ranks_used,
        training_total_steps=training_steps,
        training_latest_step=latest_step_observed,
        samples=steps_used * global_ranks_used,
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
    data: StepMemoryMeasurements,
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
