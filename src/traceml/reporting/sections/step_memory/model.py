# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Step-memory data shaping for the final-report section."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from traceml.diagnostics.step_memory import StepMemoryDiagnosis
from traceml.renderers.step_memory.schema import StepMemoryCombinedMetric
from traceml.reporting.config import DEFAULT_SUMMARY_WINDOW_ROWS
from traceml.reporting.schema import BaseGlobal, GlobalWindow
from traceml.reporting.topology import topology_mode_from_identities

MAX_SUMMARY_WINDOW_ROWS = DEFAULT_SUMMARY_WINDOW_ROWS

STEP_MEMORY_METRIC_NAMES = [
    "peak_allocated_bytes",
    "peak_reserved_bytes",
]


@dataclass(frozen=True)
class StepMemoryGlobalRankIdentity:
    """Runtime identity observed for one step-memory global rank."""

    global_rank: int
    local_rank: Optional[int]
    node_rank: Optional[int]
    hostname: Optional[str]
    local_world_size: Optional[int]
    world_size: Optional[int]


@dataclass(frozen=True)
class StepMemoryGlobalRankSummary:
    """Aligned step-memory metrics for one global rank."""

    identity: StepMemoryGlobalRankIdentity
    metrics: Mapping[str, float]


def metric_sort_key(metric: StepMemoryCombinedMetric) -> int:
    """Stable metric ordering: allocated first, reserved second."""
    if metric.metric == "peak_allocated":
        return 0
    if metric.metric == "peak_reserved":
        return 1
    return 99


def metric_label(metric_name: str) -> str:
    """Human-friendly label for one step-memory metric key."""
    if metric_name == "peak_allocated":
        return "peak allocated"
    if metric_name == "peak_reserved":
        return "peak reserved"
    return metric_name.replace("_", " ")


def metric_output_name(metric_name: str) -> str:
    """Return the public metric name used in summary JSON."""
    return f"{metric_name}_bytes"


def no_gpu_diagnosis_json() -> Dict[str, Any]:
    """
    Stable summary diagnosis block for CPU-only / no-GPU runs.
    """
    return {
        "kind": "NO_GPU",
        "status": "NO GPU",
        "severity": "info",
        "metric": None,
        "steps_used": 0,
        "worst_global_rank": None,
        "reason": (
            "No GPU detected. Step memory uses torch-based GPU memory telemetry."
        ),
        "action": "Treat step memory as not applicable for this run.",
        "note": None,
        "confidence": 1.0,
    }


def no_gpu_diagnosis_presented() -> Dict[str, Any]:
    """
    Stable end-of-run presentation block for CPU-only / no-GPU runs.
    """
    return {
        "status": "NO GPU",
        "reason": (
            "No GPU detected. Step memory uses torch-based GPU memory telemetry."
        ),
        "action": "Step memory is not applicable for this run.",
        "note": None,
    }


def primary_metric(
    metrics: list[StepMemoryCombinedMetric],
    diagnosis: Optional[StepMemoryDiagnosis],
) -> Optional[StepMemoryCombinedMetric]:
    """
    Pick the primary metric to surface in the printed summary.

    Preference order:
    1. diagnosis.metric, if available
    2. peak_reserved
    3. peak_allocated
    4. first metric
    """
    if not metrics:
        return None

    by_name = {m.metric: m for m in metrics}

    if diagnosis is not None and diagnosis.metric in by_name:
        return by_name[diagnosis.metric]
    if "peak_reserved" in by_name:
        return by_name["peak_reserved"]
    if "peak_allocated" in by_name:
        return by_name["peak_allocated"]
    return metrics[0]


def empty_global_rollup() -> Dict[str, Any]:
    """
    Return a stable empty global rollup for missing-data cases.
    """
    average = {metric: None for metric in STEP_MEMORY_METRIC_NAMES}
    rank_points = _empty_rank_points()
    return BaseGlobal(
        index_by="global_rank",
        window=GlobalWindow(
            kind="step_window",
            alignment="common_steps",
            steps_analyzed=0,
        ).to_json(),
        average=average,
        median=rank_points,
        worst=rank_points,
    ).to_json()


def _row_metric_values(
    per_global_rank: Mapping[str, StepMemoryGlobalRankSummary],
) -> Dict[str, Dict[str, float]]:
    """Return finite metric values keyed by metric name and row id."""
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
) -> Dict:
    """Average public memory metrics across per-rank memory rows."""
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
    """Sort numeric rank keys predictably while tolerating unknown strings."""
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
    """Build `{value, idx}` points from grouped step-memory rows."""
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


def _empty_rank_points() -> Dict[str, Dict[str, Any]]:
    """Return null rank points for every public step-memory metric."""
    return {
        metric_name: {"value": None, "idx": None}
        for metric_name in STEP_MEMORY_METRIC_NAMES
    }


def build_global_rollup(
    *,
    metrics: list[StepMemoryCombinedMetric],
    diagnosis: Optional[StepMemoryDiagnosis],
    per_global_rank: Mapping[str, StepMemoryGlobalRankSummary],
) -> Dict[str, Any]:
    """Build the global memory rollup for the analyzed window."""
    if not metrics:
        return empty_global_rollup()

    primary = metrics[0]
    median = _rank_points_from_rows(per_global_rank, kind="median")
    worst = _rank_points_from_rows(per_global_rank, kind="worst")

    return BaseGlobal(
        index_by="global_rank",
        window=GlobalWindow(
            kind="step_window",
            alignment="common_steps",
            steps_analyzed=primary.summary.steps_used,
            end_step=primary.coverage.completed_step,
            completed_step=primary.coverage.completed_step,
            window_size=primary.summary.window_size,
        ).to_json(),
        average=_average_memory_by_metric(per_global_rank),
        median=median or _empty_rank_points(),
        worst=worst or _empty_rank_points(),
    ).to_json()


def topology_mode(
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


__all__ = [
    "MAX_SUMMARY_WINDOW_ROWS",
    "STEP_MEMORY_METRIC_NAMES",
    "StepMemoryGlobalRankIdentity",
    "StepMemoryGlobalRankSummary",
    "build_global_rollup",
    "empty_global_rollup",
    "metric_label",
    "metric_output_name",
    "metric_sort_key",
    "no_gpu_diagnosis_json",
    "no_gpu_diagnosis_presented",
    "primary_metric",
    "topology_mode",
]
