# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Step-memory data shaping for the final-report section."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from traceml_ai.diagnostics.step_memory import StepMemoryDiagnosis
from traceml_ai.renderers.step_memory.schema import (
    StepMemoryCombinedCoverage,
    StepMemoryCombinedMetric,
    StepMemoryCombinedSeries,
    StepMemoryCombinedSummary,
)
from traceml_ai.reporting.config import DEFAULT_SUMMARY_WINDOW_ROWS
from traceml_ai.reporting.schema import BaseGlobal, GlobalWindow
from traceml_ai.reporting.topology import topology_mode_from_identities

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


@dataclass(frozen=True)
class StepMemoryStepMetrics:
    """Memory metrics captured for one completed training step."""

    peak_allocated_bytes: float
    peak_reserved_bytes: float


@dataclass(frozen=True)
class StepMemoryRankWindow:
    """Aligned step-memory time series for one global rank."""

    identity: StepMemoryGlobalRankIdentity
    device: Optional[str]
    step_metrics: Mapping[int, StepMemoryStepMetrics]


@dataclass(frozen=True)
class StepMemoryAlignedWindow:
    """Latest common step-memory window shared by observed global ranks."""

    steps: tuple[int, ...]
    per_global_rank: Mapping[int, StepMemoryRankWindow]
    window_size: int
    global_ranks_seen: int

    @property
    def global_ranks_used(self) -> int:
        """Return global ranks included in the aligned memory window."""
        return len(self.per_global_rank)

    @property
    def completed_step(self) -> Optional[int]:
        """Return latest aligned step, if any."""
        return self.steps[-1] if self.steps else None


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


def _metric_value(
    metrics: StepMemoryStepMetrics,
    metric_name: str,
) -> float:
    """Read one internal step-memory metric by canonical metric name."""
    if metric_name == "peak_allocated":
        return float(metrics.peak_allocated_bytes)
    if metric_name == "peak_reserved":
        return float(metrics.peak_reserved_bytes)
    raise ValueError(f"Unsupported Step Memory metric: {metric_name}")


def _majority_device(devices: Iterable[Optional[str]]) -> Optional[str]:
    """Return the most common non-empty device label."""
    clean = [str(device) for device in devices if device]
    if not clean:
        return None
    return max(set(clean), key=clean.count)


def _median(values: Sequence[float]) -> float:
    """Return the median of a non-empty numeric sequence."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def build_combined_metrics_from_window(
    aligned_window: StepMemoryAlignedWindow,
    *,
    metric_names: Sequence[str] = ("peak_allocated", "peak_reserved"),
) -> list[StepMemoryCombinedMetric]:
    """Build diagnosis-ready combined metrics from one aligned window."""
    steps = list(aligned_window.steps)
    ranks = sorted(aligned_window.per_global_rank.keys())
    if not steps or not ranks:
        return []

    out: list[StepMemoryCombinedMetric] = []
    for metric_name in metric_names:
        values_by_rank: list[list[float]] = []
        devices: list[Optional[str]] = []

        for rank in ranks:
            rank_window = aligned_window.per_global_rank[rank]
            row_values = [
                _metric_value(rank_window.step_metrics[step], metric_name)
                for step in steps
            ]
            values_by_rank.append(row_values)
            devices.append(rank_window.device)

        if not values_by_rank:
            continue

        columns = list(zip(*values_by_rank))
        median_series = [
            float(_median([float(value) for value in column]))
            for column in columns
        ]
        worst_series = [
            float(max(float(value) for value in column)) for column in columns
        ]
        rank_peaks = [max(values) for values in values_by_rank]
        median_peak = float(_median(rank_peaks))
        worst_peak = float(max(rank_peaks))
        worst_idx = rank_peaks.index(worst_peak)
        worst_rank = int(ranks[worst_idx])
        skew_ratio = worst_peak / median_peak if median_peak > 0.0 else 0.0
        skew_pct = (
            (worst_peak - median_peak) / median_peak
            if median_peak > 0.0
            else 0.0
        )

        out.append(
            StepMemoryCombinedMetric(
                metric=str(metric_name),
                device=_majority_device(devices),
                series=StepMemoryCombinedSeries(
                    steps=[int(step) for step in steps],
                    median=median_series,
                    worst=worst_series,
                ),
                summary=StepMemoryCombinedSummary(
                    window_size=int(aligned_window.window_size),
                    steps_used=len(steps),
                    median_peak=median_peak,
                    worst_peak=worst_peak,
                    worst_rank=worst_rank,
                    skew_ratio=float(skew_ratio),
                    skew_pct=float(skew_pct),
                ),
                coverage=StepMemoryCombinedCoverage(
                    expected_steps=int(aligned_window.window_size),
                    steps_used=len(steps),
                    completed_step=aligned_window.completed_step,
                    world_size=int(aligned_window.global_ranks_seen),
                    ranks_present=int(aligned_window.global_ranks_used),
                    incomplete=(
                        aligned_window.global_ranks_used
                        < aligned_window.global_ranks_seen
                    ),
                ),
            )
        )

    return out


def build_global_rank_summaries_from_window(
    aligned_window: StepMemoryAlignedWindow,
) -> Dict[str, StepMemoryGlobalRankSummary]:
    """Build JSON row summaries from the same aligned memory window."""
    out: Dict[str, StepMemoryGlobalRankSummary] = {}
    for rank, rank_window in sorted(aligned_window.per_global_rank.items()):
        values = list(rank_window.step_metrics.values())
        if not values:
            continue
        out[str(rank)] = StepMemoryGlobalRankSummary(
            identity=rank_window.identity,
            metrics={
                "peak_allocated_bytes": (
                    sum(item.peak_allocated_bytes for item in values)
                    / len(values)
                ),
                "peak_reserved_bytes": (
                    sum(item.peak_reserved_bytes for item in values)
                    / len(values)
                ),
            },
        )
    return out


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
    "StepMemoryAlignedWindow",
    "StepMemoryGlobalRankIdentity",
    "StepMemoryGlobalRankSummary",
    "StepMemoryRankWindow",
    "StepMemoryStepMetrics",
    "build_combined_metrics_from_window",
    "build_global_rollup",
    "build_global_rank_summaries_from_window",
    "empty_global_rollup",
    "metric_label",
    "metric_sort_key",
    "primary_metric",
    "topology_mode",
]
