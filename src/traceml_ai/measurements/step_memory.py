# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Read, align, and reduce step-memory measurements."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

from traceml_ai.measurements.window import AnalysisWindow
from traceml_ai.utils.step_windows import common_suffix_steps


@dataclass(frozen=True)
class StepMemoryCombinedSeries:
    """Per-step aggregated memory time series."""

    steps: list[int]
    median: list[float]
    worst: list[float]


@dataclass(frozen=True)
class StepMemoryCombinedSummary:
    """Window-level summary over the latest complete steps."""

    window_size: int
    steps_used: int
    median_peak: float
    worst_peak: float
    worst_rank: Optional[int]
    skew_ratio: float
    skew_pct: float


@dataclass(frozen=True)
class StepMemoryCombinedCoverage:
    """Coverage information for one combined memory metric."""

    expected_steps: int
    steps_used: int
    completed_step: Optional[int]
    world_size: int
    ranks_present: int
    incomplete: bool


@dataclass(frozen=True)
class StepMemoryCombinedMetric:
    """One aligned and reduced ``peak_allocated`` or ``peak_reserved`` metric.

    Values remain in bytes. ``device`` is the best-effort majority device
    label and is ``None`` when it is unavailable.
    """

    metric: str
    device: Optional[str]
    series: StepMemoryCombinedSeries
    summary: StepMemoryCombinedSummary
    coverage: StepMemoryCombinedCoverage


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
        """Return the latest aligned step, if any."""
        return self.steps[-1] if self.steps else None


@dataclass(frozen=True)
class StepMemoryMeasurements:
    """Selected Step Memory measurements used by reporting and diagnosis."""

    training_steps: int
    latest_step_observed: Optional[int]
    metrics: list[StepMemoryCombinedMetric]
    gpu_total_bytes: Optional[float]
    no_gpu_detected: bool
    per_global_rank: Dict[str, StepMemoryGlobalRankSummary]
    aligned_window: StepMemoryAlignedWindow


@dataclass(frozen=True)
class _StepMemoryCandidateRow:
    """Latest complete memory row for one ``(global_rank, step)`` pair."""

    identity: StepMemoryGlobalRankIdentity
    device: Optional[str]
    metrics: StepMemoryStepMetrics


def metric_sort_key(metric: StepMemoryCombinedMetric) -> int:
    """Keep allocated before reserved in shared metric collections."""
    if metric.metric == "peak_allocated":
        return 0
    if metric.metric == "peak_reserved":
        return 1
    return 99


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
    """Build diagnosis-ready metrics from one aligned memory window."""
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
    """Build per-rank summaries from the same aligned memory window."""
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


def load_gpu_total_bytes(connection: sqlite3.Connection) -> Optional[float]:
    """Return best-effort device capacity from Process telemetry.

    Capacity is optional, but it lets both live and final-summary diagnosis
    evaluate pressure relative to the same device total.
    """
    try:
        row = connection.execute(
            "SELECT MAX(gpu_mem_total_bytes) FROM process_samples;"
        ).fetchone()
    except Exception:
        return None

    if not row or row[0] is None:
        return None
    try:
        value = float(row[0])
    except Exception:
        return None
    return value if value > 0.0 else None


def _load_latest_step_observed(
    connection: sqlite3.Connection,
) -> Optional[int]:
    """Return the latest observed memory step across all ranks."""
    row = connection.execute(
        "SELECT MAX(step) FROM step_memory_samples;"
    ).fetchone()
    if not row or row[0] is None:
        return None
    return int(row[0])


def _load_gpu_available(
    connection: sqlite3.Connection,
) -> Optional[bool]:
    """Best-effort check for whether the run reported GPU availability."""
    saw_signal = False
    for query in (
        "SELECT MAX(gpu_available) FROM process_samples;",
        "SELECT MAX(gpu_available) FROM system_samples;",
    ):
        try:
            row = connection.execute(query).fetchone()
        except Exception:
            continue
        if not row or row[0] is None:
            continue
        saw_signal = True
        try:
            if bool(int(row[0])):
                return True
        except Exception:
            if bool(row[0]):
                return True
    return False if saw_signal else None


def _load_global_ranks_seen(connection: sqlite3.Connection) -> int:
    """Return the number of global ranks with step-memory rows."""
    row = connection.execute("""
        SELECT COUNT(DISTINCT global_rank)
        FROM step_memory_samples
        WHERE global_rank IS NOT NULL
          AND step IS NOT NULL;
        """).fetchone()
    return int(row[0] or 0) if row else 0


def _load_candidate_rows(
    connection: sqlite3.Connection,
    *,
    analysis_window: Optional[AnalysisWindow],
) -> Dict[int, Dict[int, _StepMemoryCandidateRow]]:
    """Load complete memory rows, deduplicated by rank and step.

    Complete-field filtering deliberately happens before ``ROW_NUMBER`` so an
    incomplete later duplicate does not hide the latest complete sample.
    """
    predicates: list[str] = []
    params: list[int] = []
    if analysis_window is not None:
        if analysis_window.start_step is not None:
            predicates.append("step >= ?")
            params.append(int(analysis_window.start_step))
        elif analysis_window.start_time_unix_ns is not None:
            predicates.append("timestamp_unix_ns >= ?")
            params.append(int(analysis_window.start_time_unix_ns))

        if analysis_window.end_step is not None:
            predicates.append("step <= ?")
            params.append(int(analysis_window.end_step))
        elif analysis_window.end_time_unix_ns is not None:
            predicates.append("timestamp_unix_ns <= ?")
            params.append(int(analysis_window.end_time_unix_ns))

    step_filter = (
        "\n              AND " + " AND ".join(predicates) if predicates else ""
    )
    rows = connection.execute(
        f"""
        WITH latest_per_step AS (
            SELECT
                global_rank,
                local_rank,
                node_rank,
                hostname,
                local_world_size,
                world_size,
                step,
                device,
                peak_alloc_bytes,
                peak_reserved_bytes,
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY global_rank, step
                    ORDER BY id DESC
                ) AS duplicate_row
            FROM step_memory_samples
            WHERE global_rank IS NOT NULL
              AND step IS NOT NULL
              AND peak_alloc_bytes IS NOT NULL
              AND peak_reserved_bytes IS NOT NULL
              {step_filter}
        )
        SELECT
            global_rank,
            local_rank,
            node_rank,
            hostname,
            local_world_size,
            world_size,
            step,
            device,
            peak_alloc_bytes,
            peak_reserved_bytes
        FROM latest_per_step
        WHERE duplicate_row = 1
        ORDER BY global_rank ASC, step ASC;
        """,
        params,
    ).fetchall()

    candidates: Dict[int, Dict[int, _StepMemoryCandidateRow]] = {}
    for row in rows:
        try:
            global_rank = int(row[0])
            step = int(row[6])
            peak_allocated = float(row[8])
            peak_reserved = float(row[9])
        except Exception:
            continue

        identity = StepMemoryGlobalRankIdentity(
            global_rank=global_rank,
            local_rank=int(row[1]) if row[1] is not None else None,
            node_rank=int(row[2]) if row[2] is not None else None,
            hostname=str(row[3]) if row[3] is not None else None,
            local_world_size=int(row[4]) if row[4] is not None else None,
            world_size=int(row[5]) if row[5] is not None else None,
        )
        candidates.setdefault(global_rank, {})[step] = _StepMemoryCandidateRow(
            identity=identity,
            device=str(row[7]) if row[7] else None,
            metrics=StepMemoryStepMetrics(
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
            ),
        )

    return candidates


def _build_aligned_window(
    connection: sqlite3.Connection,
    *,
    analysis_window: Optional[AnalysisWindow],
) -> StepMemoryAlignedWindow:
    """Build the complete common step-memory history across global ranks."""
    global_ranks_seen = _load_global_ranks_seen(connection)
    candidates = _load_candidate_rows(
        connection,
        analysis_window=analysis_window,
    )
    common_steps = common_suffix_steps(candidates)
    if not common_steps:
        return StepMemoryAlignedWindow(
            steps=(),
            per_global_rank={},
            window_size=0,
            global_ranks_seen=global_ranks_seen,
        )

    per_global_rank: Dict[int, StepMemoryRankWindow] = {}
    for global_rank, step_rows in sorted(candidates.items()):
        aligned_rows = {
            int(step): step_rows[step]
            for step in common_steps
            if step in step_rows
        }
        if len(aligned_rows) != len(common_steps):
            continue

        latest_row = aligned_rows[max(aligned_rows.keys())]
        per_global_rank[int(global_rank)] = StepMemoryRankWindow(
            identity=latest_row.identity,
            device=latest_row.device,
            step_metrics={
                int(step): row.metrics
                for step, row in sorted(aligned_rows.items())
            },
        )

    return StepMemoryAlignedWindow(
        steps=tuple(int(step) for step in common_steps),
        per_global_rank=per_global_rank,
        window_size=len(common_steps),
        global_ranks_seen=global_ranks_seen,
    )


def read_step_memory_measurements(
    connection: sqlite3.Connection,
    *,
    analysis_window: Optional[AnalysisWindow] = None,
) -> StepMemoryMeasurements:
    """Read selected Step Memory measurements from an existing connection."""
    latest_step_observed = _load_latest_step_observed(connection)
    # Completed captures are one-based; the counter survives retention.
    training_steps = (
        max(0, latest_step_observed) if latest_step_observed is not None else 0
    )
    gpu_total_bytes = load_gpu_total_bytes(connection)
    gpu_available = _load_gpu_available(connection)
    aligned_window = _build_aligned_window(
        connection,
        analysis_window=analysis_window,
    )
    metrics = sorted(
        build_combined_metrics_from_window(aligned_window),
        key=metric_sort_key,
    )
    per_global_rank = build_global_rank_summaries_from_window(aligned_window)
    return StepMemoryMeasurements(
        training_steps=training_steps,
        latest_step_observed=latest_step_observed,
        metrics=metrics,
        gpu_total_bytes=gpu_total_bytes,
        no_gpu_detected=bool(gpu_available is False),
        per_global_rank=per_global_rank,
        aligned_window=aligned_window,
    )


__all__ = [
    "StepMemoryAlignedWindow",
    "StepMemoryCombinedCoverage",
    "StepMemoryCombinedMetric",
    "StepMemoryCombinedSeries",
    "StepMemoryCombinedSummary",
    "StepMemoryGlobalRankIdentity",
    "StepMemoryGlobalRankSummary",
    "StepMemoryMeasurements",
    "StepMemoryRankWindow",
    "StepMemoryStepMetrics",
    "build_combined_metrics_from_window",
    "build_global_rank_summaries_from_window",
    "load_gpu_total_bytes",
    "metric_sort_key",
    "read_step_memory_measurements",
]
