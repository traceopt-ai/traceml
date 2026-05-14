# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""System aggregation helpers for the final-report section."""

import sqlite3
from dataclasses import dataclass, field
from statistics import median
from typing import Dict, Iterable, Optional

MAX_SUMMARY_ROWS = 10_000

SYSTEM_METRIC_NAMES = [
    "cpu_percent",
    "ram_bytes",
    "ram_percent",
    "gpu_util_percent",
    "gpu_mem_bytes",
    "gpu_mem_percent",
    "gpu_temp_c",
    "gpu_power_w",
    "gpu_headroom_bytes",
]


def table_has_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
) -> bool:
    """Return whether a SQLite table exposes a column."""
    rows = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
    return any(str(row[1]) == column_name for row in rows)


def percent(
    numerator: Optional[float],
    denominator: Optional[float],
) -> Optional[float]:
    """Return `numerator / denominator` as a non-negative percentage."""
    if numerator is None or denominator is None or float(denominator) <= 0.0:
        return None
    return max(0.0, float(numerator) / float(denominator) * 100.0)


@dataclass
class SystemSummaryAgg:
    """
    Aggregated system metrics loaded from `system_samples`.

    Notes
    -----
    - Memory values remain in raw bytes while aggregating and are converted
      only at formatting / serialization time.
    - GPU fields are optional because CPU-only runs are fully supported.
    """

    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    system_samples: int = 0

    cpu_avg_percent: Optional[float] = None
    cpu_peak_percent: Optional[float] = None

    ram_avg_bytes: Optional[float] = None
    ram_peak_bytes: Optional[float] = None
    ram_total_bytes: Optional[float] = None

    gpu_available: Optional[bool] = None
    gpu_count: Optional[int] = None

    gpu_util_avg_percent: Optional[float] = None
    gpu_util_peak_percent: Optional[float] = None

    gpu_mem_avg_bytes: Optional[float] = None
    gpu_mem_peak_bytes: Optional[float] = None

    gpu_temp_avg_c: Optional[float] = None
    gpu_temp_peak_c: Optional[float] = None

    gpu_power_avg_w: Optional[float] = None
    gpu_power_peak_w: Optional[float] = None


@dataclass
class PerGPUSummary:
    """
    Aggregated metrics for one physical GPU across the sampled summary window.

    Notes
    -----
    - Values are aggregated across all selected `system_gpu_samples` rows for
      the device index.
    - Memory values remain in raw bytes while aggregating and are converted
      only during final summary serialization.
    """

    gpu_idx: int

    util_avg_percent: Optional[float] = None
    util_peak_percent: Optional[float] = None

    mem_avg_bytes: Optional[float] = None
    mem_peak_bytes: Optional[float] = None
    mem_total_bytes: Optional[float] = None

    temp_avg_c: Optional[float] = None
    temp_peak_c: Optional[float] = None

    power_avg_w: Optional[float] = None
    power_peak_w: Optional[float] = None
    power_limit_w: Optional[float] = None


@dataclass(frozen=True)
class SystemNodeIdentity:
    """Stable identity for one node-level SystemSampler source."""

    label: str
    node_rank: Optional[int]
    hostname: Optional[str]
    global_rank: Optional[int]
    local_rank: Optional[int]
    local_world_size: Optional[int]
    world_size: Optional[int]


@dataclass(frozen=True)
class SystemNodeSummary:
    """Aggregated System metrics for one node."""

    identity: SystemNodeIdentity
    aggregate: SystemSummaryAgg
    per_gpu: Dict[int, PerGPUSummary] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricRollup:
    """Median/worst view for one node-level metric."""

    median: Optional[float]
    worst: Optional[float]
    worst_node: Optional[str]


@dataclass(frozen=True)
class SystemClusterSummary:
    """Cluster-shaped System summary for single-node and multi-node runs."""

    aggregate: SystemSummaryAgg
    nodes: Dict[str, SystemNodeSummary]
    expected_nodes: int

    @property
    def observed_nodes(self) -> int:
        """Number of nodes represented in the summary window."""
        return len(self.nodes)

    @property
    def partial(self) -> bool:
        """Whether fewer nodes were observed than expected."""
        return self.observed_nodes < self.expected_nodes

    @property
    def observed_gpus(self) -> int:
        """Number of per-node GPUs represented in detailed GPU rows."""
        return sum(len(node.per_gpu) for node in self.nodes.values())


def rollup_metric(
    nodes: Dict[str, SystemNodeSummary],
    *,
    value_fn,
    higher_is_worse: bool,
) -> MetricRollup:
    """Build median/worst values for one metric across nodes."""
    values: list[tuple[str, float]] = []
    for label, node in nodes.items():
        value = value_fn(node)
        if value is not None:
            values.append((label, float(value)))
    if not values:
        return MetricRollup(median=None, worst=None, worst_node=None)
    worst_label, worst_value = sorted(
        values,
        key=lambda item: (
            item[1] if higher_is_worse else -item[1],
            item[0],
        ),
        reverse=True,
    )[0]
    return MetricRollup(
        median=float(median([item[1] for item in values])),
        worst=float(worst_value),
        worst_node=worst_label,
    )


def node_gpu_mem_peak_percent(node: SystemNodeSummary) -> Optional[float]:
    """Highest GPU memory pressure observed on one node."""
    values = [
        percent(gpu.mem_peak_bytes, gpu.mem_total_bytes)
        for gpu in node.per_gpu.values()
    ]
    nums = [value for value in values if value is not None]
    return max(nums) if nums else None


def node_gpu_headroom_min_gb(node: SystemNodeSummary) -> Optional[float]:
    """Smallest GPU memory headroom observed on one node, in GB."""
    values = []
    for gpu in node.per_gpu.values():
        if gpu.mem_total_bytes is None or gpu.mem_peak_bytes is None:
            continue
        values.append(
            max(0.0, float(gpu.mem_total_bytes) - float(gpu.mem_peak_bytes))
            / 1_000_000_000.0
        )
    return min(values) if values else None


def average_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    """Average non-null numeric values, returning None for empty input."""
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else None


def max_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    """Return the maximum non-null numeric value, if any."""
    nums = [float(value) for value in values if value is not None]
    return max(nums) if nums else None


def max_int_optional(values: Iterable[Optional[int]]) -> Optional[int]:
    """Return the maximum non-null integer value, if any."""
    nums = [int(value) for value in values if value is not None]
    return max(nums) if nums else None


def min_timestamp(values: Iterable[Optional[float]]) -> Optional[float]:
    """Return the earliest non-null timestamp, if any."""
    nums = [float(value) for value in values if value is not None]
    return min(nums) if nums else None


def per_gpu_to_diagnosis_input(
    per_gpu: Dict[int, PerGPUSummary],
) -> Dict[int, Dict[str, Optional[float]]]:
    """
    Convert per-GPU summary objects into the bytes-based shape used by system
    diagnosis rules.

    Notes
    -----
    Summary JSON intentionally exposes memory in GB for readability, while the
    diagnosis layer keeps raw bytes so pressure calculations remain precise.
    """
    out: Dict[int, Dict[str, Optional[float]]] = {}

    for gpu_idx, item in sorted(per_gpu.items()):
        out[int(gpu_idx)] = {
            "util_avg_percent": item.util_avg_percent,
            "util_peak_percent": item.util_peak_percent,
            "mem_avg_bytes": item.mem_avg_bytes,
            "mem_peak_bytes": item.mem_peak_bytes,
            "mem_total_bytes": item.mem_total_bytes,
            "temp_avg_c": item.temp_avg_c,
            "temp_peak_c": item.temp_peak_c,
            "power_avg_w": item.power_avg_w,
            "power_peak_w": item.power_peak_w,
            "power_limit_w": item.power_limit_w,
        }

    return out
