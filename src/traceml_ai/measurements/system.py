# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Read and reduce system measurements."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from traceml_ai.measurements.window import AnalysisWindow


@dataclass
class SystemSummaryAgg:
    """System measurements aggregated across the selected samples.

    CPU and RAM values come from `system_samples`; GPU rollups come from
    `system_gpu_samples`. GPU fields remain optional for CPU-only runs.
    """

    # Sample timestamps are Unix seconds from the telemetry source.
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
    """Measurements aggregated for one physical GPU."""

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
    """Aggregated System measurements for one node."""

    identity: SystemNodeIdentity
    aggregate: SystemSummaryAgg
    per_gpu: Dict[int, PerGPUSummary] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemClusterSummary:
    """Cluster-shaped System measurements for single- and multi-node runs."""

    aggregate: SystemSummaryAgg
    nodes: Dict[str, SystemNodeSummary]
    expected_nodes: int

    @property
    def observed_nodes(self) -> int:
        """Number of nodes represented in the selected samples."""
        return len(self.nodes)

    @property
    def partial(self) -> bool:
        """Whether fewer nodes were observed than expected."""
        return self.observed_nodes < self.expected_nodes

    @property
    def observed_gpus(self) -> int:
        """Number of per-node GPUs represented in detailed GPU rows."""
        return sum(len(node.per_gpu) for node in self.nodes.values())


@dataclass(frozen=True)
class SystemMeasurements:
    """Selected System measurements used by reporting and diagnosis."""

    cluster: SystemClusterSummary


def percent(
    numerator: Optional[float],
    denominator: Optional[float],
) -> Optional[float]:
    """Return ``numerator / denominator`` as a non-negative percentage."""
    if numerator is None or denominator is None or float(denominator) <= 0.0:
        return None
    return max(0.0, float(numerator) / float(denominator) * 100.0)


def node_gpu_mem_peak_percent(node: SystemNodeSummary) -> Optional[float]:
    """Return the highest GPU memory pressure observed on one node."""
    values = [
        percent(gpu.mem_peak_bytes, gpu.mem_total_bytes)
        for gpu in node.per_gpu.values()
    ]
    nums = [value for value in values if value is not None]
    return max(nums) if nums else None


def node_gpu_headroom_min_gb(node: SystemNodeSummary) -> Optional[float]:
    """Return the smallest GPU memory headroom on one node, in GiB."""
    values = []
    for gpu in node.per_gpu.values():
        if gpu.mem_total_bytes is None or gpu.mem_peak_bytes is None:
            continue
        values.append(
            max(0.0, float(gpu.mem_total_bytes) - float(gpu.mem_peak_bytes))
            / 1024**3
        )
    return min(values) if values else None


def _average_optional(
    values: Iterable[Optional[float]],
) -> Optional[float]:
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else None


def _max_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(value) for value in values if value is not None]
    return max(nums) if nums else None


def _max_int_optional(values: Iterable[Optional[int]]) -> Optional[int]:
    nums = [int(value) for value in values if value is not None]
    return max(nums) if nums else None


def _table_has_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
) -> bool:
    rows = connection.execute(f"PRAGMA table_info({table_name});").fetchall()
    return any(str(row[1]) == column_name for row in rows)


@dataclass(frozen=True)
class _SampleRow:
    global_rank: Optional[int]
    local_rank: Optional[int]
    world_size: Optional[int]
    local_world_size: Optional[int]
    node_rank: Optional[int]
    hostname: Optional[str]
    timestamp_unix_ns: Optional[int]
    seq: Optional[int]
    cpu_percent: Optional[float]
    ram_used_bytes: Optional[float]
    ram_total_bytes: Optional[float]
    gpu_available: Optional[bool]
    gpu_count: Optional[int]


@dataclass(frozen=True)
class _SampleGpuRollup:
    """Per-system-sample GPU rollup derived from raw GPU rows."""

    gpu_util_avg: Optional[float]
    gpu_util_peak: Optional[float]
    gpu_mem_used_avg_bytes: Optional[float]
    gpu_mem_used_peak_bytes: Optional[float]
    gpu_temp_avg_c: Optional[float]
    gpu_temp_peak_c: Optional[float]
    gpu_power_avg_w: Optional[float]
    gpu_power_peak_w: Optional[float]


@dataclass(frozen=True)
class _GpuRow:
    global_rank: Optional[int]
    node_rank: Optional[int]
    seq: Optional[int]
    gpu_idx: int
    util: Optional[float]
    mem_used_bytes: Optional[float]
    mem_total_bytes: Optional[float]
    temperature_c: Optional[float]
    power_usage_w: Optional[float]
    power_limit_w: Optional[float]


_SampleKey = tuple[Optional[int], Optional[int], Optional[int]]


def _sample_key(row: _SampleRow | _GpuRow) -> _SampleKey:
    return (row.global_rank, row.node_rank, row.seq)


def _node_label(row: _SampleRow) -> str:
    if row.node_rank is not None:
        return str(int(row.node_rank))
    return str(int(row.global_rank or 0))


def _node_identity(rows: list[_SampleRow]) -> SystemNodeIdentity:
    row = rows[-1]
    return SystemNodeIdentity(
        label=_node_label(row),
        node_rank=row.node_rank,
        hostname=row.hostname,
        global_rank=row.global_rank,
        local_rank=row.local_rank,
        local_world_size=row.local_world_size,
        world_size=row.world_size,
    )


def _aggregate_sample_gpu_rows(rows: list[_GpuRow]) -> _SampleGpuRollup:
    return _SampleGpuRollup(
        gpu_util_avg=_average_optional(row.util for row in rows),
        gpu_util_peak=_max_optional(row.util for row in rows),
        gpu_mem_used_avg_bytes=_average_optional(
            row.mem_used_bytes for row in rows
        ),
        gpu_mem_used_peak_bytes=_max_optional(
            row.mem_used_bytes for row in rows
        ),
        gpu_temp_avg_c=_average_optional(row.temperature_c for row in rows),
        gpu_temp_peak_c=_max_optional(row.temperature_c for row in rows),
        gpu_power_avg_w=_average_optional(row.power_usage_w for row in rows),
        gpu_power_peak_w=_max_optional(row.power_usage_w for row in rows),
    )


def _gpu_rollups_by_sample(
    rows: list[_GpuRow],
) -> Dict[_SampleKey, _SampleGpuRollup]:
    grouped: Dict[_SampleKey, list[_GpuRow]] = {}
    for row in rows:
        grouped.setdefault(_sample_key(row), []).append(row)
    return {
        key: _aggregate_sample_gpu_rows(gpu_rows)
        for key, gpu_rows in grouped.items()
    }


def _aggregate_samples(
    rows: list[_SampleRow],
    gpu_rollups_by_key: Dict[_SampleKey, _SampleGpuRollup],
) -> SystemSummaryAgg:
    gpu_rollups = [
        gpu_rollups_by_key[_sample_key(row)]
        for row in rows
        if _sample_key(row) in gpu_rollups_by_key
    ]
    timestamps = [
        row.timestamp_unix_ns
        for row in rows
        if row.timestamp_unix_ns is not None
    ]
    return SystemSummaryAgg(
        first_ts=min(timestamps) / 1_000_000_000 if timestamps else None,
        last_ts=max(timestamps) / 1_000_000_000 if timestamps else None,
        system_samples=len(rows),
        cpu_avg_percent=_average_optional(row.cpu_percent for row in rows),
        cpu_peak_percent=_max_optional(row.cpu_percent for row in rows),
        ram_avg_bytes=_average_optional(row.ram_used_bytes for row in rows),
        ram_peak_bytes=_max_optional(row.ram_used_bytes for row in rows),
        ram_total_bytes=_max_optional(row.ram_total_bytes for row in rows),
        gpu_available=(
            any(bool(row.gpu_available) for row in rows) if rows else None
        ),
        gpu_count=_max_int_optional(row.gpu_count for row in rows),
        gpu_util_avg_percent=_average_optional(
            rollup.gpu_util_avg for rollup in gpu_rollups
        ),
        gpu_util_peak_percent=_max_optional(
            rollup.gpu_util_peak for rollup in gpu_rollups
        ),
        gpu_mem_avg_bytes=_average_optional(
            rollup.gpu_mem_used_avg_bytes for rollup in gpu_rollups
        ),
        gpu_mem_peak_bytes=_max_optional(
            rollup.gpu_mem_used_peak_bytes for rollup in gpu_rollups
        ),
        gpu_temp_avg_c=_average_optional(
            rollup.gpu_temp_avg_c for rollup in gpu_rollups
        ),
        gpu_temp_peak_c=_max_optional(
            rollup.gpu_temp_peak_c for rollup in gpu_rollups
        ),
        gpu_power_avg_w=_average_optional(
            rollup.gpu_power_avg_w for rollup in gpu_rollups
        ),
        gpu_power_peak_w=_max_optional(
            rollup.gpu_power_peak_w for rollup in gpu_rollups
        ),
    )


def _aggregate_gpu(rows: list[_GpuRow]) -> Dict[int, PerGPUSummary]:
    grouped: Dict[int, list[_GpuRow]] = {}
    for row in rows:
        grouped.setdefault(int(row.gpu_idx), []).append(row)

    out: Dict[int, PerGPUSummary] = {}
    for gpu_idx, gpu_rows in sorted(grouped.items()):
        out[gpu_idx] = PerGPUSummary(
            gpu_idx=gpu_idx,
            util_avg_percent=_average_optional(row.util for row in gpu_rows),
            util_peak_percent=_max_optional(row.util for row in gpu_rows),
            mem_avg_bytes=_average_optional(
                row.mem_used_bytes for row in gpu_rows
            ),
            mem_peak_bytes=_max_optional(
                row.mem_used_bytes for row in gpu_rows
            ),
            mem_total_bytes=_max_optional(
                row.mem_total_bytes for row in gpu_rows
            ),
            temp_avg_c=_average_optional(
                row.temperature_c for row in gpu_rows
            ),
            temp_peak_c=_max_optional(row.temperature_c for row in gpu_rows),
            power_avg_w=_average_optional(
                row.power_usage_w for row in gpu_rows
            ),
            power_peak_w=_max_optional(row.power_usage_w for row in gpu_rows),
            power_limit_w=_max_optional(row.power_limit_w for row in gpu_rows),
        )
    return out


def _expected_nodes(rows: list[_SampleRow]) -> int:
    candidates = {
        int(math.ceil(float(row.world_size) / float(row.local_world_size)))
        for row in rows
        if row.world_size and row.local_world_size
    }
    if len(candidates) == 1:
        return max(1, candidates.pop())
    return max(1, len({_node_label(row) for row in rows}))


def _system_samples_cte(
    *,
    node_rank: Optional[int],
    analysis_window: Optional[AnalysisWindow],
) -> tuple[str, tuple[float | int, ...]]:
    """
    Return the CTE used for the selected system history.
    """
    predicates = []
    params: list[float | int] = []
    if node_rank is not None:
        predicates.append("node_rank = ?")
        params.append(int(node_rank))
    if (
        analysis_window is not None
        and analysis_window.start_time_unix_ns is not None
        and analysis_window.end_time_unix_ns is not None
    ):
        predicates.append("timestamp_unix_ns BETWEEN ? AND ?")
        params.extend(
            (
                analysis_window.start_time_unix_ns,
                analysis_window.end_time_unix_ns,
            )
        )
    where_clause = "WHERE " + " AND ".join(predicates) if predicates else ""
    return f"""
        WITH selected_system_samples AS (
            SELECT s.*
            FROM system_samples AS s
            {where_clause}
        )
    """, tuple(
        params
    )


def _sample_rows(
    conn: sqlite3.Connection,
    *,
    node_rank: Optional[int],
    analysis_window: Optional[AnalysisWindow],
) -> list[_SampleRow]:
    sample_cte, params = _system_samples_cte(
        node_rank=node_rank,
        analysis_window=analysis_window,
    )
    rows = conn.execute(
        sample_cte + """
        SELECT global_rank, local_rank, world_size, local_world_size,
               node_rank, hostname, timestamp_unix_ns, seq, cpu_percent,
               ram_used_bytes, ram_total_bytes, gpu_available, gpu_count
        FROM selected_system_samples
        ORDER BY COALESCE(node_rank, global_rank, 0) ASC, id ASC;
        """,
        params,
    ).fetchall()
    return [
        _SampleRow(
            global_rank=row[0],
            local_rank=row[1],
            world_size=row[2],
            local_world_size=row[3],
            node_rank=row[4],
            hostname=row[5],
            timestamp_unix_ns=row[6],
            seq=row[7],
            cpu_percent=row[8],
            ram_used_bytes=row[9],
            ram_total_bytes=row[10],
            gpu_available=bool(row[11]) if row[11] is not None else None,
            gpu_count=row[12],
        )
        for row in rows
    ]


def _gpu_rows(
    conn: sqlite3.Connection,
    *,
    node_rank: Optional[int],
    analysis_window: Optional[AnalysisWindow],
) -> list[_GpuRow]:
    power_limit_expr = (
        "g.power_limit_w"
        if _table_has_column(conn, "system_gpu_samples", "power_limit_w")
        else "NULL"
    )
    sample_cte, params = _system_samples_cte(
        node_rank=node_rank,
        analysis_window=analysis_window,
    )
    rows = conn.execute(
        sample_cte + f"""
        SELECT g.global_rank, g.node_rank, g.seq, g.gpu_idx, g.util,
               g.mem_used_bytes, g.mem_total_bytes, g.temperature_c,
               g.power_usage_w, {power_limit_expr}
        FROM system_gpu_samples AS g
        INNER JOIN selected_system_samples AS recent
            ON g.global_rank IS recent.global_rank
           AND g.node_rank IS recent.node_rank
           AND g.seq IS recent.seq
        ORDER BY g.global_rank ASC, g.seq ASC, g.gpu_idx ASC;
        """,
        params,
    ).fetchall()
    return [
        _GpuRow(
            global_rank=row[0],
            node_rank=row[1],
            seq=row[2],
            gpu_idx=int(row[3]),
            util=row[4],
            mem_used_bytes=row[5],
            mem_total_bytes=row[6],
            temperature_c=row[7],
            power_usage_w=row[8],
            power_limit_w=row[9],
        )
        for row in rows
    ]


def _load_cluster_summary(
    conn: sqlite3.Connection,
    *,
    node_rank: Optional[int],
    analysis_window: Optional[AnalysisWindow],
) -> SystemClusterSummary:

    # Load the bounded system rows, keeping the latest window per node.
    samples = _sample_rows(
        conn,
        node_rank=node_rank,
        analysis_window=analysis_window,
    )
    # Load GPU rows that belong to the same retained system samples.
    gpus = _gpu_rows(
        conn,
        node_rank=node_rank,
        analysis_window=analysis_window,
    )

    # GPU summary fields are derived from raw GPU rows at read time so the
    # storage projection does not duplicate per-sample aggregates.
    gpu_by_key: Dict[_SampleKey, list[_GpuRow]] = {}
    for row in gpus:
        gpu_by_key.setdefault(_sample_key(row), []).append(row)
    gpu_rollups_by_key = _gpu_rollups_by_sample(gpus)

    # Group system samples by node so each node gets its own rollup.
    sample_groups: Dict[str, list[_SampleRow]] = {}
    for row in samples:
        sample_groups.setdefault(_node_label(row), []).append(row)

    # Build one node summary at a time from its system samples and GPU rows.
    nodes: Dict[str, SystemNodeSummary] = {}
    for label, node_rows in sorted(sample_groups.items()):
        # Collect GPU rows that match this node's retained system samples.
        node_gpu_rows: list[_GpuRow] = []
        for row in node_rows:
            node_gpu_rows.extend(gpu_by_key.get(_sample_key(row), []))
        # Store the final per-node identity, system rollup, and GPU rollup.
        nodes[label] = SystemNodeSummary(
            identity=_node_identity(node_rows),
            aggregate=_aggregate_samples(node_rows, gpu_rollups_by_key),
            per_gpu=_aggregate_gpu(node_gpu_rows),
        )

    # Return the cluster-level rollup plus the per-node summaries.
    return SystemClusterSummary(
        aggregate=_aggregate_samples(samples, gpu_rollups_by_key),
        nodes=nodes,
        expected_nodes=_expected_nodes(samples),
    )


def read_system_measurements(
    connection: sqlite3.Connection,
    *,
    node_rank: Optional[int] = None,
    analysis_window: Optional[AnalysisWindow] = None,
) -> SystemMeasurements:
    """Read selected System measurements from an existing connection."""
    return SystemMeasurements(
        cluster=_load_cluster_summary(
            connection,
            node_rank=node_rank,
            analysis_window=analysis_window,
        )
    )


__all__ = [
    "PerGPUSummary",
    "SystemClusterSummary",
    "SystemMeasurements",
    "SystemNodeIdentity",
    "SystemNodeSummary",
    "SystemSummaryAgg",
    "node_gpu_headroom_min_gb",
    "node_gpu_mem_peak_percent",
    "percent",
    "read_system_measurements",
]
