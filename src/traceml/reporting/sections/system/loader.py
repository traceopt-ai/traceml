# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""SQLite loader for the final-report system section."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

from traceml.reporting.sections.system.model import (
    MAX_SUMMARY_ROWS,
    PerGPUSummary,
    SystemClusterSummary,
    SystemNodeIdentity,
    SystemNodeSummary,
    SystemSummaryAgg,
    _avg,
    _load_per_gpu_summary,
    _load_system_summary_agg,
    _max,
    _max_int,
    _min_ts,
    _table_has_column,
)


@dataclass(frozen=True)
class SystemSectionData:
    """Loaded inputs for the system final-report section."""

    aggregate: SystemSummaryAgg
    per_gpu: Dict[int, PerGPUSummary]
    cluster: SystemClusterSummary


@dataclass(frozen=True)
class _SampleRow:
    global_rank: Optional[int]
    local_rank: Optional[int]
    world_size: Optional[int]
    local_world_size: Optional[int]
    node_rank: Optional[int]
    hostname: Optional[str]
    pid: Optional[int]
    sample_ts_s: Optional[float]
    seq: Optional[int]
    cpu_percent: Optional[float]
    ram_used_bytes: Optional[float]
    ram_total_bytes: Optional[float]
    gpu_available: Optional[bool]
    gpu_count: Optional[int]
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
    seq: Optional[int]
    gpu_idx: int
    util: Optional[float]
    mem_used_bytes: Optional[float]
    mem_total_bytes: Optional[float]
    temperature_c: Optional[float]
    power_usage_w: Optional[float]
    power_limit_w: Optional[float]


def _node_label(row: _SampleRow) -> str:
    if row.node_rank is not None:
        return f"n{int(row.node_rank)}"
    return f"g{int(row.global_rank or 0)}"


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
        pid=row.pid,
    )


def _aggregate_samples(rows: list[_SampleRow]) -> SystemSummaryAgg:
    return SystemSummaryAgg(
        first_ts=_min_ts(row.sample_ts_s for row in rows),
        last_ts=_max(row.sample_ts_s for row in rows),
        system_samples=len(rows),
        cpu_avg_percent=_avg(row.cpu_percent for row in rows),
        cpu_peak_percent=_max(row.cpu_percent for row in rows),
        ram_avg_bytes=_avg(row.ram_used_bytes for row in rows),
        ram_peak_bytes=_max(row.ram_used_bytes for row in rows),
        ram_total_bytes=_max(row.ram_total_bytes for row in rows),
        gpu_available=(
            any(bool(row.gpu_available) for row in rows) if rows else None
        ),
        gpu_count=_max_int(row.gpu_count for row in rows),
        gpu_util_avg_percent=_avg(row.gpu_util_avg for row in rows),
        gpu_util_peak_percent=_max(row.gpu_util_peak for row in rows),
        gpu_mem_avg_bytes=_avg(row.gpu_mem_used_avg_bytes for row in rows),
        gpu_mem_peak_bytes=_max(row.gpu_mem_used_peak_bytes for row in rows),
        gpu_temp_avg_c=_avg(row.gpu_temp_avg_c for row in rows),
        gpu_temp_peak_c=_max(row.gpu_temp_peak_c for row in rows),
        gpu_power_avg_w=_avg(row.gpu_power_avg_w for row in rows),
        gpu_power_peak_w=_max(row.gpu_power_peak_w for row in rows),
    )


def _aggregate_gpu(rows: list[_GpuRow]) -> Dict[int, PerGPUSummary]:
    grouped: Dict[int, list[_GpuRow]] = {}
    for row in rows:
        grouped.setdefault(int(row.gpu_idx), []).append(row)

    out: Dict[int, PerGPUSummary] = {}
    for gpu_idx, gpu_rows in sorted(grouped.items()):
        out[gpu_idx] = PerGPUSummary(
            gpu_idx=gpu_idx,
            util_avg_percent=_avg(row.util for row in gpu_rows),
            util_peak_percent=_max(row.util for row in gpu_rows),
            mem_avg_bytes=_avg(row.mem_used_bytes for row in gpu_rows),
            mem_peak_bytes=_max(row.mem_used_bytes for row in gpu_rows),
            mem_total_bytes=_max(row.mem_total_bytes for row in gpu_rows),
            temp_avg_c=_avg(row.temperature_c for row in gpu_rows),
            temp_peak_c=_max(row.temperature_c for row in gpu_rows),
            power_avg_w=_avg(row.power_usage_w for row in gpu_rows),
            power_peak_w=_max(row.power_usage_w for row in gpu_rows),
            power_limit_w=_max(row.power_limit_w for row in gpu_rows),
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


def _sample_rows(
    conn: sqlite3.Connection,
    *,
    node_rank: Optional[int],
    max_system_rows: int,
) -> list[_SampleRow]:
    where_clause = "WHERE node_rank = ?" if node_rank is not None else ""
    params: tuple[Any, ...] = (
        (int(node_rank),) if node_rank is not None else ()
    )
    rows = conn.execute(
        f"""
        SELECT global_rank, local_rank, world_size, local_world_size,
               node_rank, hostname, pid, sample_ts_s, seq, cpu_percent,
               ram_used_bytes, ram_total_bytes, gpu_available, gpu_count,
               gpu_util_avg, gpu_util_peak, gpu_mem_used_avg_bytes,
               gpu_mem_used_peak_bytes, gpu_temp_avg_c, gpu_temp_peak_c,
               gpu_power_avg_w, gpu_power_peak_w
        FROM system_samples
        {where_clause}
        ORDER BY id ASC
        LIMIT ?;
        """,
        (*params, int(max_system_rows)),
    ).fetchall()
    return [
        _SampleRow(
            global_rank=row[0],
            local_rank=row[1],
            world_size=row[2],
            local_world_size=row[3],
            node_rank=row[4],
            hostname=row[5],
            pid=row[6],
            sample_ts_s=row[7],
            seq=row[8],
            cpu_percent=row[9],
            ram_used_bytes=row[10],
            ram_total_bytes=row[11],
            gpu_available=bool(row[12]) if row[12] is not None else None,
            gpu_count=row[13],
            gpu_util_avg=row[14],
            gpu_util_peak=row[15],
            gpu_mem_used_avg_bytes=row[16],
            gpu_mem_used_peak_bytes=row[17],
            gpu_temp_avg_c=row[18],
            gpu_temp_peak_c=row[19],
            gpu_power_avg_w=row[20],
            gpu_power_peak_w=row[21],
        )
        for row in rows
    ]


def _gpu_rows(
    conn: sqlite3.Connection,
    *,
    node_rank: Optional[int],
    max_system_rows: int,
) -> list[_GpuRow]:
    where_clause = "WHERE s.node_rank = ?" if node_rank is not None else ""
    params: tuple[Any, ...] = (
        (int(node_rank),) if node_rank is not None else ()
    )
    power_limit_expr = (
        "g.power_limit_w"
        if _table_has_column(conn, "system_gpu_samples", "power_limit_w")
        else "NULL"
    )
    rows = conn.execute(
        f"""
        SELECT g.global_rank, g.seq, g.gpu_idx, g.util, g.mem_used_bytes,
               g.mem_total_bytes, g.temperature_c, g.power_usage_w,
               {power_limit_expr}
        FROM system_gpu_samples AS g
        INNER JOIN (
            SELECT s.global_rank, s.seq
            FROM system_samples AS s
            {where_clause}
            ORDER BY s.id ASC
            LIMIT ?
        ) AS recent
            ON g.global_rank IS recent.global_rank
           AND g.seq = recent.seq
        ORDER BY g.global_rank ASC, g.seq ASC, g.gpu_idx ASC;
        """,
        (*params, int(max_system_rows)),
    ).fetchall()
    return [
        _GpuRow(
            global_rank=row[0],
            seq=row[1],
            gpu_idx=int(row[2]),
            util=row[3],
            mem_used_bytes=row[4],
            mem_total_bytes=row[5],
            temperature_c=row[6],
            power_usage_w=row[7],
            power_limit_w=row[8],
        )
        for row in rows
    ]


def _load_cluster_summary(
    conn: sqlite3.Connection,
    *,
    node_rank: Optional[int],
    max_system_rows: int,
) -> SystemClusterSummary:
    samples = _sample_rows(
        conn,
        node_rank=node_rank,
        max_system_rows=max_system_rows,
    )
    gpus = _gpu_rows(
        conn,
        node_rank=node_rank,
        max_system_rows=max_system_rows,
    )
    gpu_by_key: Dict[tuple[Optional[int], Optional[int]], list[_GpuRow]] = {}
    for row in gpus:
        gpu_by_key.setdefault((row.global_rank, row.seq), []).append(row)

    sample_groups: Dict[str, list[_SampleRow]] = {}
    for row in samples:
        sample_groups.setdefault(_node_label(row), []).append(row)

    nodes: Dict[str, SystemNodeSummary] = {}
    for label, node_rows in sorted(sample_groups.items()):
        node_gpu_rows: list[_GpuRow] = []
        for row in node_rows:
            node_gpu_rows.extend(
                gpu_by_key.get((row.global_rank, row.seq), [])
            )
        nodes[label] = SystemNodeSummary(
            identity=_node_identity(node_rows),
            aggregate=_aggregate_samples(node_rows),
            per_gpu=_aggregate_gpu(node_gpu_rows),
        )

    return SystemClusterSummary(
        aggregate=_aggregate_samples(samples),
        nodes=nodes,
        expected_nodes=_expected_nodes(samples),
    )


def load_system_section_data(
    db_path: str,
    *,
    node_rank: Optional[int] = None,
    max_system_rows: int = MAX_SUMMARY_ROWS,
) -> SystemSectionData:
    """
    Load bounded system-section data from the SQLite history database.
    """
    row_limit = min(max(1, int(max_system_rows)), MAX_SUMMARY_ROWS)
    conn = sqlite3.connect(db_path)
    try:
        cluster = _load_cluster_summary(
            conn,
            node_rank=node_rank,
            max_system_rows=row_limit,
        )
        aggregate = _load_system_summary_agg(
            conn,
            node_rank=node_rank,
            max_system_rows=row_limit,
        )
        per_gpu = _load_per_gpu_summary(
            conn,
            node_rank=node_rank,
            max_system_rows=row_limit,
        )
    finally:
        conn.close()

    return SystemSectionData(
        aggregate=aggregate,
        per_gpu=per_gpu,
        cluster=cluster,
    )


__all__ = [
    "SystemSectionData",
    "load_system_section_data",
]
