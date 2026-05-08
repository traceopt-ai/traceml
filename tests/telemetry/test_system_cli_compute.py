# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3
from pathlib import Path

from traceml.aggregator.sqlite_writers import system as system_projection
from traceml.renderers.system.cli_compute import SystemCLIComputer


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    system_projection.init_schema(conn)
    return conn


def _insert_system_sample(
    conn: sqlite3.Connection,
    *,
    row_id: int,
    global_rank: int,
    node_rank: int,
    seq: int,
    cpu: float,
    ram_used: float,
    ram_total: float,
    gpu_util: float,
    gpu_mem_used: float,
    gpu_mem_total: float,
    gpu_temp: float,
    world_size: int = 2,
    local_world_size: int = 1,
) -> None:
    conn.execute(
        """
        INSERT INTO system_samples(
            recv_ts_ns,
            global_rank,
            local_rank,
            world_size,
            local_world_size,
            node_rank,
            hostname,
            pid,
            sample_ts_s,
            seq,
            cpu_percent,
            ram_used_bytes,
            ram_total_bytes,
            gpu_available,
            gpu_count,
            gpu_util_avg,
            gpu_util_peak,
            gpu_mem_used_avg_bytes,
            gpu_mem_used_peak_bytes,
            gpu_temp_avg_c,
            gpu_temp_peak_c,
            gpu_power_avg_w,
            gpu_power_peak_w
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row_id,
            global_rank,
            0,
            world_size,
            local_world_size,
            node_rank,
            f"worker-{node_rank}",
            10_000 + node_rank,
            float(seq),
            seq,
            cpu,
            ram_used,
            ram_total,
            1,
            1,
            gpu_util,
            gpu_util,
            gpu_mem_used,
            gpu_mem_used,
            gpu_temp,
            gpu_temp,
            None,
            None,
        ),
    )
    conn.execute(
        """
        INSERT INTO system_gpu_samples(
            recv_ts_ns,
            global_rank,
            local_rank,
            world_size,
            local_world_size,
            node_rank,
            hostname,
            pid,
            sample_ts_s,
            seq,
            gpu_idx,
            util,
            mem_used_bytes,
            mem_total_bytes,
            temperature_c,
            power_usage_w,
            power_limit_w
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row_id,
            global_rank,
            0,
            world_size,
            local_world_size,
            node_rank,
            f"worker-{node_rank}",
            10_000 + node_rank,
            float(seq),
            seq,
            0,
            gpu_util,
            gpu_mem_used,
            gpu_mem_total,
            gpu_temp,
            None,
            None,
        ),
    )


def test_system_cli_uses_single_node_snapshot_for_one_node(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "system.db"
    conn = _connect(db_path)
    try:
        _insert_system_sample(
            conn,
            row_id=1,
            global_rank=0,
            node_rank=0,
            seq=10,
            cpu=22.0,
            ram_used=34.0,
            ram_total=100.0,
            gpu_util=82.0,
            gpu_mem_used=48.0,
            gpu_mem_total=100.0,
            gpu_temp=61.0,
            world_size=1,
            local_world_size=1,
        )
        conn.commit()
    finally:
        conn.close()

    payload = SystemCLIComputer(str(db_path)).compute()

    assert payload.get("view") is None
    assert payload["cpu"] == 22.0
    assert payload["ram_used"] == 34.0
    assert payload["gpu_util_total"] == 82.0


def test_system_cli_rolls_up_aligned_nodes(tmp_path: Path) -> None:
    db_path = tmp_path / "system.db"
    conn = _connect(db_path)
    try:
        _insert_system_sample(
            conn,
            row_id=1,
            global_rank=0,
            node_rank=0,
            seq=10,
            cpu=22.0,
            ram_used=34.0,
            ram_total=100.0,
            gpu_util=82.0,
            gpu_mem_used=48.0,
            gpu_mem_total=100.0,
            gpu_temp=61.0,
        )
        _insert_system_sample(
            conn,
            row_id=2,
            global_rank=1,
            node_rank=1,
            seq=10,
            cpu=61.0,
            ram_used=71.0,
            ram_total=100.0,
            gpu_util=45.0,
            gpu_mem_used=79.0,
            gpu_mem_total=100.0,
            gpu_temp=77.0,
        )
        conn.commit()
    finally:
        conn.close()

    payload = SystemCLIComputer(str(db_path)).compute()
    metrics = payload["metrics"]

    assert payload["view"] == "cluster"
    assert payload["title_suffix"] == "(med/worst, nodes 2/2)"
    assert metrics["cpu"] == {
        "median": 41.5,
        "worst": 61.0,
        "worst_node": "n1",
    }
    assert metrics["ram"] == {
        "median": 52.5,
        "worst": 71.0,
        "worst_node": "n1",
    }
    assert metrics["gpu_util"] == {
        "median": 63.5,
        "worst": 45.0,
        "worst_node": "n1",
    }
    assert metrics["gpu_mem"] == {
        "median": 63.5,
        "worst": 79.0,
        "worst_node": "n1",
    }
    assert metrics["gpu_temp"] == {
        "median": 69.0,
        "worst": 77.0,
        "worst_node": "n1",
    }
    assert metrics["gpu_headroom"] == {
        "median": 36.5,
        "worst": 21.0,
        "worst_node": "n1",
    }


def test_system_cli_marks_undercovered_sequence_partial(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "system.db"
    conn = _connect(db_path)
    try:
        for idx in range(3):
            _insert_system_sample(
                conn,
                row_id=idx + 1,
                global_rank=idx,
                node_rank=idx,
                seq=20,
                cpu=20.0 + idx,
                ram_used=30.0 + idx,
                ram_total=100.0,
                gpu_util=80.0 - idx,
                gpu_mem_used=40.0 + idx,
                gpu_mem_total=100.0,
                gpu_temp=60.0 + idx,
                world_size=4,
                local_world_size=1,
            )
        conn.commit()
    finally:
        conn.close()

    payload = SystemCLIComputer(str(db_path)).compute()

    assert payload["view"] == "cluster"
    assert payload["partial"] is True
    assert payload["title_suffix"] == "(med/worst, partial view, nodes 3/4)"
