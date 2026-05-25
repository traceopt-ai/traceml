# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

import sqlite3

from traceml_ai.reporting.sections.process import ProcessSummarySection
from traceml_ai.reporting.sections.process.loader import (
    load_process_section_data,
)
from traceml_ai.reporting.sections.system import SystemSummarySection
from traceml_ai.reporting.sections.system.loader import (
    load_system_section_data,
)
from traceml_ai.reporting.summaries.process import (
    generate_process_summary_card,
)
from traceml_ai.reporting.summaries.system import generate_system_summary_card


def _create_system_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE system_samples (
            id INTEGER PRIMARY KEY,
            global_rank INTEGER,
            local_rank INTEGER,
            world_size INTEGER,
            local_world_size INTEGER,
            node_rank INTEGER,
            hostname TEXT,
            seq INTEGER,
            sample_ts_s REAL,
            cpu_percent REAL,
            ram_used_bytes REAL,
            ram_total_bytes REAL,
            gpu_available INTEGER,
            gpu_count INTEGER,
            gpu_util_avg REAL,
            gpu_util_peak REAL,
            gpu_mem_used_avg_bytes REAL,
            gpu_mem_used_peak_bytes REAL,
            gpu_temp_avg_c REAL,
            gpu_temp_peak_c REAL,
            gpu_power_avg_w REAL,
            gpu_power_peak_w REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE system_gpu_samples (
            id INTEGER PRIMARY KEY,
            global_rank INTEGER,
            local_rank INTEGER,
            world_size INTEGER,
            local_world_size INTEGER,
            node_rank INTEGER,
            hostname TEXT,
            sample_ts_s REAL,
            seq INTEGER,
            gpu_idx INTEGER,
            util REAL,
            mem_used_bytes REAL,
            mem_total_bytes REAL,
            temperature_c REAL,
            power_usage_w REAL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO system_samples VALUES (
            1, 0, 0, 1, 1, 0, 'worker-0', 1, 10.0,
            40.0, 8.0, 16.0, 1, 1,
            55.0, 70.0, 4.0, 5.0, 60.0, 68.0, 100.0, 120.0
        )
        """
    )
    conn.execute(
        """
        INSERT INTO system_gpu_samples VALUES (
            1, 0, 0, 1, 1, 0, 'worker-0', 10.0, 1, 0,
            70.0, 5.0, 10.0, 68.0, 120.0
        )
        """
    )


def _create_process_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE process_samples (
            id INTEGER PRIMARY KEY,
            rank INTEGER,
            global_rank INTEGER,
            local_rank INTEGER,
            world_size INTEGER,
            local_world_size INTEGER,
            node_rank INTEGER,
            hostname TEXT,
            sample_ts_s REAL,
            cpu_percent REAL,
            cpu_logical_core_count INTEGER,
            ram_used_bytes REAL,
            ram_total_bytes REAL,
            gpu_available INTEGER,
            gpu_count INTEGER,
            gpu_device_index INTEGER,
            gpu_mem_used_bytes REAL,
            gpu_mem_reserved_bytes REAL,
            gpu_mem_total_bytes REAL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO process_samples VALUES (
            1, 0, 0, 0, 1, 1, 0, 'worker-0',
            10.0, 80.0, 8, 4000000000.0, 16000000000.0,
            1, 1, 0, 5000000000.0, 6000000000.0, 10000000000.0
        )
        """
    )
    conn.execute(
        """
        INSERT INTO process_samples VALUES (
            2, 0, 0, 0, 1, 1, 0, 'worker-0',
            11.0, 40.0, 8, 2000000000.0, 16000000000.0,
            1, 1, 0, 3000000000.0, 4000000000.0, 10000000000.0
        )
        """
    )


def test_system_section_loader_and_builder_use_sqlite_fixture(tmp_path):
    db_path = tmp_path / "system.db"
    conn = sqlite3.connect(db_path)
    try:
        _create_system_tables(conn)
        conn.commit()
    finally:
        conn.close()

    data = load_system_section_data(str(db_path))
    result = SystemSummarySection().build(str(db_path))

    assert data.cluster.aggregate.system_samples == 1
    assert data.cluster.nodes["0"].per_gpu[0].util_peak_percent == 70.0
    assert result.section == "system"
    assert result.payload["metadata"]["samples"] == 1
    assert "TraceML System Summary" in result.text
    assert "- Diagnosis: NORMAL" in result.text
    assert (
        "- Stats: CPU 40% | RAM 50% | GPU util 55% | "
        "GPU memory 50% | GPU temp 68.0C"
    ) in result.text
    assert "GPU:" not in result.text
    assert result.payload["metadata"]["nodes_coverage"] == "1/1"
    assert result.payload["metadata"]["mode"] == "single_node"
    assert result.payload["global"]["average"]["cpu_percent"] == 40.0
    assert result.payload["global"]["average"]["ram_percent"] == 50.0
    assert result.payload["global"]["average"]["gpu_util_percent"] == 55.0
    assert (
        result.payload["groups"]["rows"]["0"]["metrics"]["gpu_mem_percent"]
        == 50.0
    )
    assert result.payload["groups"]["rows"]["0"]["identity"] == {
        "node_rank": 0,
        "hostname": "worker-0",
        "global_rank": 0,
        "local_rank": 0,
        "local_world_size": 1,
        "world_size": 1,
    }
    assert "- Issues:" not in result.text


def test_system_section_reports_scoped_multinode_primary_issue(tmp_path):
    db_path = tmp_path / "system_multinode.db"
    conn = sqlite3.connect(db_path)
    try:
        _create_system_tables(conn)
        conn.execute(
            """
            INSERT INTO system_samples VALUES (
                2, 1, 0, 2, 1, 1, 'worker-1', 2, 11.0,
                41.0, 8.0, 16.0, 1, 1,
                45.0, 80.0, 4.0, 5.0, 90.0, 95.0, 100.0, 120.0
            )
            """
        )
        conn.execute(
            """
            INSERT INTO system_gpu_samples VALUES (
                2, 1, 0, 2, 1, 1, 'worker-1', 11.0, 2, 0,
                45.0, 5.0, 10.0, 95.0, 120.0
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    payload = SystemSummarySection().build(str(db_path)).payload

    assert payload["metadata"]["nodes_coverage"] == "2/2"
    assert payload["diagnosis"]["kind"] == "HIGH_GPU_TEMPERATURE"
    assert payload["diagnosis"]["scope"] == {
        "level": "gpu",
        "node": "1",
        "node_rank": 1,
        "gpu_idx": 0,
    }
    assert "1 gpu0" in payload["diagnosis"]["reason"]
    assert payload["issues"][0]["scope"]["node"] == "1"
    assert "diagnosis" not in payload["groups"]["rows"]["1"]
    assert "issues" not in payload["groups"]["rows"]["1"]
    assert payload["global"]["worst"]["gpu_temp_c"]["idx"] == "1"


def test_system_loader_uses_latest_bounded_window(tmp_path):
    db_path = tmp_path / "system_latest.db"
    conn = sqlite3.connect(db_path)
    try:
        _create_system_tables(conn)
        conn.execute(
            """
            INSERT INTO system_samples VALUES (
                2, 0, 0, 1, 1, 0, 'worker-0', 2, 20.0,
                90.0, 12.0, 16.0, 1, 1,
                80.0, 85.0, 7.0, 8.0, 62.0, 70.0, 110.0, 130.0
            )
            """
        )
        conn.execute(
            """
            INSERT INTO system_gpu_samples VALUES (
                2, 0, 0, 1, 1, 0, 'worker-0', 20.0, 2, 0,
                80.0, 8.0, 10.0, 70.0, 130.0
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    data = load_system_section_data(str(db_path), max_system_rows=1)

    assert data.cluster.aggregate.system_samples == 1
    assert data.cluster.aggregate.cpu_avg_percent == 90.0
    assert data.cluster.aggregate.ram_avg_bytes == 12.0
    assert data.cluster.nodes["0"].per_gpu[0].util_peak_percent == 80.0


def test_system_loader_uses_latest_bounded_window_per_node(tmp_path):
    db_path = tmp_path / "system_latest_per_node.db"
    conn = sqlite3.connect(db_path)
    try:
        _create_system_tables(conn)
        conn.execute("DELETE FROM system_gpu_samples")
        conn.execute("DELETE FROM system_samples")

        row_id = 1
        for node_rank in (0, 1):
            for seq in (1, 2):
                conn.execute(
                    """
                    INSERT INTO system_samples VALUES (
                        ?, ?, 0, 2, 1, ?, ?, ?, ?,
                        ?, 8.0, 16.0, 1, 1,
                        ?, ?, 4.0, 5.0, 60.0, 68.0, 100.0, 120.0
                    )
                    """,
                    (
                        row_id,
                        node_rank,
                        node_rank,
                        f"worker-{node_rank}",
                        seq,
                        10.0 + seq,
                        10.0 * row_id,
                        50.0 + row_id,
                        60.0 + row_id,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO system_gpu_samples VALUES (
                        ?, ?, 0, 2, 1, ?, ?, ?, ?, 0,
                        ?, 5.0, 10.0, 68.0, 120.0
                    )
                    """,
                    (
                        row_id,
                        node_rank,
                        node_rank,
                        f"worker-{node_rank}",
                        10.0 + seq,
                        seq,
                        60.0 + row_id,
                    ),
                )
                row_id += 1
        conn.commit()
    finally:
        conn.close()

    data = load_system_section_data(str(db_path), max_system_rows=1)

    assert data.cluster.aggregate.system_samples == 2
    assert set(data.cluster.nodes) == {"0", "1"}
    assert data.cluster.nodes["0"].aggregate.cpu_avg_percent == 20.0
    assert data.cluster.nodes["1"].aggregate.cpu_avg_percent == 40.0
    assert data.cluster.nodes["0"].per_gpu[0].util_peak_percent == 62.0
    assert data.cluster.nodes["1"].per_gpu[0].util_peak_percent == 64.0


def test_process_section_loader_and_builder_use_sqlite_fixture(tmp_path):
    db_path = tmp_path / "process.db"
    conn = sqlite3.connect(db_path)
    try:
        _create_process_tables(conn)
        conn.commit()
    finally:
        conn.close()

    data = load_process_section_data(str(db_path))
    result = ProcessSummarySection().build(str(db_path))

    assert data.aggregate.process_samples == 2
    assert result.section == "process"
    assert result.payload["metadata"]["samples"] == 2
    assert "TraceML Process Summary" in result.text
    assert "- Diagnosis: NORMAL" in result.text
    assert (
        "- Stats: global ranks 1 | CPU avg 60% | "
        "RSS peak 4.0 / 16.0 GB | GPU reserved peak 60%"
    ) in result.text
    assert "- Takeaway:" not in result.text
    assert "- Issues:" not in result.text
    assert result.payload["global"]["average"]["cpu_capacity_percent"] == 7.5
    assert result.payload["global"]["average"]["ram_percent"] == 18.75
    assert (
        result.payload["global"]["average"]["gpu_mem_reserved_percent"] == 50.0
    )
    assert (
        result.payload["global"]["average"]["gpu_mem_headroom_bytes"]
        == 5000000000.0
    )
    assert (
        "gpu_mem_reserved_overhang_ratio"
        not in result.payload["global"]["average"]
    )
    assert result.payload["metadata"]["mode"] == "single_node"
    assert result.payload["metadata"]["global_ranks_used"] == 1
    assert result.payload["global"]["window"]["samples"] == 2
    assert (
        result.payload["global"]["worst"]["gpu_mem_reserved_bytes"]["idx"]
        == "0"
    )
    assert "global_rank_rollup" not in result.payload
    assert result.payload["groups"]["rows"]["0"]["identity"] == {
        "global_rank": 0,
        "local_rank": 0,
        "node_rank": 0,
        "hostname": "worker-0",
        "local_world_size": 1,
        "world_size": 1,
    }
    assert "takeaway" not in result.payload["global"]


def test_process_loader_uses_latest_bounded_window(tmp_path):
    db_path = tmp_path / "process_latest.db"
    conn = sqlite3.connect(db_path)
    try:
        _create_process_tables(conn)
        conn.commit()
    finally:
        conn.close()

    data = load_process_section_data(str(db_path), max_process_rows=1)

    assert data.aggregate.process_samples == 1
    assert data.aggregate.cpu_avg_percent == 40.0
    assert data.aggregate.ram_avg_bytes == 2000000000.0
    assert data.per_global_rank[0].cpu_avg_percent == 40.0
    assert data.per_global_rank[0].gpu_mem_reserved_avg_bytes == 4000000000.0


def test_process_loader_ignores_rows_without_global_rank(tmp_path):
    db_path = tmp_path / "process_legacy_rank.db"
    conn = sqlite3.connect(db_path)
    try:
        _create_process_tables(conn)
        conn.execute(
            """
            INSERT INTO process_samples VALUES (
                3, 9, NULL, 0, 1, 1, 0, 'legacy-worker',
                12.0, 99.0, 8, 9000000000.0, 16000000000.0,
                1, 1, 0, 9000000000.0, 9000000000.0, 10000000000.0
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    data = load_process_section_data(str(db_path))

    assert data.aggregate.process_samples == 2
    assert data.aggregate.distinct_global_ranks == 1
    assert set(data.per_global_rank) == {0}


def test_summary_wrappers_delegate_to_section_paths(tmp_path):
    db_path = tmp_path / "combined.db"
    conn = sqlite3.connect(db_path)
    try:
        _create_system_tables(conn)
        _create_process_tables(conn)
        conn.commit()
    finally:
        conn.close()

    system = generate_system_summary_card(str(db_path), print_to_stdout=False)
    process = generate_process_summary_card(
        str(db_path), print_to_stdout=False
    )

    assert system["metadata"]["samples"] == 1
    assert process["metadata"]["samples"] == 2
    assert (tmp_path / "combined.db_summary_card.json").exists()
    assert (tmp_path / "combined.db_summary_card.txt").exists()
