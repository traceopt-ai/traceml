# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from tests.sqlite_fixtures import (
    insert_process_sample,
    insert_system_sample,
    summary_database,
)
from traceml_ai.reporting.sections.process import ProcessSummarySection
from traceml_ai.reporting.analysis_window import AnalysisWindow
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


def _seed_system_samples(conn) -> None:
    insert_system_sample(
        conn,
        row_id=1,
        rank=0,
        ts=10.0,
        gpu_available=True,
        gpu_count=2,
        cpu_percent=40.0,
        ram_used_bytes=8.0,
        ram_total_bytes=16.0,
        gpu_samples=(
            {
                "gpu_idx": 0,
                "util": 70.0,
                "mem_used_bytes": 5.0,
                "mem_total_bytes": 10.0,
                "temperature_c": 68.0,
                "power_usage_w": 120.0,
            },
            {
                "recv_ts_ns": 10,
                "gpu_idx": 1,
                "util": 40.0,
                "mem_used_bytes": 5.0,
                "mem_total_bytes": 10.0,
                "temperature_c": 52.0,
                "power_usage_w": 80.0,
            },
        ),
    )


def _seed_process_samples(conn) -> None:
    for row_id, ts, cpu, ram, used, reserved in (
        (1, 10.0, 80.0, 4e9, 5e9, 6e9),
        (2, 11.0, 40.0, 2e9, 3e9, 4e9),
    ):
        insert_process_sample(
            conn,
            row_id=row_id,
            rank=0,
            ts=ts,
            gpu_available=True,
            gpu_count=1,
            cpu_percent=cpu,
            ram_used_bytes=ram,
            ram_total_bytes=16e9,
            gpu_device_index=0,
            gpu_mem_used_bytes=used,
            gpu_mem_reserved_bytes=reserved,
            gpu_mem_total_bytes=10e9,
        )


def test_system_section_loader_and_builder_use_sqlite_fixture(tmp_path):
    db_path = tmp_path / "system.db"
    with summary_database(db_path) as conn:
        _seed_system_samples(conn)

    data = load_system_section_data(str(db_path))
    result = SystemSummarySection().build(str(db_path))

    assert data.cluster.aggregate.system_samples == 1
    assert data.cluster.nodes["0"].per_gpu[0].util_peak_percent == 70.0
    assert result.section == "system"
    assert result.payload["metadata"]["samples"] == 1
    assert "TraceML System Summary" in result.text
    assert "- Diagnosis: MODERATE GPU UTIL" in result.text
    assert result.payload["diagnosis"]["kind"] == "MODERATE_GPU_UTILIZATION"
    assert result.payload["diagnosis"] == result.payload["issues"][0]
    assert result.payload["diagnosis"]["summary"].startswith(
        "GPU utilization was moderate, averaging 55.0%"
    )
    assert result.payload["diagnosis"]["evidence"]["lowest_util_gpu_idx"] == 1
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
    with summary_database(db_path) as conn:
        _seed_system_samples(conn)
        insert_system_sample(
            conn,
            row_id=2,
            rank=1,
            ts=11.0,
            gpu_available=True,
            gpu_count=1,
            world_size=2,
            local_world_size=1,
            cpu_percent=41.0,
            ram_used_bytes=8.0,
            ram_total_bytes=16.0,
            gpu_samples=(
                {
                    "gpu_idx": 0,
                    "util": 45.0,
                    "mem_used_bytes": 5.0,
                    "mem_total_bytes": 10.0,
                    "temperature_c": 95.0,
                    "power_usage_w": 120.0,
                },
            ),
        )

    payload = SystemSummarySection().build(str(db_path)).payload

    assert payload["metadata"]["nodes_coverage"] == "2/2"
    assert payload["diagnosis"]["kind"] == "HIGH_GPU_TEMPERATURE"
    assert payload["diagnosis"] == payload["issues"][0]
    assert payload["diagnosis"]["evidence"]["scope"] == {
        "level": "gpu",
        "node": "1",
        "node_rank": 1,
        "gpu_idx": 0,
    }
    assert "1 gpu0" in payload["diagnosis"]["summary"]
    assert payload["issues"][0]["evidence"]["scope"]["node"] == "1"
    assert "diagnosis" not in payload["groups"]["rows"]["1"]
    assert "issues" not in payload["groups"]["rows"]["1"]
    assert payload["global"]["worst"]["gpu_temp_c"]["idx"] == "1"


def test_system_loader_uses_selected_time_window(tmp_path):
    db_path = tmp_path / "system_latest.db"
    with summary_database(db_path) as conn:
        _seed_system_samples(conn)
        insert_system_sample(
            conn,
            row_id=2,
            rank=0,
            ts=20.0,
            gpu_available=True,
            gpu_count=1,
            cpu_percent=90.0,
            ram_used_bytes=12.0,
            ram_total_bytes=16.0,
            gpu_samples=(
                {
                    "gpu_idx": 0,
                    "util": 80.0,
                    "mem_used_bytes": 8.0,
                    "mem_total_bytes": 10.0,
                    "temperature_c": 70.0,
                    "power_usage_w": 130.0,
                },
            ),
        )

    data = load_system_section_data(
        str(db_path),
        analysis_window=AnalysisWindow(30.0, start_ts_s=20.0, end_ts_s=20.0),
    )

    assert data.cluster.aggregate.system_samples == 1
    assert data.cluster.aggregate.cpu_avg_percent == 90.0
    assert data.cluster.aggregate.ram_avg_bytes == 12.0
    assert data.cluster.nodes["0"].per_gpu[0].util_peak_percent == 80.0


def test_system_loader_uses_selected_time_window_per_node(tmp_path):
    db_path = tmp_path / "system_latest_per_node.db"
    with summary_database(db_path) as conn:
        _seed_system_samples(conn)
        conn.execute("DELETE FROM system_gpu_samples")
        conn.execute("DELETE FROM system_samples")

        row_id = 1
        for node_rank in (0, 1):
            for seq in (1, 2):
                insert_system_sample(
                    conn,
                    row_id=row_id,
                    rank=node_rank,
                    ts=10.0 + seq,
                    seq=seq,
                    gpu_available=True,
                    gpu_count=1,
                    world_size=2,
                    local_world_size=1,
                    cpu_percent=10.0 * row_id,
                    ram_used_bytes=8.0,
                    ram_total_bytes=16.0,
                    gpu_samples=(
                        {
                            "gpu_idx": 0,
                            "util": 60.0 + row_id,
                            "mem_used_bytes": 5.0,
                            "mem_total_bytes": 10.0,
                            "temperature_c": 68.0,
                            "power_usage_w": 120.0,
                        },
                    ),
                )
                row_id += 1

    data = load_system_section_data(
        str(db_path),
        analysis_window=AnalysisWindow(30.0, start_ts_s=12.0, end_ts_s=12.0),
    )

    assert data.cluster.aggregate.system_samples == 2
    assert set(data.cluster.nodes) == {"0", "1"}
    assert data.cluster.nodes["0"].aggregate.cpu_avg_percent == 20.0
    assert data.cluster.nodes["1"].aggregate.cpu_avg_percent == 40.0
    assert data.cluster.nodes["0"].per_gpu[0].util_peak_percent == 62.0
    assert data.cluster.nodes["1"].per_gpu[0].util_peak_percent == 64.0


def test_process_section_loader_and_builder_use_sqlite_fixture(tmp_path):
    db_path = tmp_path / "process.db"
    with summary_database(db_path) as conn:
        _seed_process_samples(conn)

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


def test_process_loader_uses_selected_time_window(tmp_path):
    db_path = tmp_path / "process_latest.db"
    with summary_database(db_path) as conn:
        _seed_process_samples(conn)

    data = load_process_section_data(
        str(db_path),
        analysis_window=AnalysisWindow(30.0, start_ts_s=11.0, end_ts_s=11.0),
    )

    assert data.aggregate.process_samples == 1
    assert data.aggregate.cpu_avg_percent == 40.0
    assert data.aggregate.ram_avg_bytes == 2000000000.0
    assert data.per_global_rank[0].cpu_avg_percent == 40.0
    assert data.per_global_rank[0].gpu_mem_reserved_avg_bytes == 4000000000.0


def test_process_loader_ignores_rows_without_global_rank(tmp_path):
    db_path = tmp_path / "process_legacy_rank.db"
    with summary_database(db_path) as conn:
        _seed_process_samples(conn)
        insert_process_sample(
            conn,
            row_id=3,
            rank=9,
            global_rank=None,
            hostname="legacy-worker",
            ts=12.0,
            gpu_available=True,
            gpu_count=1,
            cpu_percent=99.0,
            ram_used_bytes=9e9,
            ram_total_bytes=16e9,
            gpu_device_index=0,
            gpu_mem_used_bytes=9e9,
            gpu_mem_reserved_bytes=9e9,
            gpu_mem_total_bytes=10e9,
        )

    data = load_process_section_data(str(db_path))

    assert data.aggregate.process_samples == 2
    assert data.aggregate.distinct_global_ranks == 1
    assert set(data.per_global_rank) == {0}


def test_summary_wrappers_delegate_to_section_paths(tmp_path):
    db_path = tmp_path / "combined.db"
    with summary_database(db_path) as conn:
        _seed_system_samples(conn)
        _seed_process_samples(conn)

    system = generate_system_summary_card(str(db_path), print_to_stdout=False)
    process = generate_process_summary_card(
        str(db_path), print_to_stdout=False
    )

    assert system["metadata"]["samples"] == 1
    assert process["metadata"]["samples"] == 2
    assert (tmp_path / "combined.db_summary_card.json").exists()
    assert (tmp_path / "combined.db_summary_card.txt").exists()
