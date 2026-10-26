# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
from contextlib import closing

import pytest

from tests.sqlite_fixtures import (
    insert_process_sample,
    insert_system_sample,
    summary_database,
)
from traceml_ai.measurements.process import read_process_measurements
from traceml_ai.measurements.system import read_system_measurements
from traceml_ai.measurements.window import (
    AnalysisWindow,
    resolve_analysis_window,
)
from traceml_ai.reporting.analysis_window import build_analysis_window_payload
from traceml_ai.reporting.sections.process import ProcessSummarySection
from traceml_ai.reporting.sections.system import SystemSummarySection
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
                "observed_timestamp_unix_ns": 10,
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
        (1, 10.0, 80.0, 4 * 1024**3, 5e9, 6e9),
        (2, 11.0, 40.0, 2 * 1024**3, 3e9, 4e9),
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
            ram_total_bytes=16 * 1024**3,
            gpu_device_index=0,
            gpu_mem_used_bytes=used,
            gpu_mem_reserved_bytes=reserved,
            gpu_mem_total_bytes=10e9,
        )


def _system_rows(conn, rows, *, ts_values=(10.0, 11.0)) -> None:
    """Insert rows of (global_rank, node_rank, world, local_world, host)."""
    row_id = 0
    for global_rank, node_rank, world_size, local_world_size, host in rows:
        for ts in ts_values:
            row_id += 1
            insert_system_sample(
                conn,
                row_id=row_id,
                rank=global_rank,
                ts=ts,
                gpu_available=True,
                gpu_count=1,
                node_rank=node_rank,
                global_rank=global_rank,
                world_size=world_size,
                local_world_size=local_world_size,
                hostname=host,
                gpu_samples=(
                    {
                        "gpu_idx": 0,
                        "util": 50.0,
                        "mem_used_bytes": 1.0,
                        "mem_total_bytes": 2.0,
                    },
                ),
            )


def test_rows_without_node_rank_stay_one_node_per_host(tmp_path):
    db_path = tmp_path / "system.db"
    with summary_database(db_path) as conn:
        _system_rows(
            conn,
            [
                (0, None, 1, 1, "host-a"),
                (1, None, 1, 1, "host-a"),
                (2, None, 1, 1, "host-b"),
            ],
        )

    result = SystemSummarySection().build(str(db_path))

    # A global rank is not a node rank, so it no longer labels a node, and
    # different hosts are not merged into one.
    rows = result.payload["groups"]["rows"]
    assert sorted(rows) == ["unknown@host-a", "unknown@host-b"]
    assert all(row["identity"]["node_rank"] is None for row in rows.values())
    assert result.payload["metadata"]["nodes_observed"] == 2
    assert result.payload["metadata"]["gpus_observed"] == 2


def test_multi_node_card_marks_a_node_without_a_rank():
    from traceml_ai.reporting.sections.system.builder import _fmt_node_idx

    assert _fmt_node_idx("3") == "n3"
    assert _fmt_node_idx("unknown@host-b") == "n?"
    assert _fmt_node_idx(None) == "n/a"


@pytest.mark.parametrize(
    ("rows", "mode"),
    [
        pytest.param([(0, 0, 4, 4, "a")], "single_node", id="one_node"),
        pytest.param(
            [(0, 0, 8, 4, "a"), (4, 1, 8, 4, "b")],
            "multi_node",
            id="two_node_ranks",
        ),
        # The second node never reported, but the world size says it exists.
        pytest.param(
            [(0, 0, 8, 4, "a")], "multi_node", id="silent_second_node"
        ),
        pytest.param(
            [(0, None, 8, 4, "a")],
            "multi_node",
            id="no_node_rank_wider_world",
        ),
        pytest.param(
            [(0, None, 1, 1, "a"), (1, None, 1, 1, "a")],
            "single_node",
            id="no_node_rank_two_ranks",
        ),
        pytest.param(
            [(rank, None, 8, 8, "a") for rank in range(8)],
            "single_node",
            id="one_host_eight_ranks",
        ),
    ],
)
def test_system_mode_follows_the_shared_identity_rule(tmp_path, rows, mode):
    db_path = tmp_path / "system.db"
    with summary_database(db_path) as conn:
        _system_rows(conn, rows)

    result = SystemSummarySection().build(str(db_path))

    assert result.payload["metadata"]["mode"] == mode


def test_topology_reads_every_sample_not_only_the_last(tmp_path):
    # The node's later sample lost its world sizes; the earlier one says
    # the world is twice the local world, so a second node exists.
    db_path = tmp_path / "system.db"
    with summary_database(db_path) as conn:
        _system_rows(conn, [(0, 0, 8, 4, "a")], ts_values=(10.0,))
        conn.execute(
            "INSERT INTO system_samples("
            "observed_timestamp_unix_ns, global_rank, local_rank, "
            "world_size, local_world_size, node_rank, hostname, "
            "timestamp_unix_ns) VALUES (99, 0, 0, NULL, NULL, 0, 'a', "
            "11000000000)"
        )

    result = SystemSummarySection().build(str(db_path))

    assert result.payload["metadata"]["mode"] == "multi_node"


def test_system_measurements_and_builder_use_sqlite_fixture(tmp_path):
    db_path = tmp_path / "system.db"
    with summary_database(db_path) as conn:
        _seed_system_samples(conn)
        conn.execute(
            "INSERT INTO system_samples("
            "observed_timestamp_unix_ns, global_rank, local_rank, world_size, "
            "local_world_size, node_rank, hostname, timestamp_unix_ns"
            ") VALUES (11, 0, 0, 1, 1, 0, 'worker-0', 11000000000)"
        )

    with closing(sqlite3.connect(db_path)) as connection:
        data = read_system_measurements(connection)
    result = SystemSummarySection().build(str(db_path))

    assert data.cluster.aggregate.system_samples == 2
    assert data.cluster.nodes["0"].per_gpu[0].util_peak_percent == 70.0
    assert result.section == "system"
    assert result.payload["metadata"]["samples"] == 2
    assert result.payload["metadata"]["duration_s"] == 1.0
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
    assert "on node 1, gpu0" in payload["diagnosis"]["summary"]
    assert payload["issues"][0]["evidence"]["scope"]["node"] == "1"
    assert "diagnosis" not in payload["groups"]["rows"]["1"]
    assert "issues" not in payload["groups"]["rows"]["1"]
    assert payload["global"]["worst"]["gpu_temp_c"]["idx"] == "1"


def test_system_measurements_use_selected_time_window(tmp_path):
    db_path = tmp_path / "system_latest.db"
    timestamp_s = 1_792_336_915.3680508
    with summary_database(db_path) as conn:
        _seed_system_samples(conn)
        insert_system_sample(
            conn,
            row_id=2,
            rank=0,
            ts=timestamp_s,
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

    with closing(sqlite3.connect(db_path)) as connection:
        window = resolve_analysis_window(connection, retention_s=1.0)
    assert window.end_time_unix_ns == round(timestamp_s * 1_000_000_000)
    with closing(sqlite3.connect(db_path)) as connection:
        data = read_system_measurements(
            connection,
            analysis_window=window,
        )

    assert data.cluster.aggregate.system_samples == 1
    assert data.cluster.aggregate.cpu_avg_percent == 90.0
    assert data.cluster.aggregate.ram_avg_bytes == 12.0
    assert data.cluster.nodes["0"].per_gpu[0].util_peak_percent == 80.0


def test_system_measurements_use_selected_time_window_per_node(tmp_path):
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

    with closing(sqlite3.connect(db_path)) as connection:
        data = read_system_measurements(
            connection,
            analysis_window=AnalysisWindow(
                30.0,
                start_time_unix_ns=12_000_000_000,
                end_time_unix_ns=12_000_000_000,
            ),
        )

    assert data.cluster.aggregate.system_samples == 2
    assert set(data.cluster.nodes) == {"0", "1"}
    assert data.cluster.nodes["0"].aggregate.cpu_avg_percent == 20.0
    assert data.cluster.nodes["1"].aggregate.cpu_avg_percent == 40.0
    assert data.cluster.nodes["0"].per_gpu[0].util_peak_percent == 62.0
    assert data.cluster.nodes["1"].per_gpu[0].util_peak_percent == 64.0


def test_process_measurements_and_builder_use_sqlite_fixture(tmp_path):
    db_path = tmp_path / "process.db"
    with summary_database(db_path) as conn:
        _seed_process_samples(conn)

    with closing(sqlite3.connect(db_path)) as connection:
        data = read_process_measurements(connection)
    result = ProcessSummarySection().build(str(db_path))

    assert data.aggregate.process_samples == 2
    assert result.section == "process"
    assert result.payload["metadata"]["samples"] == 2
    assert result.payload["metadata"]["duration_s"] == 1.0
    assert "TraceML Process Summary" in result.text
    assert "- Diagnosis: NORMAL" in result.text
    assert (
        "- Stats: global ranks 1 | CPU avg 60% | "
        "RSS peak 4.0 / 16.0 GiB | GPU reserved peak 60%"
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


def test_process_measurements_use_selected_time_window(tmp_path):
    db_path = tmp_path / "process_latest.db"
    boundary_unix_ns = 1_792_336_915_368_049_000
    with summary_database(db_path) as conn:
        _seed_process_samples(conn)
        conn.execute(
            "UPDATE process_samples SET timestamp_unix_ns = ? WHERE id = 2",
            (boundary_unix_ns,),
        )
        conn.execute(
            "UPDATE process_samples SET timestamp_unix_ns = ? WHERE id = 1",
            (boundary_unix_ns - 1,),
        )
        conn.execute(
            "INSERT INTO process_samples(observed_timestamp_unix_ns, global_rank, "
            "timestamp_unix_ns, cpu_percent) VALUES (3, 0, ?, 90.0)",
            (boundary_unix_ns + 1,),
        )

    window = AnalysisWindow(
        30.0,
        start_time_unix_ns=boundary_unix_ns,
        end_time_unix_ns=boundary_unix_ns,
    )
    with closing(sqlite3.connect(db_path)) as connection:
        data = read_process_measurements(
            connection,
            analysis_window=window,
        )

    assert (
        build_analysis_window_payload(str(db_path), window)["sections"][
            "process"
        ]["samples"]
        == 1
    )
    assert data.aggregate.process_samples == 1
    assert data.aggregate.cpu_avg_percent == 40.0
    assert data.aggregate.ram_avg_bytes == 2 * 1024**3
    assert data.per_global_rank[0].cpu_avg_percent == 40.0
    assert data.per_global_rank[0].gpu_mem_reserved_avg_bytes == 4000000000.0


def test_process_measurements_ignore_rows_without_global_rank(tmp_path):
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

    with closing(sqlite3.connect(db_path)) as connection:
        data = read_process_measurements(connection)

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
