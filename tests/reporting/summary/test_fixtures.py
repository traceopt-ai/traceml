# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""SQLite fixture coverage for final-report summary sections.

These tests keep section assertions schema-oriented instead of snapshotting full
cards. The goal is to make contributor changes safe while leaving copy/layout
free to evolve intentionally.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from traceml_ai.aggregator.sqlite_writers import (
    process as process_projection,
    step_memory as step_memory_projection,
    step_time as step_time_projection,
    system as system_projection,
)
from traceml_ai.reporting.final import build_summary_payload
from traceml_ai.reporting.schema import BaseSectionPayload
from traceml_ai.reporting.sections.process import ProcessSummarySection
from traceml_ai.reporting.sections.step_memory import StepMemorySummarySection
from traceml_ai.reporting.sections.step_time import StepTimeSummarySection
from traceml_ai.reporting.sections.system import SystemSummarySection


SECTION_KEYS = {
    "metadata",
    "diagnosis",
    "issues",
    "global",
    "groups",
    "units",
    "card",
}

GLOBAL_KEYS = {
    "index_by",
    "window",
    "average",
    "median",
    "worst",
}

GLOBAL_WINDOW_KEYS = {
    "kind",
    "alignment",
    "samples",
    "steps_analyzed",
    "start_step",
    "end_step",
    "completed_step",
    "window_size",
}

METADATA_KEYS = {
    "mode",
    "duration_s",
    "samples",
    "nodes_expected",
    "nodes_observed",
    "nodes_coverage",
    "nodes_partial",
    "gpus_observed",
    "global_ranks_seen",
    "global_ranks_used",
    "training_total_steps",
    "training_latest_step",
    "section_metric_names",
}

GROUP_ROW_KEYS = {
    "identity",
    "metrics",
}

IDENTITY_KEYS = {
    "global_rank",
    "local_rank",
    "node_rank",
    "hostname",
    "local_world_size",
    "world_size",
}


def _assert_section_shape(payload: dict, *, group_by: str) -> None:
    assert set(payload) == SECTION_KEYS
    assert set(payload["metadata"]) == METADATA_KEYS
    assert set(payload["global"]) == GLOBAL_KEYS
    assert payload["global"]["index_by"] in {"node_rank", "global_rank"}
    assert payload["global"]["index_by"] == group_by
    assert set(payload["global"]["window"]) == GLOBAL_WINDOW_KEYS
    metric_names = payload["metadata"]["section_metric_names"]
    if metric_names is not None:
        expected_metrics = set(metric_names)
        assert set(payload["global"]["average"]) == expected_metrics
        assert set(payload["global"]["median"]) == expected_metrics
        assert set(payload["global"]["worst"]) == expected_metrics
        for metric in expected_metrics:
            assert set(payload["global"]["median"][metric]) == {
                "value",
                "idx",
            }
            assert set(payload["global"]["worst"][metric]) == {
                "value",
                "idx",
            }
    assert payload["groups"]["by"] == group_by
    assert isinstance(payload["groups"]["rows"], dict)
    for row in payload["groups"]["rows"].values():
        assert set(row) == GROUP_ROW_KEYS
        assert set(row["identity"]) == IDENTITY_KEYS
        if metric_names is not None:
            assert set(row["metrics"]) == set(metric_names)


def test_base_section_payload_rejects_metric_contract_mismatch() -> None:
    try:
        BaseSectionPayload(
            metadata={"section_metric_names": ["cpu_percent"]},
            diagnosis=None,
            issues=[],
            global_summary={
                "index_by": "global_rank",
                "window": {},
                "average": {"cpu_percent": 1.0, "extra_metric": 2.0},
                "median": {"cpu_percent": {"value": 1.0, "idx": "0"}},
                "worst": {"cpu_percent": {"value": 1.0, "idx": "0"}},
            },
            groups={"by": "global_rank", "rows": {}},
            units={},
            card="",
        ).to_json()
    except ValueError as exc:
        assert "section_metric_names" in str(exc)
    else:
        raise AssertionError("Expected metric contract mismatch to fail")


def test_base_section_payload_rejects_group_index_mismatch() -> None:
    try:
        BaseSectionPayload(
            metadata={"section_metric_names": ["cpu_percent"]},
            diagnosis=None,
            issues=[],
            global_summary={
                "index_by": "global_rank",
                "window": {},
                "average": {"cpu_percent": 1.0},
                "median": {"cpu_percent": {"value": 1.0, "idx": "0"}},
                "worst": {"cpu_percent": {"value": 1.0, "idx": "0"}},
            },
            groups={"by": "node_rank", "rows": {}},
            units={},
            card="",
        ).to_json()
    except ValueError as exc:
        assert "groups.by" in str(exc)
    else:
        raise AssertionError("Expected group index mismatch to fail")


def test_base_section_payload_rejects_group_metric_mismatch() -> None:
    try:
        BaseSectionPayload(
            metadata={"section_metric_names": ["cpu_percent"]},
            diagnosis=None,
            issues=[],
            global_summary={
                "index_by": "global_rank",
                "window": {},
                "average": {"cpu_percent": 1.0},
                "median": {"cpu_percent": {"value": 1.0, "idx": "0"}},
                "worst": {"cpu_percent": {"value": 1.0, "idx": "0"}},
            },
            groups={
                "by": "global_rank",
                "rows": {
                    "0": {
                        "identity": {},
                        "metrics": {
                            "cpu_percent": 1.0,
                            "extra_metric": 2.0,
                        },
                    },
                },
            },
            units={},
            card="",
        ).to_json()
    except ValueError as exc:
        assert "groups.rows" in str(exc)
        assert "section_metric_names" in str(exc)
    else:
        raise AssertionError("Expected group metric mismatch to fail")


def test_base_section_payload_rejects_extra_group_row_fields() -> None:
    try:
        BaseSectionPayload(
            metadata={"section_metric_names": ["cpu_percent"]},
            diagnosis=None,
            issues=[],
            global_summary={
                "index_by": "global_rank",
                "window": {},
                "average": {"cpu_percent": 1.0},
                "median": {"cpu_percent": {"value": 1.0, "idx": "0"}},
                "worst": {"cpu_percent": {"value": 1.0, "idx": "0"}},
            },
            groups={
                "by": "global_rank",
                "rows": {
                    "0": {
                        "identity": {},
                        "diagnosis": None,
                        "metrics": {"cpu_percent": 1.0},
                    },
                },
            },
            units={},
            card="",
        ).to_json()
    except ValueError as exc:
        assert "identity and metrics" in str(exc)
    else:
        raise AssertionError("Expected extra group row fields to fail")


def _connect_with_summary_schema(db_path: Path) -> sqlite3.Connection:
    """Create a tiny TraceML SQLite fixture with all summary tables."""
    conn = sqlite3.connect(db_path)
    system_projection.init_schema(conn)
    process_projection.init_schema(conn)
    step_time_projection.init_schema(conn)
    step_memory_projection.init_schema(conn)
    return conn


def _insert_system_sample(
    conn: sqlite3.Connection,
    *,
    row_id: int,
    rank: int,
    ts: float,
    gpu_available: bool,
    gpu_count: int,
    gpu_util: float | None = None,
    world_size: int = 1,
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
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row_id,
            rank,
            0,
            world_size,
            local_world_size,
            rank,
            f"worker-{rank}",
            ts,
            row_id,
            30.0 + rank,
            4_000.0 + rank,
            16_000.0,
            int(gpu_available),
            gpu_count,
            gpu_util,
            gpu_util,
            2_000.0 if gpu_available else None,
            2_500.0 if gpu_available else None,
            None,
            None,
            None,
            None,
        ),
    )
    if gpu_available and gpu_count > 0:
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                row_id,
                rank,
                0,
                world_size,
                local_world_size,
                rank,
                f"worker-{rank}",
                ts,
                row_id,
                rank,
                gpu_util,
                2_500.0,
                10_000.0,
                None,
                None,
                None,
            ),
        )


def _insert_process_sample(
    conn: sqlite3.Connection,
    *,
    row_id: int,
    rank: int,
    ts: float,
    gpu_available: bool,
    gpu_count: int,
) -> None:
    conn.execute(
        """
        INSERT INTO process_samples(
            recv_ts_ns,
            rank,
            global_rank,
            sample_ts_s,
            seq,
            cpu_percent,
            cpu_logical_core_count,
            ram_used_bytes,
            ram_total_bytes,
            gpu_available,
            gpu_count,
            gpu_device_index,
            gpu_mem_used_bytes,
            gpu_mem_reserved_bytes,
            gpu_mem_total_bytes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row_id,
            rank,
            rank,
            ts,
            row_id,
            50.0 + rank,
            8,
            1_000.0 + rank * 100.0,
            16_000.0,
            int(gpu_available),
            gpu_count,
            rank if gpu_available else None,
            2_000.0 + rank * 500.0 if gpu_available else None,
            2_500.0 + rank * 600.0 if gpu_available else None,
            10_000.0 if gpu_available else None,
        ),
    )


def _step_time_events(
    *,
    dataloader: float,
    forward: float,
    backward: float,
    optimizer: float,
    step_time: float,
) -> str:
    events = {
        "_traceml_internal:dataloader_next": {
            "cpu": {"is_gpu": False, "duration_ms": dataloader, "n_calls": 1}
        },
        "_traceml_internal:forward_time": {
            "cpu": {"is_gpu": False, "duration_ms": forward, "n_calls": 1}
        },
        "_traceml_internal:backward_time": {
            "cpu": {"is_gpu": False, "duration_ms": backward, "n_calls": 1}
        },
        "_traceml_internal:optimizer_step": {
            "cpu": {"is_gpu": False, "duration_ms": optimizer, "n_calls": 1}
        },
        "_traceml_internal:step_time": {
            "cpu": {"is_gpu": False, "duration_ms": step_time, "n_calls": 1}
        },
    }
    return json.dumps(events)


def _insert_step_time_sample(
    conn: sqlite3.Connection,
    *,
    row_id: int,
    rank: int,
    step: int,
    step_time: float,
) -> None:
    conn.execute(
        """
        INSERT INTO step_time_samples(
            recv_ts_ns,
            rank,
            global_rank,
            local_rank,
            world_size,
            local_world_size,
            node_rank,
            hostname,
            runtime_pid,
            sample_ts_s,
            seq,
            step,
            events_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row_id,
            rank,
            rank,
            0,
            1,
            1,
            rank,
            f"worker-{rank}",
            10_000 + rank,
            float(step),
            row_id,
            step,
            _step_time_events(
                dataloader=1.0,
                forward=2.0 + rank,
                backward=3.0 + rank,
                optimizer=1.0,
                step_time=step_time,
            ),
        ),
    )


def _insert_step_memory_sample(
    conn: sqlite3.Connection,
    *,
    row_id: int,
    rank: int,
    step: int,
    alloc: float | None,
    reserved: float | None,
    device: str | None = "cuda:0",
    world_size: int = 1,
    local_world_size: int = 1,
) -> None:
    conn.execute(
        """
        INSERT INTO step_memory_samples(
            recv_ts_ns,
            rank,
            global_rank,
            local_rank,
            world_size,
            local_world_size,
            node_rank,
            hostname,
            sample_ts_s,
            seq,
            model_id,
            device,
            step,
            peak_alloc_bytes,
            peak_reserved_bytes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row_id,
            rank,
            rank,
            0,
            world_size,
            local_world_size,
            rank,
            f"worker-{rank}",
            float(step),
            row_id,
            1,
            device,
            step,
            alloc,
            reserved,
        ),
    )


def test_summary_sections_handle_empty_tables_with_stable_schema(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "empty.db"
    conn = _connect_with_summary_schema(db_path)
    conn.close()

    sections = (
        SystemSummarySection(),
        ProcessSummarySection(),
        StepTimeSummarySection(max_rows=4),
        StepMemorySummarySection(window_size=4),
    )

    results = {
        section.name: section.build(str(db_path)) for section in sections
    }

    assert results["system"].payload["metadata"]["samples"] == 0
    assert results["process"].payload["metadata"]["samples"] == 0
    assert results["step_time"].payload["metadata"]["mode"] == "no_data"
    assert (
        results["step_memory"].payload["global"]["window"]["steps_analyzed"]
        == 0
    )
    assert "steps_used" not in results["step_memory"].payload["metadata"]
    _assert_section_shape(results["system"].payload, group_by="node_rank")
    _assert_section_shape(
        results["process"].payload,
        group_by="global_rank",
    )
    _assert_section_shape(
        results["step_time"].payload,
        group_by="global_rank",
    )
    _assert_section_shape(
        results["step_memory"].payload,
        group_by="global_rank",
    )
    for name, result in results.items():
        assert result.section == name
        assert "card" in result.payload
        assert "diagnosis" in result.payload
        assert isinstance(result.payload["issues"], list)
        assert result.text == result.payload["card"]


def test_summary_sections_cover_single_rank_gpu_run(tmp_path: Path) -> None:
    db_path = tmp_path / "single_rank.db"
    conn = _connect_with_summary_schema(db_path)
    try:
        _insert_system_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=True,
            gpu_count=1,
            gpu_util=70.0,
        )
        _insert_process_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=True,
            gpu_count=1,
        )
        for step in range(1, 5):
            _insert_step_time_sample(
                conn,
                row_id=step,
                rank=0,
                step=step,
                step_time=10.0,
            )
            _insert_step_memory_sample(
                conn,
                row_id=step,
                rank=0,
                step=step,
                alloc=100.0 + step,
                reserved=200.0 + step,
            )
        conn.commit()
    finally:
        conn.close()

    system = SystemSummarySection().build(str(db_path)).payload
    process = ProcessSummarySection().build(str(db_path)).payload
    step_time = StepTimeSummarySection(max_rows=4).build(str(db_path)).payload
    step_memory = (
        StepMemorySummarySection(window_size=4).build(str(db_path)).payload
    )

    assert system["metadata"]["nodes_coverage"] == "1/1"
    assert system["metadata"]["gpus_observed"] == 1
    assert system["metadata"]["mode"] == "single_node"
    assert system["global"]["average"]["gpu_util_percent"] is not None
    _assert_section_shape(system, group_by="node_rank")
    _assert_section_shape(process, group_by="global_rank")
    _assert_section_shape(step_time, group_by="global_rank")
    _assert_section_shape(step_memory, group_by="global_rank")
    assert set(system["groups"]["rows"]) == {"0"}
    assert process["metadata"]["global_ranks_seen"] == 1
    assert step_time["metadata"]["mode"] == "single_node"
    assert step_time["metadata"]["global_ranks_seen"] == 1
    assert step_time["global"]["median"]["total_step_ms"]["value"] == 11.0
    assert step_time["units"] == {"time": "ms"}
    assert step_memory["metadata"]["global_ranks_seen"] == 1
    assert step_memory["metadata"]["global_ranks_used"] == 1
    assert "aligned_steps_analyzed" not in step_memory["metadata"]
    assert "steps_used" not in step_memory["metadata"]
    assert step_memory["global"]["window"]["steps_analyzed"] == 4
    assert set(step_memory["global"]["average"]) == {
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }


def test_summary_sections_cover_multi_rank_aligned_run(tmp_path: Path) -> None:
    db_path = tmp_path / "multi_rank.db"
    conn = _connect_with_summary_schema(db_path)
    try:
        gpu_mem_by_rank = {0: 2_500.0, 1: 5_000.0}
        for rank in (0, 1):
            _insert_system_sample(
                conn,
                row_id=10 + rank,
                rank=rank,
                ts=1.0 + rank,
                gpu_available=True,
                gpu_count=2,
                gpu_util=80.0 - rank * 20.0,
                world_size=2,
                local_world_size=1,
            )
            conn.execute(
                """
                UPDATE system_gpu_samples
                SET mem_used_bytes = ?
                WHERE recv_ts_ns = ? AND node_rank = ?;
                """,
                (gpu_mem_by_rank[rank], 10 + rank, rank),
            )
            _insert_process_sample(
                conn,
                row_id=10 + rank,
                rank=rank,
                ts=1.0 + rank,
                gpu_available=True,
                gpu_count=2,
            )
            for step in range(1, 6):
                row_id = rank * 100 + step
                _insert_step_time_sample(
                    conn,
                    row_id=row_id,
                    rank=rank,
                    step=step,
                    step_time=10.0 + rank,
                )
                _insert_step_memory_sample(
                    conn,
                    row_id=row_id,
                    rank=rank,
                    step=step,
                    alloc=100.0 + rank * 20.0 + step,
                    reserved=200.0 + rank * 30.0 + step,
                    device=f"cuda:{rank}",
                    world_size=2,
                    local_world_size=1,
                )
        conn.commit()
    finally:
        conn.close()

    step_time = StepTimeSummarySection(max_rows=5).build(str(db_path)).payload
    step_memory = (
        StepMemorySummarySection(window_size=5).build(str(db_path)).payload
    )
    process = ProcessSummarySection().build(str(db_path)).payload

    system = SystemSummarySection().build(str(db_path)).payload
    assert system["global"]["average"]["gpu_headroom_bytes"] == 6_250.0
    assert system["global"]["worst"]["gpu_headroom_bytes"]["value"] == 5_000.0
    assert step_time["metadata"]["mode"] == "multi_node"
    assert step_time["metadata"]["global_ranks_seen"] == 2
    assert step_time["global"]["window"]["steps_analyzed"] == 5
    assert step_time["global"]["window"]["window_size"] == 5
    assert "aligned_steps_analyzed" not in step_time["metadata"]
    assert set(step_time["groups"]["rows"]) == {"0", "1"}
    assert step_memory["metadata"]["global_ranks_seen"] == 2
    assert step_memory["metadata"]["global_ranks_used"] == 2
    assert "aligned_steps_analyzed" not in step_memory["metadata"]
    assert "steps_used" not in step_memory["metadata"]
    assert step_memory["global"]["window"]["steps_analyzed"] == 5
    assert set(step_memory["groups"]["rows"]) == {"0", "1"}
    assert (
        step_memory["global"]["median"]["peak_allocated_bytes"]["idx"]
        in step_memory["groups"]["rows"]
    )
    assert (
        step_memory["global"]["median"]["peak_reserved_bytes"]["idx"]
        in step_memory["groups"]["rows"]
    )
    assert process["metadata"]["global_ranks_seen"] == 2
    assert set(process["groups"]["rows"]) == {"0", "1"}


def test_step_memory_section_reports_no_gpu_without_throwing(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "no_gpu.db"
    conn = _connect_with_summary_schema(db_path)
    try:
        _insert_system_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=False,
            gpu_count=0,
            gpu_util=None,
        )
        _insert_process_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=False,
            gpu_count=0,
        )
        _insert_step_memory_sample(
            conn,
            row_id=1,
            rank=0,
            step=1,
            alloc=None,
            reserved=None,
            device=None,
        )
        conn.commit()
    finally:
        conn.close()

    payload = (
        StepMemorySummarySection(window_size=4).build(str(db_path)).payload
    )

    assert payload["metadata"]["training_total_steps"] == 2
    assert payload["diagnosis"]["status"] == "NO GPU"
    assert "action" not in payload["diagnosis"]
    assert "- Next:" not in payload["card"]
    assert payload["card"].index("- Diagnosis:") < payload["card"].index(
        "- Scope:"
    )
    assert payload["card"].index("- Scope:") < payload["card"].index(
        "- Stats:"
    )
    assert payload["card"].index("- Stats:") < payload["card"].index("- Why:")
    assert payload["global"]["window"]["steps_analyzed"] == 0
    assert "steps_used" not in payload["global"]["window"]
    assert payload["issues"] == []


def test_final_summary_fixture_schema_contains_all_sections(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "final.db"
    conn = _connect_with_summary_schema(db_path)
    try:
        _insert_system_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=False,
            gpu_count=0,
            gpu_util=None,
        )
        _insert_process_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=False,
            gpu_count=0,
        )
        conn.commit()
    finally:
        conn.close()

    payload = build_summary_payload(str(db_path))

    assert payload["schema_version"] == 1.3
    assert set(payload) == {
        "schema_version",
        "generated_at",
        "duration_s",
        "system",
        "process",
        "step_time",
        "step_memory",
        "text",
    }
    for key in ("system", "process", "step_time", "step_memory"):
        assert "metadata" in payload[key]
        assert "card" in payload[key]
        assert "diagnosis" in payload[key]
        assert "- Next:" not in payload[key]["card"]
        diagnosis = payload[key]["diagnosis"]
        if diagnosis is not None:
            assert "action" not in diagnosis
    assert payload["system"]["diagnosis"]["status"] == "NORMAL"
    assert "NO GPU" not in payload["system"]["card"]
    assert "- Next:" not in payload["text"]
