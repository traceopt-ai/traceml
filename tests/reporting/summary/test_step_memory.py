# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

import sqlite3

from traceml_ai.diagnostics.step_memory import SUMMARY_STEP_MEMORY_POLICY
from traceml_ai.reporting.sections.step_memory import StepMemorySummarySection
from traceml_ai.reporting.sections.step_memory.loader import (
    load_step_memory_section_data,
)
from traceml_ai.reporting.summaries.step_memory import (
    generate_step_memory_summary_card,
)


def _create_step_memory_db(path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE step_memory_samples (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                recv_ts_ns           INTEGER NOT NULL,
                rank                 INTEGER,
                global_rank          INTEGER,
                local_rank           INTEGER,
                world_size           INTEGER,
                local_world_size     INTEGER,
                node_rank            INTEGER,
                hostname             TEXT,
                sample_ts_s          REAL,
                seq                  INTEGER,
                model_id             INTEGER,
                device               TEXT,
                step                 INTEGER,
                peak_alloc_bytes     REAL,
                peak_reserved_bytes  REAL
            );
            """
        )
        rows = [
            (
                1,
                0,
                0,
                0,
                1,
                1,
                0,
                "worker-0",
                1.0,
                1,
                10,
                "cuda:0",
                1,
                100.0,
                200.0,
            ),
            (
                2,
                0,
                0,
                0,
                1,
                1,
                0,
                "worker-0",
                2.0,
                2,
                10,
                "cuda:0",
                2,
                110.0,
                210.0,
            ),
            (
                3,
                0,
                0,
                0,
                1,
                1,
                0,
                "worker-0",
                3.0,
                3,
                10,
                "cuda:0",
                3,
                120.0,
                220.0,
            ),
        ]
        conn.executemany(
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
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def test_step_memory_section_loader_and_builder_use_sqlite_fixture(tmp_path):
    db_path = tmp_path / "memory.db"
    _create_step_memory_db(str(db_path))

    data = load_step_memory_section_data(str(db_path), window_size=3)
    result = StepMemorySummarySection(window_size=3).build(str(db_path))

    assert data.training_steps == 4
    assert data.latest_step_observed == 3
    assert [metric.metric for metric in data.metrics] == [
        "peak_allocated",
        "peak_reserved",
    ]
    assert result.section == "step_memory"
    assert result.payload["metadata"]["mode"] == "single_node"
    assert result.payload["metadata"]["global_ranks_seen"] == 1
    assert result.payload["metadata"]["global_ranks_used"] == 1
    assert "aligned_steps_analyzed" not in result.payload["metadata"]
    assert "steps_used" not in result.payload["metadata"]
    assert result.payload["global"]["window"]["steps_analyzed"] == 3
    assert (
        result.payload["global"]["median"]["peak_allocated_bytes"]["idx"]
        == "0"
    )
    assert (
        result.payload["global"]["median"]["peak_reserved_bytes"]["idx"] == "0"
    )
    assert result.payload["groups"]["rows"]["0"]["identity"] == {
        "global_rank": 0,
        "local_rank": 0,
        "node_rank": 0,
        "hostname": "worker-0",
        "local_world_size": 1,
        "world_size": 1,
    }
    assert "TraceML Step Memory Summary" in result.text
    assert "- Diagnosis:" in result.text
    assert "- Scope:" in result.text
    assert "- Stats:" in result.text
    assert "- Why:" in result.text
    assert result.text.index("- Diagnosis:") < result.text.index("- Scope:")
    assert result.text.index("- Scope:") < result.text.index("- Stats:")
    assert result.text.index("- Stats:") < result.text.index("- Why:")
    assert "- Primary:" not in result.text
    assert "- Trend:" not in result.text
    assert "- Note:" not in result.text
    assert "- Issues:" not in result.text


def test_step_memory_section_diagnosis_input_uses_summary_policy(tmp_path):
    db_path = tmp_path / "memory.db"
    _create_step_memory_db(str(db_path))

    section = StepMemorySummarySection(window_size=3)
    data = section.load(str(db_path))
    diagnosis_input = section.to_diagnosis_input(data)

    assert diagnosis_input.thresholds is SUMMARY_STEP_MEMORY_POLICY.thresholds
    assert len(diagnosis_input.metrics) == 2


def test_step_memory_section_reports_no_gpu_without_memory_rows(tmp_path):
    db_path = tmp_path / "memory.db"
    _create_step_memory_db(str(db_path))

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DELETE FROM step_memory_samples;")
        conn.execute("CREATE TABLE system_samples (gpu_available INTEGER);")
        conn.execute("INSERT INTO system_samples(gpu_available) VALUES (0);")
        conn.commit()
    finally:
        conn.close()

    result = StepMemorySummarySection(window_size=3).build(str(db_path))

    assert result.payload["metadata"]["mode"] == "no_data"
    assert result.payload["diagnosis"]["status"] == "NO GPU"
    assert result.payload["card"].find("Diagnosis: NO GPU") > -1


def test_step_memory_loader_uses_latest_common_steps_per_global_rank(tmp_path):
    db_path = tmp_path / "memory.db"
    _create_step_memory_db(str(db_path))

    conn = sqlite3.connect(str(db_path))
    try:
        rows = [
            (
                4,
                1,
                1,
                0,
                2,
                1,
                0,
                "worker-0",
                2.0,
                1,
                10,
                "cuda:0",
                2,
                210.0,
                310.0,
            ),
            (
                5,
                1,
                1,
                0,
                2,
                1,
                0,
                "worker-0",
                3.0,
                2,
                10,
                "cuda:0",
                3,
                220.0,
                320.0,
            ),
            (
                6,
                1,
                1,
                0,
                2,
                1,
                0,
                "worker-0",
                4.0,
                3,
                10,
                "cuda:0",
                4,
                230.0,
                330.0,
            ),
        ]
        conn.executemany(
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
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    data = load_step_memory_section_data(str(db_path), window_size=2)

    assert data.aligned_window.steps == (2, 3)
    assert data.aligned_window.global_ranks_seen == 2
    assert data.aligned_window.global_ranks_used == 2
    assert data.per_global_rank["0"].metrics["peak_allocated_bytes"] == 115.0
    assert data.per_global_rank["1"].metrics["peak_allocated_bytes"] == 215.0


def test_step_memory_summary_wrapper_delegates_to_section_path(tmp_path):
    db_path = tmp_path / "memory.db"
    _create_step_memory_db(str(db_path))

    summary = generate_step_memory_summary_card(
        str(db_path),
        window_size=3,
        print_to_stdout=False,
    )

    assert summary["metadata"]["global_ranks_seen"] == 1
    assert summary["global"]["window"]["steps_analyzed"] == 3
    assert "steps_used" not in summary["metadata"]
    assert (tmp_path / "memory.db_summary_card.json").exists()
    assert (tmp_path / "memory.db_summary_card.txt").exists()
