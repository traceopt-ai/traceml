# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3

from traceml_ai.aggregator.sqlite_writer import (
    SQLiteWriterConfig,
    SQLiteWriterSimple,
)
from traceml_ai.aggregator.sqlite_writers import system as system_projection
from traceml_ai.telemetry.envelope import normalize_telemetry_envelope


def _system_payload() -> dict:
    return {
        "global_rank": 5,
        "local_rank": 1,
        "world_size": 8,
        "local_world_size": 4,
        "node_rank": 1,
        "hostname": "worker-1",
        "sampler": "SystemSampler",
        "timestamp": 123.0,
        "tables": {
            "SystemTable": [
                {
                    "seq": 7,
                    "ts": 123.0,
                    "cpu": 42.0,
                    "ram_used": 4_000.0,
                    "ram_total": 16_000.0,
                    "gpu_available": True,
                    "gpu_count": 1,
                    "gpus": [[70.0, 5_000.0, 10_000.0, 68.0, 120.0, 250.0]],
                }
            ]
        },
    }


def test_system_projection_stores_global_and_local_rank_identity() -> None:
    conn = sqlite3.connect(":memory:")
    system_projection.init_schema(conn)

    envelope = normalize_telemetry_envelope(_system_payload())
    assert envelope is not None
    rows = system_projection.build_rows(envelope, recv_ts_ns=999)
    system_projection.insert_rows(conn, rows)

    sample = conn.execute(
        """
        SELECT global_rank, local_rank, world_size, local_world_size,
               node_rank, hostname, seq
        FROM system_samples;
        """
    ).fetchone()
    gpu_sample = conn.execute(
        """
        SELECT global_rank, local_rank, world_size, local_world_size,
               node_rank, hostname, gpu_idx
        FROM system_gpu_samples;
        """
    ).fetchone()

    assert sample == (5, 1, 8, 4, 1, "worker-1", 7)
    assert gpu_sample == (5, 1, 8, 4, 1, "worker-1", 0)


def _sqlite_table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table';"
        )
    }


def _write_payload_through_sqlite_writer(db_path) -> None:
    writer = SQLiteWriterSimple(
        SQLiteWriterConfig(path=str(db_path), flush_interval_sec=0.01)
    )
    writer.start()
    try:
        writer.ingest(_system_payload())
        assert writer.flush_now(timeout_sec=2.0)
    finally:
        writer.stop()


def test_sqlite_writer_persists_projection_without_raw_table(tmp_path) -> None:
    db_path = tmp_path / "telemetry.db"
    _write_payload_through_sqlite_writer(db_path)

    conn = sqlite3.connect(db_path)
    try:
        assert "raw_messages" not in _sqlite_table_names(conn)
        sample = conn.execute(
            """
            SELECT global_rank, local_rank, world_size, local_world_size,
                   node_rank, hostname, seq
            FROM system_samples;
            """
        ).fetchone()
        gpu_sample = conn.execute(
            """
            SELECT global_rank, local_rank, world_size, local_world_size,
                   node_rank, hostname, gpu_idx
            FROM system_gpu_samples;
            """
        ).fetchone()
    finally:
        conn.close()

    assert sample == (5, 1, 8, 4, 1, "worker-1", 7)
    assert gpu_sample == (5, 1, 8, 4, 1, "worker-1", 0)


def test_sqlite_writer_leaves_existing_raw_table_untouched(tmp_path) -> None:
    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE raw_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                payload_mp BLOB NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO raw_messages(payload_mp) VALUES (?);",
            (b"old-raw-payload",),
        )
        conn.commit()
    finally:
        conn.close()

    _write_payload_through_sqlite_writer(db_path)

    conn = sqlite3.connect(db_path)
    try:
        assert "raw_messages" in _sqlite_table_names(conn)
        raw_count = conn.execute(
            "SELECT COUNT(*) FROM raw_messages;"
        ).fetchone()[0]
        projected_count = conn.execute(
            "SELECT COUNT(*) FROM system_samples;"
        ).fetchone()[0]
    finally:
        conn.close()

    assert raw_count == 1
    assert projected_count == 1
