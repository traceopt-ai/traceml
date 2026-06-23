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
        assert writer.force_flush(timeout_sec=2.0)
    finally:
        writer.finalize(timeout_sec=2.0)


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


def test_sqlite_writer_finalize_drains_multiple_batches(tmp_path) -> None:
    db_path = tmp_path / "telemetry.db"
    writer = SQLiteWriterSimple(
        SQLiteWriterConfig(
            path=str(db_path),
            flush_interval_sec=60.0,
            max_flush_items=1,
        )
    )
    writer.start()
    try:
        for _ in range(3):
            writer.ingest(_system_payload())
        result = writer.finalize(timeout_sec=2.0)
    finally:
        writer.finalize(timeout_sec=2.0)

    assert result.ok
    assert result.checkpoint_ok
    assert result.queue_size == 0

    conn = sqlite3.connect(db_path)
    try:
        projected_count = conn.execute(
            "SELECT COUNT(*) FROM system_samples;"
        ).fetchone()[0]
    finally:
        conn.close()

    assert projected_count == 3
    assert not db_path.with_name(db_path.name + "-wal").exists()
    assert not db_path.with_name(db_path.name + "-shm").exists()


def test_sqlite_writer_force_flush_keeps_writer_open(tmp_path) -> None:
    db_path = tmp_path / "telemetry.db"
    writer = SQLiteWriterSimple(
        SQLiteWriterConfig(path=str(db_path), flush_interval_sec=60.0)
    )
    writer.start()
    try:
        writer.ingest(_system_payload())
        assert writer.force_flush(timeout_sec=2.0)
        assert not writer.stats()["last_error"]
        writer.ingest(_system_payload())
        result = writer.finalize(timeout_sec=2.0)
    finally:
        writer.finalize(timeout_sec=2.0)

    assert result.ok

    conn = sqlite3.connect(db_path)
    try:
        projected_count = conn.execute(
            "SELECT COUNT(*) FROM system_samples;"
        ).fetchone()[0]
    finally:
        conn.close()

    assert projected_count == 2


def test_sqlite_writer_finalize_timeout_reports_failure(monkeypatch, tmp_path):
    db_path = tmp_path / "telemetry.db"
    writer = SQLiteWriterSimple(SQLiteWriterConfig(path=str(db_path)))
    writer._started = True
    monkeypatch.setattr(writer._closed, "wait", lambda timeout: False)

    result = writer.finalize(timeout_sec=0.01)

    assert not result.ok
    assert not result.checkpoint_ok
    assert "Timed out" in str(result.error)


def test_sqlite_writer_final_prune_failure_is_nonfatal(
    monkeypatch,
    tmp_path,
) -> None:
    db_path = tmp_path / "telemetry.db"
    writer = SQLiteWriterSimple(
        SQLiteWriterConfig(path=str(db_path), flush_interval_sec=60.0)
    )
    monkeypatch.setattr(
        writer,
        "_prune_all_retained_rows",
        lambda conn: (_ for _ in ()).throw(RuntimeError("prune boom")),
    )
    writer.start()
    try:
        writer.ingest(_system_payload())
        result = writer.finalize(timeout_sec=2.0)
    finally:
        writer.finalize(timeout_sec=2.0)

    assert result.ok
    assert result.checkpoint_ok
    assert not result.prune_ok
    assert "prune boom" in str(result.prune_error)
    assert result.error is None
    assert result.checkpoint_error is None


def test_sqlite_writer_finalize_preserves_prune_and_checkpoint_errors(
    tmp_path,
) -> None:
    writer = SQLiteWriterSimple(
        SQLiteWriterConfig(path=str(tmp_path / "telemetry.db"))
    )

    result = writer._build_finalize_result(
        elapsed_sec=1.0,
        checkpoint_ok=False,
        error="SQLite checkpoint/close failed: checkpoint boom",
        prune_error="Final SQLite retention prune failed: prune boom",
        checkpoint_error="SQLite checkpoint/close failed: checkpoint boom",
    )

    assert not result.ok
    assert not result.checkpoint_ok
    assert not result.prune_ok
    assert "checkpoint boom" in str(result.error)
    assert "checkpoint boom" in str(result.checkpoint_error)
    assert "prune boom" in str(result.prune_error)


def test_sqlite_writer_checkpoint_failure_is_fatal(tmp_path) -> None:
    writer = SQLiteWriterSimple(
        SQLiteWriterConfig(path=str(tmp_path / "telemetry.db"))
    )

    result = writer._build_finalize_result(
        elapsed_sec=1.0,
        checkpoint_ok=False,
        error=None,
        checkpoint_error="SQLite checkpoint/close failed: checkpoint boom",
    )

    assert not result.ok
    assert not result.checkpoint_ok
    assert result.prune_ok
    assert "checkpoint boom" in str(result.error)
