# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the set-based Step Time SQLite repository."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.step_time.scenarios import (
    BALANCED_PROFILE,
    StepTimeScenario,
    create_step_time_database,
)
from traceml_ai.step_time.model import StepTimeLoadRequest
from traceml_ai.step_time.sqlite import SQLiteStepTimeRepository
from traceml_ai.utils.step_time_sqlite import (
    load_step_time_summary_from_sqlite,
    load_step_time_window_from_sqlite,
)


def _create_minimal_table(conn: sqlite3.Connection) -> None:
    """Create the smallest schema accepted by the compatibility loader."""
    conn.execute(
        """
        CREATE TABLE step_time_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            global_rank INTEGER,
            step INTEGER,
            events_json TEXT NOT NULL
        );
        """
    )


def test_deduplication_happens_before_distinct_step_limit(
    tmp_path: Path,
) -> None:
    """Repeated latest-step rows cannot crowd older distinct steps out."""
    db_path = tmp_path / "duplicates.db"
    create_step_time_database(
        db_path,
        StepTimeScenario(
            name="duplicates",
            profiles={0: BALANCED_PROFILE, 1: BALANCED_PROFILE},
            steps=(1, 2, 3, 4, 5),
        ),
    )

    with sqlite3.connect(db_path) as conn:
        original = conn.execute(
            """
            SELECT events_json
            FROM step_time_samples
            WHERE global_rank = 0 AND step = 5;
            """
        ).fetchone()[0]
        latest = json.loads(original)
        latest["_traceml_internal:forward_time"]["cpu"]["cpu_ms"] = 99.0
        latest_json = json.dumps(latest)
        conn.executemany(
            """
            INSERT INTO step_time_samples(
                recv_ts_ns, rank, global_rank, step, events_json
            ) VALUES (?, 0, 0, 5, ?);
            """,
            [(100 + index, latest_json) for index in range(12)],
        )

        snapshot = SQLiteStepTimeRepository(conn).load_summary(
            StepTimeLoadRequest(window_size=4)
        )
        loaded = load_step_time_window_from_sqlite(conn, max_rows=4)

    by_rank = {
        rank: sorted(
            row.step for row in snapshot.rows if row.global_rank == rank
        )
        for rank in snapshot.global_ranks
    }
    assert by_rank == {0: [2, 3, 4, 5], 1: [2, 3, 4, 5]}
    assert loaded.window.steps == [2, 3, 4, 5]
    assert loaded.window.per_rank_timing[0]["forward"] == pytest.approx(47.25)


def test_repository_decodes_each_selected_row_once(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "decode-once.db"
    create_step_time_database(
        db_path,
        StepTimeScenario(
            name="decode_once",
            profiles={0: BALANCED_PROFILE, 1: BALANCED_PROFILE},
            steps=(1, 2, 3, 4),
        ),
    )

    with sqlite3.connect(db_path) as conn:
        with patch(
            "traceml_ai.step_time.sqlite.json.loads",
            wraps=json.loads,
        ) as loads:
            snapshot = SQLiteStepTimeRepository(conn).load_live(
                StepTimeLoadRequest(window_size=3)
            )

    assert len(snapshot.rows) == 6
    assert loads.call_count == len(snapshot.rows)
    assert snapshot.cursor.latest_step == 4
    assert snapshot.cursor.last_row_id == 8


def test_live_and_summary_profiles_feed_the_same_analysis(
    tmp_path: Path,
) -> None:
    """Different SQL selection strategies must not create semantic drift."""
    db_path = tmp_path / "profile-parity.db"
    create_step_time_database(
        db_path,
        StepTimeScenario(
            name="profile_parity",
            profiles={0: BALANCED_PROFILE, 1: BALANCED_PROFILE},
            steps=(1, 2, 3, 4, 5, 6),
            training_strategy="fsdp",
        ),
    )

    with sqlite3.connect(db_path) as conn:
        live_source = SQLiteStepTimeRepository(conn).load_live(
            StepTimeLoadRequest(window_size=4, lookback_factor=1)
        )
        summary_source = SQLiteStepTimeRepository(conn).load_summary(
            StepTimeLoadRequest(window_size=4)
        )
        live = load_step_time_window_from_sqlite(
            conn,
            max_rows=4,
            lookback_factor=1,
        )
        summary = load_step_time_summary_from_sqlite(conn, max_rows=4)

    live_rows = {(row.global_rank, row.step): row for row in live_source.rows}
    summary_rows = {
        (row.global_rank, row.step): row for row in summary_source.rows
    }
    assert live_rows == summary_rows
    assert live_source.global_ranks == summary_source.global_ranks
    assert live_source.training_strategy == summary_source.training_strategy
    assert live.window == summary.window


def test_live_selection_cost_is_independent_of_total_run_length(
    tmp_path: Path,
) -> None:
    """Guard the indexed tail scan against whole-table materialization."""

    def measured_batches(stored_steps: int) -> int:
        db_path = tmp_path / f"live-cost-{stored_steps}.db"
        with sqlite3.connect(db_path) as conn:
            _create_minimal_table(conn)
            conn.execute(
                """
                CREATE INDEX step_time_rank_step_id
                ON step_time_samples(global_rank, step DESC, id DESC);
                """
            )
            conn.executemany(
                """
                INSERT INTO step_time_samples(
                    global_rank, step, events_json
                ) VALUES (?, ?, '{}');
                """,
                (
                    (rank, step)
                    for rank in range(2)
                    for step in range(stored_steps)
                ),
            )
            conn.commit()

            batches = 0

            def record_batch() -> int:
                nonlocal batches
                batches += 1
                return 0

            conn.set_progress_handler(record_batch, 100)
            SQLiteStepTimeRepository(conn).load_live(
                StepTimeLoadRequest(window_size=20)
            )
            conn.set_progress_handler(None, 0)
            return batches

    short_run = measured_batches(100)
    long_run = measured_batches(5_000)

    assert long_run <= short_run + 10


def test_repository_flattens_multi_device_dual_clock_values(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "dual-clock.db"
    events = {
        "_traceml_internal:dataloader_next": {
            "cuda:0": {"cpu_ms": 1.5, "gpu_ms": 0.0},
            "cuda:1": {"cpu_ms": 2.5, "gpu_ms": 3.0},
        },
        "_traceml_internal:forward_time": {
            "cuda:0": {"cpu_ms": -4.0, "gpu_ms": None},
        },
    }
    with sqlite3.connect(db_path) as conn:
        _create_minimal_table(conn)
        conn.execute(
            """
            INSERT INTO step_time_samples(global_rank, step, events_json)
            VALUES (0, 1, ?);
            """,
            (json.dumps(events),),
        )
        snapshot = SQLiteStepTimeRepository(conn).load_live(
            StepTimeLoadRequest(window_size=1)
        )

    row = snapshot.rows[0]
    assert row.metrics["input_wait"].cpu_ms == 4.0
    assert row.metrics["input_wait"].gpu_ms == 3.0
    assert row.metrics["forward"].cpu_ms == 0.0
    assert row.metrics["forward"].gpu_ms is None


def test_repository_returns_filtered_progress_identity_and_context(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "metadata.db"
    create_step_time_database(
        db_path,
        StepTimeScenario(
            name="metadata",
            profiles={0: BALANCED_PROFILE, 1: BALANCED_PROFILE},
            steps=(3, 4, 5),
            training_strategy="fsdp",
        ),
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO step_time_samples(
                recv_ts_ns, rank, global_rank, step, events_json
            ) VALUES (99, 0, NULL, 99, '{}');
            """
        )
        snapshot = SQLiteStepTimeRepository(conn).load_summary(
            StepTimeLoadRequest(window_size=2, rank_filter=(1,))
        )

    assert snapshot.global_ranks == (1,)
    assert [row.step for row in snapshot.rows] == [4, 5]
    assert snapshot.cursor.latest_step == 99
    assert snapshot.cursor.last_row_id == 7
    assert snapshot.training_strategy == "fsdp"
    assert snapshot.identities[1].hostname == "worker-0"
    assert snapshot.identities[1].global_rank == 1


def test_empty_rank_filter_reads_no_source_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "empty-filter.db"
    create_step_time_database(
        db_path,
        StepTimeScenario(
            name="empty_filter",
            profiles={0: BALANCED_PROFILE},
            steps=(1,),
        ),
    )

    with sqlite3.connect(db_path) as conn:
        snapshot = SQLiteStepTimeRepository(conn).load_summary(
            StepTimeLoadRequest(window_size=1, rank_filter=())
        )

    assert snapshot.rows == ()
    assert snapshot.global_ranks == ()
    assert snapshot.cursor.latest_step is None
    assert snapshot.cursor.last_row_id is None


def test_minimal_schema_and_malformed_json_remain_fail_open(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "minimal.db"
    with sqlite3.connect(db_path) as conn:
        _create_minimal_table(conn)
        conn.execute(
            """
            INSERT INTO step_time_samples(global_rank, step, events_json)
            VALUES (0, 7, '{not-json');
            """
        )
        snapshot = SQLiteStepTimeRepository(conn).load_summary(
            StepTimeLoadRequest(window_size=2)
        )

    assert snapshot.rows == ()
    assert snapshot.global_ranks == (0,)
    assert snapshot.cursor.latest_step == 7
    assert snapshot.cursor.last_row_id == 1
    assert snapshot.training_strategy == "ddp"
    assert snapshot.identities[0].local_rank is None


def test_invalid_table_identifier_is_rejected() -> None:
    with sqlite3.connect(":memory:") as conn:
        with pytest.raises(ValueError, match="Invalid Step Time table"):
            SQLiteStepTimeRepository(conn, table="samples; DROP TABLE x")


def test_repository_preserves_caller_owned_transaction(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "caller-transaction.db"
    create_step_time_database(
        db_path,
        StepTimeScenario(
            name="caller_transaction",
            profiles={0: BALANCED_PROFILE},
            steps=(1,),
        ),
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute("BEGIN")
        SQLiteStepTimeRepository(conn).load_summary(
            StepTimeLoadRequest(window_size=1)
        )
        assert conn.in_transaction is True
        conn.rollback()


def test_repository_closes_owned_transaction_after_success_and_failure(
    tmp_path: Path,
) -> None:
    valid_path = tmp_path / "owned-success.db"
    create_step_time_database(
        valid_path,
        StepTimeScenario(
            name="owned_success",
            profiles={0: BALANCED_PROFILE},
            steps=(1,),
        ),
    )
    with sqlite3.connect(valid_path) as conn:
        SQLiteStepTimeRepository(conn).load_summary(
            StepTimeLoadRequest(window_size=1)
        )
        assert conn.in_transaction is False

    invalid_path = tmp_path / "owned-failure.db"
    with sqlite3.connect(invalid_path) as conn:
        conn.execute("CREATE TABLE step_time_samples(id INTEGER PRIMARY KEY);")
        conn.commit()
        with pytest.raises(ValueError, match="is missing"):
            SQLiteStepTimeRepository(conn).load_summary(
                StepTimeLoadRequest(window_size=1)
            )
        assert conn.in_transaction is False


def test_related_reads_share_one_sqlite_snapshot(tmp_path: Path) -> None:
    """A strategy commit between statements appears only on the next load."""
    db_path = tmp_path / "snapshot.db"
    create_step_time_database(
        db_path,
        StepTimeScenario(
            name="snapshot",
            profiles={0: BALANCED_PROFILE},
            steps=(1,),
            training_strategy="ddp",
        ),
    )

    reader = sqlite3.connect(db_path)
    writer = sqlite3.connect(db_path)
    try:
        reader.execute("PRAGMA journal_mode=WAL;")
        writer.execute("PRAGMA journal_mode=WAL;")
        inserted = False

        def insert_before_context_read(statement: str) -> None:
            nonlocal inserted
            if inserted or "FROM runtime_environment" not in statement:
                return
            writer.execute(
                """
                INSERT INTO runtime_environment(training_strategy)
                VALUES ('fsdp');
                """
            )
            writer.commit()
            inserted = True

        reader.set_trace_callback(insert_before_context_read)
        first = SQLiteStepTimeRepository(reader).load_summary(
            StepTimeLoadRequest(window_size=1)
        )
        reader.set_trace_callback(None)
        second = SQLiteStepTimeRepository(reader).load_summary(
            StepTimeLoadRequest(window_size=1)
        )
    finally:
        reader.close()
        writer.close()

    assert inserted is True
    assert first.training_strategy == "ddp"
    assert second.training_strategy == "fsdp"
