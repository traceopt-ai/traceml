import sqlite3

from traceml_ai.aggregator.sqlite_writers.step_time import (
    _normalize_events,
    build_rows,
    init_schema,
    insert_rows,
)
from traceml_ai.telemetry.envelope import normalize_telemetry_envelope


def test_step_time_normalize_events_preserves_cpu_and_gpu_ms() -> None:
    events = {
        "_test_event": {
            "cuda:0": {
                "is_gpu": True,
                "duration_ms": 9.0,
                "cpu_ms": 3.0,
                "gpu_ms": 9.0,
                "n_calls": 2,
                "extra": "drop me",
            },
            "cpu": {
                "is_gpu": False,
                "duration_ms": 4.0,
                "cpu_ms": 4.0,
                "gpu_ms": None,
                "n_calls": 1,
            },
        }
    }

    normalized = _normalize_events(events)

    assert normalized == {
        "_test_event": {
            "cuda:0": {
                "is_gpu": True,
                "duration_ms": 9.0,
                "cpu_ms": 3.0,
                "gpu_ms": 9.0,
                "n_calls": 2,
            },
            "cpu": {
                "is_gpu": False,
                "duration_ms": 4.0,
                "cpu_ms": 4.0,
                "gpu_ms": None,
                "n_calls": 1,
            },
        }
    }


def test_step_time_normalize_events_keeps_old_rows_valid() -> None:
    normalized = _normalize_events(
        {
            "_old_event": {
                "cpu": {
                    "is_gpu": False,
                    "duration_ms": 4.0,
                    "n_calls": 1,
                }
            }
        }
    )

    assert normalized["_old_event"]["cpu"] == {
        "is_gpu": False,
        "duration_ms": 4.0,
        "cpu_ms": None,
        "gpu_ms": None,
        "n_calls": 1,
    }


def test_step_time_schema_indexes_set_based_rank_step_selection() -> None:
    with sqlite3.connect(":memory:") as conn:
        init_schema(conn)
        indexes = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA index_list(step_time_samples);"
            ).fetchall()
        }
        columns = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(step_time_samples);")
        }

    assert "idx_step_time_samples_global_rank_step_id" in indexes
    assert len(indexes) == 4
    assert columns["timestamp_unix_ns"] == "INTEGER"


def test_step_time_projection_accepts_only_canonical_source_time():
    timestamp_unix_ns = 1_700_000_000_000_000_001
    inputs = [
        {"timestamp_unix_ns": timestamp_unix_ns},
        {"timestamp": 1.25},
        {"timestamp_unix_ns": None},
        {"timestamp_unix_ns": True},
        {"timestamp_unix_ns": 1.5},
        {"timestamp_unix_ns": 2**63},
        {},
    ]
    envelope = normalize_telemetry_envelope(
        {
            "meta": {"sampler": "StepTimeSampler"},
            "body": {"tables": {"StepTimeTable": inputs}},
        }
    )
    rows = build_rows(envelope, observed_timestamp_unix_ns=1)
    with sqlite3.connect(":memory:") as conn:
        init_schema(conn)
        insert_rows(conn, rows)
        timestamps = conn.execute(
            "SELECT timestamp_unix_ns FROM step_time_samples ORDER BY id;"
        ).fetchall()
    assert timestamps == [
        (timestamp_unix_ns,),
        *[(None,)] * (len(inputs) - 1),
    ]
