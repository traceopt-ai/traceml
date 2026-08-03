import sqlite3

from traceml_ai.aggregator.sqlite_writers.step_time import (
    _normalize_events,
    init_schema,
)


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

    assert "idx_step_time_samples_global_rank_step_id" in indexes
