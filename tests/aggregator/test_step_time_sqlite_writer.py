from traceml_ai.aggregator.sqlite_writers.step_time import _normalize_events


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
