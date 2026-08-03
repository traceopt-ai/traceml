import json
import sqlite3

from traceml_ai.diagnostics.step_time import (
    SUMMARY_STEP_TIME_POLICY,
    diagnose_step_time_window,
)
from traceml_ai.reporting.summaries.step_time import (
    generate_step_time_summary_card,
)
from traceml_ai.reporting.sections.step_time import StepTimeSummarySection
from traceml_ai.reporting.sections.step_time.builder import (
    project_step_time_summary,
)
from traceml_ai.step_time.model import (
    StepTimeLoadRequest,
    StepTimeRepositorySnapshot,
    StepTimeSourceCursor,
)
from traceml_ai.step_time.pipeline import StepTimeAnalysis
from traceml_ai.step_time.sqlite import (
    load_training_strategy_from_sqlite,
)
from tests.step_time.factories import (
    rank_average,
    window_from_events,
)


def _create_step_time_db(path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE step_time_samples (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                recv_ts_ns         INTEGER NOT NULL,
                rank               INTEGER,
                global_rank        INTEGER,
                local_rank         INTEGER,
                world_size         INTEGER,
                local_world_size   INTEGER,
                node_rank          INTEGER,
                hostname           TEXT,
                sample_ts_s        REAL,
                seq                INTEGER,
                step               INTEGER,
                events_json        TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE runtime_environment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_strategy TEXT
            );
            """
        )
        conn.executemany(
            """
            INSERT INTO runtime_environment(training_strategy)
            VALUES (?);
            """,
            [("ddp",), ("fsdp",)],
        )
        events = {
            "_traceml_internal:dataloader_next": {
                "cpu": {
                    "is_gpu": False,
                    "duration_ms": 1.0,
                    "cpu_ms": 1.0,
                    "gpu_ms": None,
                    "n_calls": 1,
                }
            },
            "_traceml_internal:forward_time": {
                "cpu": {
                    "is_gpu": False,
                    "duration_ms": 5.0,
                    "cpu_ms": 5.0,
                    "gpu_ms": None,
                    "n_calls": 1,
                }
            },
            "_traceml_internal:backward_time": {
                "cpu": {
                    "is_gpu": False,
                    "duration_ms": 10.0,
                    "cpu_ms": 10.0,
                    "gpu_ms": None,
                    "n_calls": 1,
                }
            },
            "_traceml_internal:optimizer_step": {
                "cpu": {
                    "is_gpu": False,
                    "duration_ms": 4.0,
                    "cpu_ms": 4.0,
                    "gpu_ms": None,
                    "n_calls": 1,
                }
            },
            "_traceml_internal:step_time": {
                "cpu": {
                    "is_gpu": False,
                    "duration_ms": 30.0,
                    "cpu_ms": 30.0,
                    "gpu_ms": None,
                    "n_calls": 1,
                }
            },
        }
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
                1,
                json.dumps(events),
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
                2,
                json.dumps(events),
            ),
        ]
        conn.executemany(
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
                sample_ts_s,
                seq,
                step,
                events_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def test_step_time_summary_uses_persisted_events_json(tmp_path) -> None:
    db_path = tmp_path / "telemetry"
    _create_step_time_db(str(db_path))

    summary = generate_step_time_summary_card(
        str(db_path),
        print_to_stdout=False,
    )

    assert summary["metadata"]["global_ranks_seen"] == 1
    assert summary["global"]["window"]["steps_analyzed"] == 2
    assert summary["global"]["median"]["total_step_ms"]["value"] == 31.0
    assert "Global: n/a" not in summary["card"]


def test_training_strategy_loader_uses_latest_available_row(tmp_path) -> None:
    db_path = tmp_path / "telemetry"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE runtime_environment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_strategy TEXT
            );
            """
        )
        conn.executemany(
            "INSERT INTO runtime_environment(training_strategy) VALUES (?);",
            [("fsdp",), ("ddp",)],
        )
        assert load_training_strategy_from_sqlite(conn) == "ddp"
    finally:
        conn.close()


def test_rank_summary_extracts_input_bound_clocks_from_events() -> None:
    window = window_from_events(
        {
            0: {
                1: {
                    "_traceml_internal:dataloader_next": {
                        "cuda:0": {
                            "is_gpu": False,
                            "duration_ms": 12.0,
                            "cpu_ms": 12.0,
                            "gpu_ms": 4.0,
                            "n_calls": 1,
                        }
                    },
                    "_traceml_internal:step_time": {
                        "cuda:0": {
                            "is_gpu": False,
                            "duration_ms": 60.0,
                            "cpu_ms": 60.0,
                            "gpu_ms": 20.0,
                            "n_calls": 1,
                        }
                    },
                }
            }
        },
        max_rows=1,
        expected_ranks=[0],
    )

    assert window.clock == "gpu"
    rank_facts = window.rank(0)
    assert rank_facts is not None
    assert rank_facts.steps[0].step == 1
    values = rank_facts.steps[0].values
    assert values.input_wait_ms == 4.0
    assert values.step_time_ms == 20.0
    assert rank_average(window, 0).input_wait_ms == 4.0
    assert rank_average(window, 0).step_time_ms == 20.0


def test_step_time_section_uses_summary_pipeline_and_sqlite_fixture(
    tmp_path,
    monkeypatch,
) -> None:
    from traceml_ai.reporting.sections import step_time as section_module

    db_path = tmp_path / "telemetry"
    _create_step_time_db(str(db_path))

    calls = []
    original_run = section_module.StepTimePipeline.run

    def capture_run(pipeline, request):
        calls.append((pipeline.profile, request.window_size))
        return original_run(pipeline, request)

    monkeypatch.setattr(section_module.StepTimePipeline, "run", capture_run)
    result = StepTimeSummarySection().build(str(db_path))

    assert calls == [("summary", StepTimeSummarySection().max_rows)]
    assert result.section == "step_time"
    assert result.payload["metadata"]["training_total_steps"] == 3
    assert result.payload["metadata"]["training_latest_step"] == 2
    assert result.payload["metadata"]["global_ranks_seen"] == 1
    assert result.payload["global"]["median"]["total_step_ms"]["value"] == 31.0
    assert result.payload["groups"]["rows"]["0"]["identity"] == {
        "global_rank": 0,
        "local_rank": 0,
        "node_rank": 0,
        "hostname": "worker-0",
        "local_world_size": 1,
        "world_size": 1,
    }
    assert "TraceML Step Timing Summary" in result.text


def test_distributed_step_time_scope_shows_actual_analyzed_steps() -> None:
    def event_stats(cpu_ms: float) -> dict[str, dict[str, float | bool | int]]:
        return {
            "cpu": {
                "is_gpu": False,
                "duration_ms": cpu_ms,
                "cpu_ms": cpu_ms,
                "gpu_ms": None,
                "n_calls": 1,
            }
        }

    per_rank_steps = {
        rank: {
            step: {
                "_traceml_internal:dataloader_next": event_stats(1.0),
                "_traceml_internal:h2d_time": event_stats(0.0),
                "_traceml_internal:forward_time": event_stats(2.0),
                "_traceml_internal:backward_time": event_stats(3.0),
                "_traceml_internal:optimizer_step": event_stats(1.0),
                "_traceml_internal:step_time": event_stats(8.0),
            }
            for step in range(1, 129)
        }
        for rank in range(4)
    }
    window = window_from_events(
        per_rank_steps,
        max_rows=10000,
        expected_ranks=range(4),
    )
    analysis = StepTimeAnalysis(
        request=StepTimeLoadRequest(window_size=10000),
        snapshot=StepTimeRepositorySnapshot(
            cursor=StepTimeSourceCursor(latest_step=128),
        ),
        window=window,
        diagnosis=diagnose_step_time_window(
            window,
            policy=SUMMARY_STEP_TIME_POLICY,
        ),
    )
    summary = project_step_time_summary(analysis)
    card = summary["card"]

    assert "compared over last 128 aligned steps across 4 global ranks" in card
    assert "10000 steps" not in card
    assert summary["global"]["window"]["steps_analyzed"] == 128
    assert summary["global"]["window"]["window_size"] == 10000
    assert "aligned_steps_analyzed" not in summary["metadata"]
    assert "steps_analyzed_min_per_global_rank" not in summary["metadata"]
    assert "steps_analyzed_max_per_global_rank" not in summary["metadata"]
