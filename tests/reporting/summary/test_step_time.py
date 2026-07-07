import json
import sqlite3

from traceml_ai.diagnostics.step_time.adapters import (
    StepTimeDiagnosisInput,
    diagnose_step_time_summary,
)
from traceml_ai.reporting.summaries.step_time import (
    generate_step_time_summary_card,
)
from traceml_ai.reporting.sections.step_time import StepTimeSummarySection
from traceml_ai.reporting.sections.step_time.loader import (
    StepTimeSectionData,
    load_step_time_section_data,
)
from traceml_ai.reporting.sections.step_time.model import (
    rank_summaries_from_window,
)
from traceml_ai.utils.step_time_window import (
    INPUT_WAIT_CPU_MS_KEY,
    STEP_TIME_CPU_MS_KEY,
    build_step_time_window_from_events,
    build_step_time_window_from_step_metrics,
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


def test_rank_summary_extracts_input_bound_clocks_from_events() -> None:
    window = build_step_time_window_from_events(
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
    metrics = window.per_rank_step_timing[0][1]
    assert metrics["input_wait"] == 4.0
    assert metrics["step_time"] == 20.0
    assert window.per_rank_timing[0]["input_wait"] == 4.0
    assert window.per_rank_timing[0]["step_time"] == 20.0


def test_step_time_section_loader_and_builder_use_sqlite_fixture(
    tmp_path,
) -> None:
    db_path = tmp_path / "telemetry"
    _create_step_time_db(str(db_path))

    data = load_step_time_section_data(str(db_path))
    result = StepTimeSummarySection().build(str(db_path))

    assert data.training_steps == 3
    assert data.latest_step_observed == 2
    assert data.per_global_rank_summary[0].steps_analyzed == 2
    assert data.aligned_window.coverage.steps_used == 2
    assert result.section == "step_time"
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
    from traceml_ai.reporting.sections.step_time.builder import (
        build_step_time_payload,
    )

    per_global_rank_step_metrics = {
        rank: {
            step: {
                INPUT_WAIT_CPU_MS_KEY: 1.0,
                "h2d_cpu_ms": 0.0,
                "forward_cpu_ms": 2.0,
                "backward_cpu_ms": 3.0,
                "optimizer_step_cpu_ms": 1.0,
                STEP_TIME_CPU_MS_KEY: 8.0,
            }
            for step in range(1, 129)
        }
        for rank in range(4)
    }
    window = build_step_time_window_from_step_metrics(
        per_global_rank_step_metrics,
        max_rows=10000,
        expected_ranks=range(4),
    )
    per_global_rank = rank_summaries_from_window(window)

    data = StepTimeSectionData(
        training_steps=129,
        latest_step_observed=128,
        step_time_window=window,
        aligned_summary=per_global_rank,
        aligned_step_metrics=window.per_rank_step_timing,
        aligned_window=window,
        per_global_rank_summary=per_global_rank,
        per_global_rank_step_metrics=window.per_rank_step_timing,
        identities={},
        max_rows=10000,
    )
    diagnosis = diagnose_step_time_summary(
        StepTimeDiagnosisInput(
            window=window,
        )
    )
    summary = build_step_time_payload(data, diagnosis)
    card = summary["card"]

    assert "compared over last 128 aligned steps across 4 global ranks" in card
    assert "10000 steps" not in card
    assert summary["global"]["window"]["steps_analyzed"] == 128
    assert summary["global"]["window"]["window_size"] == 10000
    assert "aligned_steps_analyzed" not in summary["metadata"]
    assert "steps_analyzed_min_per_global_rank" not in summary["metadata"]
    assert "steps_analyzed_max_per_global_rank" not in summary["metadata"]
