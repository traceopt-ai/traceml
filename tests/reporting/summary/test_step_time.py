import json
import sqlite3

from traceml_ai.diagnostics.step_time.adapters import (
    StepTimeDiagnosisInput,
    diagnose_step_time_summary,
)
from traceml_ai.reporting.sections.step_time.alignment import AlignedStepWindow
from traceml_ai.reporting.summaries.step_time import (
    generate_step_time_summary_card,
)
from traceml_ai.reporting.sections.step_time import StepTimeSummarySection
from traceml_ai.reporting.sections.step_time.loader import (
    StepTimeSectionData,
    load_step_time_section_data,
)
from traceml_ai.reporting.sections.step_time.model import build_rank_summary
from traceml_ai.reporting.sections.step_time.model import to_rank_signals
from traceml_ai.utils.step_time_input_bound import (
    INPUT_BOUND_CLOCK_IS_GPU_KEY,
    INPUT_BOUND_STEP_MS_KEY,
    INPUT_WAIT_MS_KEY,
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
                    "n_calls": 1,
                }
            },
            "_traceml_internal:forward_time": {
                "cpu": {
                    "is_gpu": False,
                    "duration_ms": 5.0,
                    "n_calls": 1,
                }
            },
            "_traceml_internal:backward_time": {
                "cpu": {
                    "is_gpu": False,
                    "duration_ms": 10.0,
                    "n_calls": 1,
                }
            },
            "_traceml_internal:optimizer_step": {
                "cpu": {
                    "is_gpu": False,
                    "duration_ms": 4.0,
                    "n_calls": 1,
                }
            },
            "_traceml_internal:step_time": {
                "cpu": {
                    "is_gpu": False,
                    "duration_ms": 30.0,
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
    analysis = build_rank_summary(
        [
            {
                "step": 1,
                "events": {
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
                },
            }
        ]
    )

    assert analysis is not None
    metrics = analysis.per_step_metrics[1]
    assert metrics["dataloader_fetch"] == 12.0
    assert metrics["step_time"] == 60.0
    assert metrics[INPUT_WAIT_MS_KEY] == 4.0
    assert metrics[INPUT_BOUND_STEP_MS_KEY] == 20.0
    assert metrics[INPUT_BOUND_CLOCK_IS_GPU_KEY] == 1.0


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
    assert data.aligned_window.steps_analyzed == 2
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
    from traceml_ai.reporting.summaries.step_time import RankStepSummary

    per_global_rank = {
        rank: RankStepSummary(
            steps_analyzed=128,
            avg_dataloader_ms=1.0,
            avg_h2d_ms=0.0,
            avg_forward_ms=2.0,
            avg_backward_ms=3.0,
            avg_optimizer_ms=1.0,
            avg_step_cpu_ms=8.0,
            avg_traced_step_ms=8.0,
            avg_gpu_compute_ms=6.0,
            avg_total_step_ms=9.0,
        )
        for rank in range(4)
    }

    data = StepTimeSectionData(
        training_steps=129,
        latest_step_observed=128,
        aligned_summary=per_global_rank,
        aligned_step_metrics={},
        aligned_window=AlignedStepWindow(
            alignment="common_steps",
            steps_analyzed=128,
            start_step=None,
            end_step=None,
            window_size=10000,
            global_ranks_used=4,
            global_ranks_observed=4,
        ),
        per_global_rank_summary=per_global_rank,
        per_global_rank_step_metrics={},
        identities={},
        max_rows=10000,
    )
    diagnosis = diagnose_step_time_summary(
        StepTimeDiagnosisInput(
            rank_signals=to_rank_signals(per_global_rank),
            per_rank_step_metrics={},
            max_rows=10000,
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
