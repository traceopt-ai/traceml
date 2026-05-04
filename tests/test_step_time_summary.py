import json
import sqlite3

from traceml.reporting.summaries.step_time import (
    generate_step_time_summary_card,
)
from traceml.reporting.sections.step_time import StepTimeSummarySection
from traceml.reporting.sections.step_time.loader import (
    load_step_time_section_data,
)


def _create_step_time_db(path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE step_time_samples (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                recv_ts_ns    INTEGER NOT NULL,
                rank          INTEGER,
                sample_ts_s   REAL,
                seq           INTEGER,
                step          INTEGER,
                events_json   TEXT NOT NULL
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
            (1, 0, 1.0, 1, 1, json.dumps(events)),
            (2, 0, 2.0, 2, 2, json.dumps(events)),
        ]
        conn.executemany(
            """
            INSERT INTO step_time_samples(
                recv_ts_ns,
                rank,
                sample_ts_s,
                seq,
                step,
                events_json
            )
            VALUES (?, ?, ?, ?, ?, ?);
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

    assert summary["overview"]["ranks_seen"] == 1
    assert summary["global"]["typical"]["steps_analyzed"] == 2
    assert summary["global"]["typical"]["step_avg_ms"] == 31.0
    assert "Global: n/a" not in summary["card"]


def test_step_time_section_loader_and_builder_use_sqlite_fixture(
    tmp_path,
) -> None:
    db_path = tmp_path / "telemetry"
    _create_step_time_db(str(db_path))

    data = load_step_time_section_data(str(db_path))
    result = StepTimeSummarySection().build(str(db_path))

    assert data.training_steps == 3
    assert data.latest_step_observed == 2
    assert data.per_rank_summary[0].steps_analyzed == 2
    assert result.section == "step_time"
    assert result.payload["overview"]["ranks_seen"] == 1
    assert result.payload["global"]["typical"]["step_avg_ms"] == 31.0
    assert "TraceML Step Timing Summary" in result.text


def test_distributed_step_time_scope_shows_actual_analyzed_steps() -> None:
    from traceml.reporting.sections.step_time.builder import (
        build_step_time_card,
    )
    from traceml.reporting.summaries.step_time import RankStepSummary

    per_rank = {
        rank: RankStepSummary(
            steps_analyzed=128,
            avg_dataloader_ms=1.0,
            avg_forward_ms=2.0,
            avg_backward_ms=3.0,
            avg_optimizer_ms=1.0,
            avg_step_cpu_ms=8.0,
            avg_gpu_compute_ms=6.0,
            avg_total_step_ms=9.0,
        )
        for rank in range(4)
    }

    card, summary = build_step_time_card(
        training_steps=129,
        latest_step_observed=128,
        per_rank_summary=per_rank,
        per_rank_step_metrics={},
        max_rows=10000,
    )

    assert "compared over last 128 steps per rank" in card
    assert "10000 steps" not in card
    assert summary["overview"]["steps_analyzed_min"] == 128
    assert summary["overview"]["steps_analyzed_max"] == 128
