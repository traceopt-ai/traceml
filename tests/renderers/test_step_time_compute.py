import json
import sqlite3

from traceml_ai.renderers.step_time.compute import StepCombinedComputer


def test_step_time_compute_uses_selected_gpu_diagnosis_clock(
    tmp_path,
) -> None:
    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE step_time_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rank INTEGER,
                step INTEGER,
                events_json TEXT NOT NULL
            );
            """
        )
        events_1 = {
            "_traceml_internal:dataloader_next": {
                "cuda:0": {
                    "duration_ms": 12.0,
                    "cpu_ms": 12.0,
                    "gpu_ms": 4.0,
                    "is_gpu": False,
                    "n_calls": 1,
                }
            },
            "_traceml_internal:step_time": {
                "cuda:0": {
                    "duration_ms": 60.0,
                    "cpu_ms": 60.0,
                    "gpu_ms": 20.0,
                    "is_gpu": False,
                    "n_calls": 1,
                }
            },
        }
        events_2 = {
            "_traceml_internal:dataloader_next": {
                "cuda:0": {
                    "duration_ms": 18.0,
                    "cpu_ms": 18.0,
                    "gpu_ms": 6.0,
                    "is_gpu": False,
                    "n_calls": 1,
                }
            },
            "_traceml_internal:step_time": {
                "cuda:0": {
                    "duration_ms": 90.0,
                    "cpu_ms": 90.0,
                    "gpu_ms": 40.0,
                    "is_gpu": False,
                    "n_calls": 1,
                }
            },
        }
        conn.execute(
            """
            INSERT INTO step_time_samples(rank, step, events_json)
            VALUES (?, ?, ?);
            """,
            (0, 1, json.dumps(events_1)),
        )
        conn.execute(
            """
            INSERT INTO step_time_samples(rank, step, events_json)
            VALUES (?, ?, ?);
            """,
            (0, 2, json.dumps(events_2)),
        )
        conn.commit()
    finally:
        conn.close()

    result = StepCombinedComputer(
        db_path=str(db_path),
        window_size=2,
    ).compute_cli()

    metrics = {metric.metric: metric for metric in result.diagnosis_metrics}
    assert "dataloader_fetch" not in metrics
    assert metrics["input_wait"].summary.worst_total == 5.0
    assert metrics["step_time"].summary.worst_total == 30.0
    assert metrics["input_wait"].series is not None
    assert metrics["input_wait"].series.worst == [4.0, 6.0]
    assert metrics["input_wait"].series.sum == [4.0, 6.0]
    assert result.diagnosis_clock == "gpu"

    per_rank = result.per_rank_timing[0]
    assert per_rank["input_wait"] == 5.0
    assert per_rank["step_time"] == 30.0
