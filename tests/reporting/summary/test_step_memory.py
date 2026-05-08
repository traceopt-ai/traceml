import sqlite3

from traceml.diagnostics.step_memory import SUMMARY_STEP_MEMORY_POLICY
from traceml.reporting.sections.step_memory import StepMemorySummarySection
from traceml.reporting.sections.step_memory.loader import (
    load_step_memory_section_data,
)
from traceml.reporting.summaries.step_memory import (
    generate_step_memory_summary_card,
)


def _create_step_memory_db(path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE step_memory_samples (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                recv_ts_ns           INTEGER NOT NULL,
                rank                 INTEGER,
                sample_ts_s          REAL,
                seq                  INTEGER,
                model_id             INTEGER,
                device               TEXT,
                step                 INTEGER,
                peak_alloc_bytes     REAL,
                peak_reserved_bytes  REAL
            );
            """
        )
        rows = [
            (1, 0, 1.0, 1, 10, "cuda:0", 1, 100.0, 200.0),
            (2, 0, 2.0, 2, 10, "cuda:0", 2, 110.0, 210.0),
            (3, 0, 3.0, 3, 10, "cuda:0", 3, 120.0, 220.0),
        ]
        conn.executemany(
            """
            INSERT INTO step_memory_samples(
                recv_ts_ns,
                rank,
                sample_ts_s,
                seq,
                model_id,
                device,
                step,
                peak_alloc_bytes,
                peak_reserved_bytes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def test_step_memory_section_loader_and_builder_use_sqlite_fixture(tmp_path):
    db_path = tmp_path / "memory.db"
    _create_step_memory_db(str(db_path))

    data = load_step_memory_section_data(str(db_path), window_size=3)
    result = StepMemorySummarySection(window_size=3).build(str(db_path))

    assert data.training_steps == 4
    assert data.latest_step_observed == 3
    assert [metric.metric for metric in data.metrics] == [
        "peak_allocated",
        "peak_reserved",
    ]
    assert result.section == "step_memory"
    assert result.payload["overview"]["ranks_seen"] == 1
    assert result.payload["overview"]["steps_used"] == 3
    assert "TraceML Step Memory Summary" in result.text
    assert "- Diagnosis:" in result.text
    assert "- Scope:" in result.text
    assert "- Stats:" in result.text
    assert "- Why:" in result.text
    assert result.text.index("- Diagnosis:") < result.text.index("- Scope:")
    assert result.text.index("- Scope:") < result.text.index("- Stats:")
    assert result.text.index("- Stats:") < result.text.index("- Why:")
    assert "- Primary:" not in result.text
    assert "- Trend:" not in result.text
    assert "- Note:" not in result.text
    assert "- Issues:" not in result.text


def test_step_memory_section_loader_uses_summary_policy(
    tmp_path,
    monkeypatch,
):
    import traceml.reporting.sections.step_memory.loader as loader_module

    db_path = tmp_path / "memory.db"
    _create_step_memory_db(str(db_path))
    captured = {}

    def fake_primary(metrics, *, thresholds, **kwargs):
        captured["primary_thresholds"] = thresholds
        return None

    def fake_result(metrics, *, thresholds, **kwargs):
        captured["result_thresholds"] = thresholds
        return None

    monkeypatch.setattr(
        loader_module,
        "build_step_memory_diagnosis",
        fake_primary,
    )
    monkeypatch.setattr(
        loader_module,
        "build_step_memory_summary_diagnosis_result",
        fake_result,
    )

    load_step_memory_section_data(str(db_path), window_size=3)

    assert (
        captured["primary_thresholds"] is SUMMARY_STEP_MEMORY_POLICY.thresholds
    )
    assert (
        captured["result_thresholds"] is SUMMARY_STEP_MEMORY_POLICY.thresholds
    )


def test_step_memory_legacy_wrapper_delegates_to_section_path(tmp_path):
    db_path = tmp_path / "memory.db"
    _create_step_memory_db(str(db_path))

    summary = generate_step_memory_summary_card(
        str(db_path),
        window_size=3,
        print_to_stdout=False,
    )

    assert summary["overview"]["ranks_seen"] == 1
    assert summary["overview"]["steps_used"] == 3
    assert (tmp_path / "memory.db_summary_card.json").exists()
    assert (tmp_path / "memory.db_summary_card.txt").exists()
