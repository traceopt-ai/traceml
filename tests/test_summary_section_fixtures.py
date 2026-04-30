"""SQLite fixture coverage for final-report summary sections.

These tests keep section assertions schema-oriented instead of snapshotting full
cards. The goal is to make contributor changes safe while leaving copy/layout
free to evolve intentionally.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from traceml.aggregator.sqlite_writers import (
    process as process_projection,
    step_memory as step_memory_projection,
    step_time as step_time_projection,
    system as system_projection,
)
from traceml.reporting.final import build_summary_payload
from traceml.reporting.sections.process import ProcessSummarySection
from traceml.reporting.sections.step_memory import StepMemorySummarySection
from traceml.reporting.sections.step_time import StepTimeSummarySection
from traceml.reporting.sections.system import SystemSummarySection


def _connect_with_summary_schema(db_path: Path) -> sqlite3.Connection:
    """Create a tiny TraceML SQLite fixture with all summary tables."""
    conn = sqlite3.connect(db_path)
    system_projection.init_schema(conn)
    process_projection.init_schema(conn)
    step_time_projection.init_schema(conn)
    step_memory_projection.init_schema(conn)
    return conn


def _insert_system_sample(
    conn: sqlite3.Connection,
    *,
    row_id: int,
    rank: int,
    ts: float,
    gpu_available: bool,
    gpu_count: int,
    gpu_util: float | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO system_samples(
            recv_ts_ns,
            rank,
            sample_ts_s,
            seq,
            cpu_percent,
            ram_used_bytes,
            ram_total_bytes,
            gpu_available,
            gpu_count,
            gpu_util_avg,
            gpu_util_peak,
            gpu_mem_used_avg_bytes,
            gpu_mem_used_peak_bytes,
            gpu_temp_avg_c,
            gpu_temp_peak_c,
            gpu_power_avg_w,
            gpu_power_peak_w
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row_id,
            rank,
            ts,
            row_id,
            30.0 + rank,
            4_000.0 + rank,
            16_000.0,
            int(gpu_available),
            gpu_count,
            gpu_util,
            gpu_util,
            2_000.0 if gpu_available else None,
            2_500.0 if gpu_available else None,
            None,
            None,
            None,
            None,
        ),
    )
    if gpu_available and gpu_count > 0:
        conn.execute(
            """
            INSERT INTO system_gpu_samples(
                recv_ts_ns,
                rank,
                sample_ts_s,
                seq,
                gpu_idx,
                util,
                mem_used_bytes,
                mem_total_bytes,
                temperature_c,
                power_usage_w,
                power_limit_w
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                row_id,
                rank,
                ts,
                row_id,
                rank,
                gpu_util,
                2_500.0,
                10_000.0,
                None,
                None,
                None,
            ),
        )


def _insert_process_sample(
    conn: sqlite3.Connection,
    *,
    row_id: int,
    rank: int,
    ts: float,
    gpu_available: bool,
    gpu_count: int,
) -> None:
    conn.execute(
        """
        INSERT INTO process_samples(
            recv_ts_ns,
            rank,
            sample_ts_s,
            seq,
            pid,
            cpu_percent,
            cpu_logical_core_count,
            ram_used_bytes,
            ram_total_bytes,
            gpu_available,
            gpu_count,
            gpu_device_index,
            gpu_mem_used_bytes,
            gpu_mem_reserved_bytes,
            gpu_mem_total_bytes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row_id,
            rank,
            ts,
            row_id,
            10_000 + rank,
            50.0 + rank,
            8,
            1_000.0 + rank * 100.0,
            16_000.0,
            int(gpu_available),
            gpu_count,
            rank if gpu_available else None,
            2_000.0 + rank * 500.0 if gpu_available else None,
            2_500.0 + rank * 600.0 if gpu_available else None,
            10_000.0 if gpu_available else None,
        ),
    )


def _step_time_events(
    *,
    dataloader: float,
    forward: float,
    backward: float,
    optimizer: float,
    step_time: float,
) -> str:
    events = {
        "_traceml_internal:dataloader_next": {
            "cpu": {"is_gpu": False, "duration_ms": dataloader, "n_calls": 1}
        },
        "_traceml_internal:forward_time": {
            "cpu": {"is_gpu": False, "duration_ms": forward, "n_calls": 1}
        },
        "_traceml_internal:backward_time": {
            "cpu": {"is_gpu": False, "duration_ms": backward, "n_calls": 1}
        },
        "_traceml_internal:optimizer_step": {
            "cpu": {"is_gpu": False, "duration_ms": optimizer, "n_calls": 1}
        },
        "_traceml_internal:step_time": {
            "cpu": {"is_gpu": False, "duration_ms": step_time, "n_calls": 1}
        },
    }
    return json.dumps(events)


def _insert_step_time_sample(
    conn: sqlite3.Connection,
    *,
    row_id: int,
    rank: int,
    step: int,
    step_time: float,
) -> None:
    conn.execute(
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
        (
            row_id,
            rank,
            float(step),
            row_id,
            step,
            _step_time_events(
                dataloader=1.0,
                forward=2.0 + rank,
                backward=3.0 + rank,
                optimizer=1.0,
                step_time=step_time,
            ),
        ),
    )


def _insert_step_memory_sample(
    conn: sqlite3.Connection,
    *,
    row_id: int,
    rank: int,
    step: int,
    alloc: float | None,
    reserved: float | None,
    device: str | None = "cuda:0",
) -> None:
    conn.execute(
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
        (
            row_id,
            rank,
            float(step),
            row_id,
            1,
            device,
            step,
            alloc,
            reserved,
        ),
    )


def test_summary_sections_handle_empty_tables_with_stable_schema(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "empty.db"
    conn = _connect_with_summary_schema(db_path)
    conn.close()

    sections = (
        SystemSummarySection(),
        ProcessSummarySection(),
        StepTimeSummarySection(max_rows=4),
        StepMemorySummarySection(window_size=4),
    )

    results = {
        section.name: section.build(str(db_path)) for section in sections
    }

    assert results["system"].payload["overview"]["samples"] == 0
    assert results["process"].payload["overview"]["samples"] == 0
    assert results["step_time"].payload["overview"]["mode"] == "no_data"
    assert results["step_memory"].payload["overview"]["steps_used"] == 0
    for name, result in results.items():
        assert result.section == name
        assert "card" in result.payload
        assert "primary_diagnosis" in result.payload
        assert isinstance(result.payload["issues"], list)
        assert result.text == result.payload["card"]


def test_summary_sections_cover_single_rank_gpu_run(tmp_path: Path) -> None:
    db_path = tmp_path / "single_rank.db"
    conn = _connect_with_summary_schema(db_path)
    try:
        _insert_system_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=True,
            gpu_count=1,
            gpu_util=70.0,
        )
        _insert_process_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=True,
            gpu_count=1,
        )
        for step in range(1, 5):
            _insert_step_time_sample(
                conn,
                row_id=step,
                rank=0,
                step=step,
                step_time=10.0,
            )
            _insert_step_memory_sample(
                conn,
                row_id=step,
                rank=0,
                step=step,
                alloc=100.0 + step,
                reserved=200.0 + step,
            )
        conn.commit()
    finally:
        conn.close()

    system = SystemSummarySection().build(str(db_path)).payload
    process = ProcessSummarySection().build(str(db_path)).payload
    step_time = StepTimeSummarySection(max_rows=4).build(str(db_path)).payload
    step_memory = (
        StepMemorySummarySection(window_size=4).build(str(db_path)).payload
    )

    assert system["overview"]["gpu_available"] is True
    assert system["overview"]["gpu_count"] == 1
    assert process["overview"]["ranks_seen"] == 1
    assert process["per_rank"]["0"]["pid_count"] == 1.0
    assert step_time["overview"]["mode"] == "single_rank"
    assert step_time["overview"]["ranks_seen"] == 1
    assert step_time["global"]["typical"]["steps_analyzed"] == 4
    assert step_memory["overview"]["ranks_seen"] == 1
    assert step_memory["overview"]["steps_used"] == 4
    assert set(step_memory["global"]["metric_rollup"]) == {
        "peak_allocated",
        "peak_reserved",
    }


def test_summary_sections_cover_multi_rank_aligned_run(tmp_path: Path) -> None:
    db_path = tmp_path / "multi_rank.db"
    conn = _connect_with_summary_schema(db_path)
    try:
        for rank in (0, 1):
            _insert_system_sample(
                conn,
                row_id=10 + rank,
                rank=rank,
                ts=1.0 + rank,
                gpu_available=True,
                gpu_count=2,
                gpu_util=80.0 - rank * 20.0,
            )
            _insert_process_sample(
                conn,
                row_id=10 + rank,
                rank=rank,
                ts=1.0 + rank,
                gpu_available=True,
                gpu_count=2,
            )
            for step in range(1, 6):
                row_id = rank * 100 + step
                _insert_step_time_sample(
                    conn,
                    row_id=row_id,
                    rank=rank,
                    step=step,
                    step_time=10.0 + rank,
                )
                _insert_step_memory_sample(
                    conn,
                    row_id=row_id,
                    rank=rank,
                    step=step,
                    alloc=100.0 + rank * 20.0 + step,
                    reserved=200.0 + rank * 30.0 + step,
                    device=f"cuda:{rank}",
                )
        conn.commit()
    finally:
        conn.close()

    step_time = StepTimeSummarySection(max_rows=5).build(str(db_path)).payload
    step_memory = (
        StepMemorySummarySection(window_size=5).build(str(db_path)).payload
    )
    process = ProcessSummarySection().build(str(db_path)).payload

    assert step_time["overview"]["mode"] == "distributed"
    assert step_time["overview"]["ranks_seen"] == 2
    assert step_time["overview"]["steps_analyzed_min"] == 5
    assert set(step_time["per_rank"]) == {"0", "1"}
    assert step_memory["overview"]["ranks_seen"] == 2
    assert step_memory["overview"]["steps_used"] == 5
    assert step_memory["global"]["analysis_window"]["ranks_seen"] == 2
    assert set(step_memory["per_rank"]) == {"0", "1"}
    assert process["overview"]["ranks_seen"] == 2
    assert set(process["per_rank"]) == {"0", "1"}


def test_step_memory_section_reports_no_gpu_without_throwing(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "no_gpu.db"
    conn = _connect_with_summary_schema(db_path)
    try:
        _insert_system_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=False,
            gpu_count=0,
            gpu_util=None,
        )
        _insert_process_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=False,
            gpu_count=0,
        )
        _insert_step_memory_sample(
            conn,
            row_id=1,
            rank=0,
            step=1,
            alloc=None,
            reserved=None,
            device=None,
        )
        conn.commit()
    finally:
        conn.close()

    payload = (
        StepMemorySummarySection(window_size=4).build(str(db_path)).payload
    )

    assert payload["overview"]["training_steps"] == 2
    assert payload["primary_diagnosis"]["status"] == "NO GPU"
    assert payload["global"]["analysis_window"]["steps_used"] == 0
    assert payload["issues"] == []


def test_final_summary_fixture_schema_contains_all_sections(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "final.db"
    conn = _connect_with_summary_schema(db_path)
    try:
        _insert_system_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=False,
            gpu_count=0,
            gpu_util=None,
        )
        _insert_process_sample(
            conn,
            row_id=1,
            rank=0,
            ts=1.0,
            gpu_available=False,
            gpu_count=0,
        )
        conn.commit()
    finally:
        conn.close()

    payload = build_summary_payload(str(db_path))

    assert payload["schema_version"] == 1.2
    assert set(payload) == {
        "schema_version",
        "generated_at",
        "duration_s",
        "system",
        "process",
        "step_time",
        "step_memory",
        "text",
    }
    for key in ("system", "process", "step_time", "step_memory"):
        assert "overview" in payload[key]
        assert "card" in payload[key]
        assert "primary_diagnosis" in payload[key]
