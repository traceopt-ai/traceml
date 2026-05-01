import sqlite3

from traceml.reporting.sections.process import ProcessSummarySection
from traceml.reporting.sections.process.loader import load_process_section_data
from traceml.reporting.sections.system import SystemSummarySection
from traceml.reporting.sections.system.loader import load_system_section_data
from traceml.reporting.summaries.process import generate_process_summary_card
from traceml.reporting.summaries.system import generate_system_summary_card


def _create_system_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE system_samples (
            id INTEGER PRIMARY KEY,
            rank INTEGER,
            seq INTEGER,
            sample_ts_s REAL,
            cpu_percent REAL,
            ram_used_bytes REAL,
            ram_total_bytes REAL,
            gpu_available INTEGER,
            gpu_count INTEGER,
            gpu_util_avg REAL,
            gpu_util_peak REAL,
            gpu_mem_used_avg_bytes REAL,
            gpu_mem_used_peak_bytes REAL,
            gpu_temp_avg_c REAL,
            gpu_temp_peak_c REAL,
            gpu_power_avg_w REAL,
            gpu_power_peak_w REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE system_gpu_samples (
            id INTEGER PRIMARY KEY,
            rank INTEGER,
            seq INTEGER,
            gpu_idx INTEGER,
            util REAL,
            mem_used_bytes REAL,
            mem_total_bytes REAL,
            temperature_c REAL,
            power_usage_w REAL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO system_samples VALUES (
            1, 0, 1, 10.0, 40.0, 8.0, 16.0, 1, 1,
            55.0, 70.0, 4.0, 5.0, 60.0, 68.0, 100.0, 120.0
        )
        """
    )
    conn.execute(
        """
        INSERT INTO system_gpu_samples VALUES (
            1, 0, 1, 0, 70.0, 5.0, 10.0, 68.0, 120.0
        )
        """
    )


def _create_process_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE process_samples (
            id INTEGER PRIMARY KEY,
            rank INTEGER,
            pid INTEGER,
            sample_ts_s REAL,
            cpu_percent REAL,
            cpu_logical_core_count INTEGER,
            ram_used_bytes REAL,
            ram_total_bytes REAL,
            gpu_available INTEGER,
            gpu_count INTEGER,
            gpu_device_index INTEGER,
            gpu_mem_used_bytes REAL,
            gpu_mem_reserved_bytes REAL,
            gpu_mem_total_bytes REAL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO process_samples VALUES (
            1, 0, 123, 10.0, 80.0, 8, 4000000000.0, 16000000000.0,
            1, 1, 0, 5000000000.0, 6000000000.0, 10000000000.0
        )
        """
    )


def test_system_section_loader_and_builder_use_sqlite_fixture(tmp_path):
    db_path = tmp_path / "system.db"
    conn = sqlite3.connect(db_path)
    try:
        _create_system_tables(conn)
        conn.commit()
    finally:
        conn.close()

    data = load_system_section_data(str(db_path))
    result = SystemSummarySection().build(str(db_path))

    assert data.aggregate.system_samples == 1
    assert data.per_gpu[0].util_peak_percent == 70.0
    assert result.section == "system"
    assert result.payload["overview"]["samples"] == 1
    assert "TraceML System Summary" in result.text
    assert "- Diagnosis: NORMAL" in result.text
    assert (
        "- Stats: CPU avg 40% | RAM peak 50% | "
        "GPU util avg 55% | GPU memory peak 50%"
    ) in result.text
    assert "GPU:" not in result.text
    assert result.payload["global"]["cpu"]["avg_band"] == "normal"
    assert result.payload["global"]["ram"]["peak_band"] == "normal"
    assert result.payload["global"]["gpu_rollup"]["util_avg_band"] == "normal"
    assert result.payload["global"]["gpu_rollup"]["mem_peak_band"] == "normal"
    assert result.payload["per_gpu"]["0"]["mem_peak_percent"] == 50.0
    assert "- Issues:" not in result.text


def test_process_section_loader_and_builder_use_sqlite_fixture(tmp_path):
    db_path = tmp_path / "process.db"
    conn = sqlite3.connect(db_path)
    try:
        _create_process_tables(conn)
        conn.commit()
    finally:
        conn.close()

    data = load_process_section_data(str(db_path))
    result = ProcessSummarySection().build(str(db_path))

    assert data.aggregate.process_samples == 1
    assert data.per_rank[0].pid_count == 1
    assert result.section == "process"
    assert result.payload["overview"]["samples"] == 1
    assert "TraceML Process Summary" in result.text
    assert "- Diagnosis: NORMAL" in result.text
    assert (
        "- Stats: ranks 1 | pids 1 | CPU avg 80% | "
        "RSS peak 4.0 / 16.0 GB | GPU reserved peak 60%"
    ) in result.text
    assert "- Takeaway:" not in result.text
    assert "- Issues:" not in result.text
    assert result.payload["global"]["cpu"]["capacity_band"] == "low"
    assert result.payload["global"]["ram"]["peak_band"] == "low"
    assert (
        result.payload["global"]["gpu_rollup"]["reserved_peak_band"]
        == "normal"
    )
    assert "takeaway" not in result.payload["global"]


def test_legacy_summary_wrappers_delegate_to_section_paths(tmp_path):
    db_path = tmp_path / "combined.db"
    conn = sqlite3.connect(db_path)
    try:
        _create_system_tables(conn)
        _create_process_tables(conn)
        conn.commit()
    finally:
        conn.close()

    system = generate_system_summary_card(str(db_path), print_to_stdout=False)
    process = generate_process_summary_card(
        str(db_path), print_to_stdout=False
    )

    assert system["overview"]["samples"] == 1
    assert process["overview"]["samples"] == 1
    assert (tmp_path / "combined.db_summary_card.json").exists()
    assert (tmp_path / "combined.db_summary_card.txt").exists()
