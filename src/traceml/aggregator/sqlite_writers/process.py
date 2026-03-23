"""
SQLite projection writer for ProcessSampler.

This module projects TraceML ProcessSampler payloads into SQLite
table while preserving the original sampler payload in `raw_messages`.

Design
------
- Keeps sampler-specific SQL logic out of the core SQLite writer
- Accepts already-decoded payload dicts from the main writer
- Produces one query-friendly table:
    1) process_samples
       One row per sampled process snapshot, keyed by rank and sample time

Storage units
-------------
This projection stores raw values wherever possible.

- cpu_percent             : percent
- ram_used_bytes          : bytes
- ram_total_bytes         : bytes
- gpu_mem_used_bytes      : bytes
- gpu_mem_reserved_bytes  : bytes
- gpu_mem_total_bytes     : bytes


Expected payload shape
----------------------
Envelope:
{
    "rank": int,
    "sampler": "ProcessSampler",
    "timestamp": float,
    "tables": {
        "<table_name>": [
            {
                "seq": int,
                "timestamp": float,
                "pid": int,
                "cpu_percent": float,
                "cpu_logical_core_count": int,
                "ram_used": float,   # bytes
                "ram_total": float,  # bytes
                "gpu_available": bool,
                "gpu_count": int,
                "gpu": {
                    "device_index": int,
                    "mem_used": float,      # bytes
                    "mem_reserved": float,  # bytes
                    "mem_total": float      # bytes
                } | None
            },
            ...
        ]
    }
}
"""

import sqlite3
from typing import Any, Dict, Optional

SAMPLER_NAME = "ProcessSampler"


def accepts_sampler(sampler: Optional[str]) -> bool:
    """Return True if this projection writer handles the given sampler."""
    return sampler == SAMPLER_NAME


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Create query-friendly projection table for ProcessSampler.

    Table
    -----
    process_samples
        One row per sampled process snapshot. Includes process CPU/RAM fields,
        rank metadata, and flattened single-device GPU memory metrics when
        available.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS process_samples (
            id                       INTEGER PRIMARY KEY AUTOINCREMENT,
            recv_ts_ns               INTEGER NOT NULL,
            rank                     INTEGER,
            sample_ts_s              REAL,
            seq                      INTEGER,
            pid                      INTEGER,
            cpu_percent              REAL,
            cpu_logical_core_count   INTEGER,
            ram_used_bytes           REAL,
            ram_total_bytes          REAL,
            gpu_available            INTEGER,
            gpu_count                INTEGER,
            gpu_device_index         INTEGER,
            gpu_mem_used_bytes       REAL,
            gpu_mem_reserved_bytes   REAL,
            gpu_mem_total_bytes      REAL
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_process_samples_rank_ts
        ON process_samples(rank, sample_ts_s, id);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_process_samples_pid_ts
        ON process_samples(pid, sample_ts_s, id);
        """
    )


def build_rows(
    payload_dict: Dict[str, Any],
    recv_ts_ns: int,
) -> Dict[str, list[tuple]]:
    """
    Build SQLite projection rows from one decoded ProcessSampler payload.

    Parameters
    ----------
    payload_dict:
        Decoded sampler payload dict from the main SQLite writer.
    recv_ts_ns:
        Receive timestamp assigned by the main writer for this payload.

    Returns
    -------
    dict[str, list[tuple]]
        {
            "process_samples": [...],
        }

    Notes
    -----
    - Returns empty lists if payload is malformed or belongs to another sampler.
    - Keeps projection logic best-effort and non-throwing.
    - Assumes payload rows follow ProcessSample.to_wire().
    """
    out: Dict[str, list[tuple]] = {
        "process_samples": [],
    }

    sampler = payload_dict.get("sampler")
    if not accepts_sampler(str(sampler) if sampler is not None else None):
        return out

    rank_raw = payload_dict.get("rank")
    try:
        rank = int(rank_raw) if rank_raw is not None else None
    except Exception:
        rank = None

    tables = payload_dict.get("tables")
    if not isinstance(tables, dict):
        return out

    for rows in tables.values():
        if not isinstance(rows, list):
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue

            seq_raw = row.get("seq")
            ts_raw = row.get("ts")
            pid_raw = row.get("pid")
            cpu_percent_raw = row.get("cpu")
            cpu_logical_core_count_raw = row.get("cpu_cores")
            ram_used_raw = row.get("ram_used")
            ram_total_raw = row.get("ram_total")
            gpu_available_raw = row.get("gpu_available")
            gpu_count_raw = row.get("gpu_count")
            gpu_raw = row.get("gpu")

            seq = int(seq_raw) if isinstance(seq_raw, int) else None
            sample_ts_s = (
                float(ts_raw) if isinstance(ts_raw, (int, float)) else None
            )
            pid = int(pid_raw) if isinstance(pid_raw, int) else None
            cpu_percent = (
                float(cpu_percent_raw)
                if isinstance(cpu_percent_raw, (int, float))
                else None
            )
            cpu_logical_core_count = (
                int(cpu_logical_core_count_raw)
                if isinstance(cpu_logical_core_count_raw, int)
                else None
            )
            ram_used_bytes = (
                float(ram_used_raw)
                if isinstance(ram_used_raw, (int, float))
                else None
            )
            ram_total_bytes = (
                float(ram_total_raw)
                if isinstance(ram_total_raw, (int, float))
                else None
            )
            gpu_available = (
                1
                if gpu_available_raw is True
                else 0 if gpu_available_raw is False else None
            )
            gpu_count = (
                int(gpu_count_raw) if isinstance(gpu_count_raw, int) else None
            )

            gpu_device_index = None
            gpu_mem_used_bytes = None
            gpu_mem_reserved_bytes = None
            gpu_mem_total_bytes = None

            if isinstance(gpu_raw, dict):
                device_index_raw = gpu_raw.get("device")
                mem_used_raw = gpu_raw.get("mem_used")
                mem_reserved_raw = gpu_raw.get("mem_reserved")
                mem_total_raw = gpu_raw.get("mem_total")

                gpu_device_index = (
                    int(device_index_raw)
                    if isinstance(device_index_raw, int)
                    else None
                )
                gpu_mem_used_bytes = (
                    float(mem_used_raw)
                    if isinstance(mem_used_raw, (int, float))
                    else None
                )
                gpu_mem_reserved_bytes = (
                    float(mem_reserved_raw)
                    if isinstance(mem_reserved_raw, (int, float))
                    else None
                )
                gpu_mem_total_bytes = (
                    float(mem_total_raw)
                    if isinstance(mem_total_raw, (int, float))
                    else None
                )

            out["process_samples"].append(
                (
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
                    gpu_mem_total_bytes,
                )
            )

    return out


def insert_rows(
    conn: sqlite3.Connection, rows_by_table: Dict[str, list[tuple]]
) -> None:
    """
    Insert projection rows into SQLite.

    Parameters
    ----------
    conn:
        SQLite connection owned by the main writer thread.
    rows_by_table:
        Output from `build_rows()`.
    """
    rows = rows_by_table.get("process_samples", [])
    if rows:
        conn.executemany(
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
            rows,
        )
