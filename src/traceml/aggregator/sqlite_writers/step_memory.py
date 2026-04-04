"""
SQLite projection writer for StepMemorySampler.

This module projects TraceML StepMemorySampler payloads into a query-friendly
SQLite table while preserving the original sampler payload in `raw_messages`.

Design
------
- Keeps sampler-specific SQL logic out of the core SQLite writer.
- Accepts already-decoded payload dicts from the main writer.
- Produces one query-friendly table:
    1) step_memory_samples
       One row per (rank, step memory event) with stable metadata and memory
       bytes fields.

Expected payload shape
----------------------
Envelope:
{
    "rank": int,
    "sampler": "StepMemorySampler",
    "timestamp": float,
    "tables": {
        "<table_name>": [
            {
                "seq": int,
                "ts": float,
                "model_id": int | null,
                "device": str | null,
                "step": int | null,
                "peak_alloc": float | null,   # bytes
                "peak_resv": float | null     # bytes
            },
            ...
        ]
    }
}
"""

import sqlite3
from typing import Any, Dict, Optional

SAMPLER_NAME = "StepMemorySampler"


def accepts_sampler(sampler: Optional[str]) -> bool:
    """Return True if this projection writer handles the given sampler."""
    return sampler == SAMPLER_NAME


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Create query-friendly projection table for StepMemorySampler.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS step_memory_samples (
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
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_step_memory_samples_rank_step_ts
        ON step_memory_samples(rank, step, sample_ts_s, id);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_step_memory_samples_step_rank
        ON step_memory_samples(step, rank, id);
        """
    )


def build_rows(
    payload_dict: Dict[str, Any],
    recv_ts_ns: int,
) -> Dict[str, list[tuple]]:
    """
    Build SQLite projection rows from one decoded StepMemorySampler payload.

    Notes
    -----
    - Returns empty lists if payload is malformed or belongs to another sampler.
    - Keeps projection logic best-effort and non-throwing.
    """
    out: Dict[str, list[tuple]] = {
        "step_memory_samples": [],
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
            model_id_raw = row.get("model_id")
            device_raw = row.get("device")
            step_raw = row.get("step")
            peak_alloc_raw = row.get("peak_alloc")
            peak_resv_raw = row.get("peak_resv")

            seq = int(seq_raw) if isinstance(seq_raw, int) else None
            sample_ts_s = (
                float(ts_raw) if isinstance(ts_raw, (int, float)) else None
            )
            model_id = (
                int(model_id_raw) if isinstance(model_id_raw, int) else None
            )
            device = str(device_raw) if isinstance(device_raw, str) else None
            step = int(step_raw) if isinstance(step_raw, int) else None
            peak_alloc_bytes = (
                float(peak_alloc_raw)
                if isinstance(peak_alloc_raw, (int, float))
                else None
            )
            peak_reserved_bytes = (
                float(peak_resv_raw)
                if isinstance(peak_resv_raw, (int, float))
                else None
            )

            out["step_memory_samples"].append(
                (
                    recv_ts_ns,
                    rank,
                    sample_ts_s,
                    seq,
                    model_id,
                    device,
                    step,
                    peak_alloc_bytes,
                    peak_reserved_bytes,
                )
            )

    return out


def insert_rows(
    conn: sqlite3.Connection, rows_by_table: Dict[str, list[tuple]]
) -> None:
    """
    Insert projection rows into SQLite.
    """
    rows = rows_by_table.get("step_memory_samples", [])
    if rows:
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
