"""
SQLite projection writer for StdoutStderrSampler.

This module projects TraceML stdout/stderr sampler payloads into a simple,
query-friendly SQLite table while preserving the original sampler payload in
`raw_messages`.

Design
------
- Keeps sampler-specific SQL logic out of the core SQLite writer
- Accepts already-decoded payload dicts from the main writer
- Produces one append-only table:
    1) stdout_stderr_samples
       One row per captured output line

Storage model
-------------
Stdout/stderr is intentionally stored as a very simple append-only stream:
- one row per line
- rank retained for filtering
- receive timestamp retained for stable ordering
- sampler timestamp retained when available

Expected payload shape
----------------------
Envelope:
{
    "rank": int,
    "sampler": "Stdout/Stderr",
    "timestamp": float,
    "tables": {
        "stdout_stderr": [
            {"line": "..."},
            ...
        ]
    }
}
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Optional

SAMPLER_NAME = "Stdout/Stderr"


def accepts_sampler(sampler: Optional[str]) -> bool:
    """
    Return True if this projection writer handles the given sampler.
    """
    return sampler == SAMPLER_NAME


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Create the query-friendly projection table for stdout/stderr lines.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stdout_stderr_samples (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            recv_ts_ns   INTEGER NOT NULL,
            rank         INTEGER,
            sample_ts_s  REAL,
            line         TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_stdout_stderr_rank_id
        ON stdout_stderr_samples(rank, id);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_stdout_stderr_sample_ts
        ON stdout_stderr_samples(sample_ts_s, id);
        """
    )


def build_rows(
    payload_dict: Dict[str, Any],
    recv_ts_ns: int,
) -> Dict[str, list[tuple]]:
    """
    Build SQLite projection rows from one decoded StdoutStderrSampler payload.

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
            "stdout_stderr_samples": [...],
        }

    Notes
    -----
    - Returns empty lists if payload is malformed or belongs to another sampler.
    - Keeps projection logic best-effort and non-throwing.
    """
    out: Dict[str, list[tuple]] = {
        "stdout_stderr_samples": [],
    }

    sampler = payload_dict.get("sampler")
    if not accepts_sampler(str(sampler) if sampler is not None else None):
        return out

    rank_raw = payload_dict.get("rank")
    try:
        rank = int(rank_raw) if rank_raw is not None else None
    except Exception:
        rank = None

    ts_raw = payload_dict.get("timestamp")
    sample_ts_s = float(ts_raw) if isinstance(ts_raw, (int, float)) else None

    tables = payload_dict.get("tables")
    if not isinstance(tables, dict):
        return out

    for rows in tables.values():
        if not isinstance(rows, list):
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue

            line_raw = row.get("line")
            if line_raw is None:
                continue

            line = str(line_raw)
            if not line:
                continue

            out["stdout_stderr_samples"].append(
                (
                    recv_ts_ns,
                    rank,
                    sample_ts_s,
                    line,
                )
            )

    return out


def insert_rows(
    conn: sqlite3.Connection, rows_by_table: Dict[str, list[tuple]]
) -> None:
    """
    Insert projection rows into SQLite.
    """
    rows = rows_by_table.get("stdout_stderr_samples", [])
    if rows:
        conn.executemany(
            """
            INSERT INTO stdout_stderr_samples(
                recv_ts_ns,
                rank,
                sample_ts_s,
                line
            )
            VALUES (?, ?, ?, ?);
            """,
            rows,
        )
