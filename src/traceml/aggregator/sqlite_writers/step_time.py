"""
SQLite projection writer for StepTimeSampler.

This module projects TraceML StepTimeSampler payloads into a query-friendly
SQLite table while preserving the original sampler payload in `raw_messages`.

Design
------
- Keeps sampler-specific SQL logic out of the core SQLite writer
- Accepts already-decoded payload dicts from the main writer
- Produces one query-friendly table:
    1) step_time_samples
       One row per (rank, step) with stable metadata columns plus a restricted
       dynamic payload column (`events_json`)

Why a restricted payload column?
--------------------------------
Step-time event names are not stable enough to justify fixed SQL columns.
Therefore, the SQL schema keeps only stable fields in first-class columns:
- rank
- step
- sample_ts_s
- seq

and stores the dynamic event map as JSON text in `events_json`.

Expected payload shape
----------------------
Envelope:
{
    "rank": int,
    "sampler": "StepTimeSampler",
    "timestamp": float,
    "tables": {
        "<table_name>": [
            {
                "seq": int,
                "timestamp": float,
                "step": int,
                "events": {
                    "<event_name>": {
                        "<device>": {
                            "is_gpu": bool,
                            "duration_ms": float,
                            "n_calls": int
                        }
                    }
                }
            },
            ...
        ]
    }
}
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Optional

SAMPLER_NAME = "StepTimeSampler"


def accepts_sampler(sampler: Optional[str]) -> bool:
    """Return True if this projection writer handles the given sampler."""
    return sampler == SAMPLER_NAME


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Create query-friendly projection table for StepTimeSampler.

    Table
    -----
    step_time_samples
        One row per step-aligned timing sample. Dynamic event content is stored
        in `events_json`, while stable fields remain queryable in dedicated
        columns.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS step_time_samples (
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
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_step_time_samples_rank_step_ts
        ON step_time_samples(rank, step, sample_ts_s, id);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_step_time_samples_step_rank
        ON step_time_samples(step, rank, id);
        """
    )


def _normalize_events(events_raw: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Normalize the dynamic events payload into a JSON-safe restricted structure.

    Output shape
    ------------
    {
        "<event_name>": {
            "<device>": {
                "is_gpu": bool | None,
                "duration_ms": float | None,
                "n_calls": int | None
            }
        }
    }

    Notes
    -----
    - Unknown extra fields are intentionally dropped.
    - Keys are stringified for safety.
    - Malformed entries are skipped best-effort.
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}

    if not isinstance(events_raw, dict):
        return out

    for event_name, by_device in events_raw.items():
        if not isinstance(by_device, dict):
            continue

        event_key = str(event_name)
        out[event_key] = {}

        for device, stats in by_device.items():
            if not isinstance(stats, dict):
                continue

            device_key = str(device)

            is_gpu_raw = stats.get("is_gpu")
            duration_raw = stats.get("duration_ms")
            n_calls_raw = stats.get("n_calls")

            is_gpu = bool(is_gpu_raw) if isinstance(is_gpu_raw, bool) else None
            duration_ms = (
                float(duration_raw)
                if isinstance(duration_raw, (int, float))
                else None
            )
            n_calls = (
                int(n_calls_raw) if isinstance(n_calls_raw, int) else None
            )

            out[event_key][device_key] = {
                "is_gpu": is_gpu,
                "duration_ms": duration_ms,
                "n_calls": n_calls,
            }

        if not out[event_key]:
            out.pop(event_key, None)

    return out


def build_rows(
    payload_dict: Dict[str, Any],
    recv_ts_ns: int,
) -> Dict[str, list[tuple]]:
    """
    Build SQLite projection rows from one decoded StepTimeSampler payload.

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
            "step_time_samples": [...],
        }

    Notes
    -----
    - Returns empty lists if payload is malformed or belongs to another sampler.
    - Keeps projection logic best-effort and non-throwing.
    - Assumes payload rows follow StepTimeEventSample.to_wire().
    """
    out: Dict[str, list[tuple]] = {
        "step_time_samples": [],
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
            ts_raw = row.get("timestamp")
            step_raw = row.get("step")
            events_raw = row.get("events")

            seq = int(seq_raw) if isinstance(seq_raw, int) else None
            sample_ts_s = (
                float(ts_raw) if isinstance(ts_raw, (int, float)) else None
            )
            step = int(step_raw) if isinstance(step_raw, int) else None

            restricted_events = _normalize_events(events_raw)
            events_json = json.dumps(
                restricted_events,
                separators=(",", ":"),
                sort_keys=True,
            )

            out["step_time_samples"].append(
                (
                    recv_ts_ns,
                    rank,
                    sample_ts_s,
                    seq,
                    step,
                    events_json,
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
    rows = rows_by_table.get("step_time_samples", [])
    if rows:
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
