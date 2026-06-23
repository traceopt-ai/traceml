# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
SQLite projection writer for StepTimeSampler.

This module projects TraceML StepTimeSampler payloads into a query-friendly
SQLite table.

Design
------
- Keeps sampler-specific SQL logic out of the core SQLite writer
- Accepts canonical telemetry envelopes from the main writer
- Produces one query-friendly table:
    1) step_time_samples
       One row per (global rank, step) with stable metadata columns plus a
       restricted dynamic payload column (`events_json`)

Why a restricted payload column?
--------------------------------
Step-time event names are not stable enough to justify fixed SQL columns.
Therefore, the SQL schema keeps only stable fields in first-class columns:
- rank, retained as a legacy alias for global_rank while live paths migrate
- global_rank, local_rank, world_size, local_world_size, node_rank
- hostname
- step
- sample_ts_s
- seq

and stores the dynamic event map as JSON text in `events_json`.

Expected payload shape
----------------------
Canonical telemetry envelope:
{
    "meta": {"sampler": "StepTimeSampler", ...},
    "body": {
        "tables": {"<table_name>": [StepTimeEventSample.to_wire(), ...]}
    }
}
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

from traceml_ai.telemetry.envelope import TelemetryEnvelope, TelemetryMeta

SAMPLER_NAME = "StepTimeSampler"
RETENTION_TABLES = ("step_time_samples",)


def _optional_int(value: Any) -> Optional[int]:
    """Best-effort integer coercion for telemetry identity fields."""
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _optional_str(value: Any) -> Optional[str]:
    """Best-effort string coercion for telemetry identity fields."""
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


@dataclass(frozen=True)
class StepTimePayloadIdentity:
    """Distributed identity stored with projected step-time telemetry."""

    rank: Optional[int]
    global_rank: Optional[int]
    local_rank: Optional[int]
    world_size: Optional[int]
    local_world_size: Optional[int]
    node_rank: Optional[int]
    hostname: Optional[str]


def _payload_identity(meta: TelemetryMeta) -> StepTimePayloadIdentity:
    """
    Return storage identity for one StepTimeSampler payload.

    ``global_rank`` is the canonical distributed identity. ``rank`` is still
    written as the same value so current live renderers keep working until they
    move to explicit identity fields.
    """
    global_rank = _optional_int(meta.global_rank)
    legacy_rank = _optional_int(meta.rank)
    rank = global_rank if global_rank is not None else legacy_rank

    return StepTimePayloadIdentity(
        rank=rank,
        global_rank=global_rank,
        local_rank=_optional_int(meta.local_rank),
        world_size=_optional_int(meta.world_size),
        local_world_size=_optional_int(meta.local_world_size),
        node_rank=_optional_int(meta.node_rank),
        hostname=_optional_str(meta.hostname),
    )


def _ensure_column(
    conn: sqlite3.Connection,
    *,
    table: str,
    column: str,
    definition: str,
) -> None:
    """Add one nullable projection column when upgrading an existing DB."""
    existing = {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table});").fetchall()
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition};")


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
        in `events_json`; runtime identity and step metadata stay queryable in
        dedicated columns.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS step_time_samples (
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
    _ensure_column(
        conn,
        table="step_time_samples",
        column="global_rank",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="step_time_samples",
        column="local_rank",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="step_time_samples",
        column="world_size",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="step_time_samples",
        column="local_world_size",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="step_time_samples",
        column="node_rank",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="step_time_samples",
        column="hostname",
        definition="TEXT",
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_step_time_samples_rank_step_ts
        ON step_time_samples(rank, step, sample_ts_s, id);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_step_time_samples_global_rank_step_ts
        ON step_time_samples(global_rank, step, sample_ts_s, id);
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
    envelope: TelemetryEnvelope,
    recv_ts_ns: int,
) -> Dict[str, list[tuple]]:
    """
    Build SQLite projection rows from one decoded StepTimeSampler payload.

    Parameters
    ----------
    envelope:
        Canonical telemetry envelope from the main SQLite writer.
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

    sampler = envelope.meta.sampler
    if not accepts_sampler(sampler):
        return out

    identity = _payload_identity(envelope.meta)

    tables = envelope.tables
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
                    identity.rank,
                    identity.global_rank,
                    identity.local_rank,
                    identity.world_size,
                    identity.local_world_size,
                    identity.node_rank,
                    identity.hostname,
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
