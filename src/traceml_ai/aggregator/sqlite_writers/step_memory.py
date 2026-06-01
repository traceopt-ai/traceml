# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
SQLite projection writer for StepMemorySampler.

This module projects TraceML StepMemorySampler payloads into a query-friendly
SQLite table while preserving the original sampler payload in `raw_messages`.

Design
------
- Keeps sampler-specific SQL logic out of the core SQLite writer.
- Accepts canonical telemetry envelopes from the main writer.
- Produces one query-friendly table:
    1) step_memory_samples
       One row per (global rank, step memory event) with stable runtime
       identity and memory bytes fields.

Expected payload shape
----------------------
Canonical telemetry envelope:
{
    "meta": {"sampler": "StepMemorySampler", ...},
    "body": {
        "tables": {"<table_name>": [StepMemorySample.to_wire(), ...]}
    }
}
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

from traceml_ai.telemetry.envelope import TelemetryEnvelope, TelemetryMeta

SAMPLER_NAME = "StepMemorySampler"
RETENTION_TABLES = ("step_memory_samples",)


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
class StepMemoryPayloadIdentity:
    """Distributed identity stored with projected step-memory telemetry."""

    rank: Optional[int]
    global_rank: Optional[int]
    local_rank: Optional[int]
    world_size: Optional[int]
    local_world_size: Optional[int]
    node_rank: Optional[int]
    hostname: Optional[str]


def _payload_identity(meta: TelemetryMeta) -> StepMemoryPayloadIdentity:
    """Return storage identity for one StepMemorySampler payload."""
    global_rank = _optional_int(meta.global_rank)
    legacy_rank = _optional_int(meta.rank)
    rank = global_rank if global_rank is not None else legacy_rank

    return StepMemoryPayloadIdentity(
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
    Create query-friendly projection table for StepMemorySampler.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS step_memory_samples (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            recv_ts_ns           INTEGER NOT NULL,
            rank                 INTEGER,
            global_rank          INTEGER,
            local_rank           INTEGER,
            world_size           INTEGER,
            local_world_size     INTEGER,
            node_rank            INTEGER,
            hostname             TEXT,
            sample_ts_s          REAL,
            seq                  INTEGER,
            device               TEXT,
            step                 INTEGER,
            peak_alloc_bytes     REAL,
            peak_reserved_bytes  REAL
        );
        """
    )
    _ensure_column(
        conn,
        table="step_memory_samples",
        column="global_rank",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="step_memory_samples",
        column="local_rank",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="step_memory_samples",
        column="world_size",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="step_memory_samples",
        column="local_world_size",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="step_memory_samples",
        column="node_rank",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="step_memory_samples",
        column="hostname",
        definition="TEXT",
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
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_step_memory_samples_global_rank_step_ts
        ON step_memory_samples(global_rank, step, sample_ts_s, id);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_step_memory_samples_step_global_rank
        ON step_memory_samples(step, global_rank, id);
        """
    )


def build_rows(
    envelope: TelemetryEnvelope,
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
            ts_raw = row.get("ts")
            device_raw = row.get("device")
            step_raw = row.get("step")
            peak_alloc_raw = row.get("peak_alloc")
            peak_resv_raw = row.get("peak_resv")

            seq = int(seq_raw) if isinstance(seq_raw, int) else None
            sample_ts_s = (
                float(ts_raw) if isinstance(ts_raw, (int, float)) else None
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
                    identity.rank,
                    identity.global_rank,
                    identity.local_rank,
                    identity.world_size,
                    identity.local_world_size,
                    identity.node_rank,
                    identity.hostname,
                    sample_ts_s,
                    seq,
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
                global_rank,
                local_rank,
                world_size,
                local_world_size,
                node_rank,
                hostname,
                sample_ts_s,
                seq,
                device,
                step,
                peak_alloc_bytes,
                peak_reserved_bytes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
