# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""SQLite projection writer for RuntimeEnvironmentSampler.

The runtime environment table stores one small rank-scoped context row, such as
topology and observed training strategy. It is sampler body data, not TCP
envelope metadata, and is intentionally excluded from retention pruning.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

from traceml_ai.telemetry.envelope import TelemetryEnvelope, TelemetryMeta

SAMPLER_NAME = "RuntimeEnvironmentSampler"
TABLE_NAME = "RuntimeEnvironmentTable"


def _optional_float(value: Any) -> Optional[float]:
    """Best-effort float coercion for runtime environment rows."""
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _optional_int(value: Any) -> Optional[int]:
    """Best-effort integer coercion for runtime environment rows."""
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _optional_str(value: Any) -> Optional[str]:
    """Best-effort string coercion for runtime environment rows."""
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


def _optional_bool_int(value: Any) -> Optional[int]:
    """Return booleans as 1/0 while preserving unknown as NULL."""
    if value is True:
        return 1
    if value is False:
        return 0
    return None


@dataclass(frozen=True)
class RuntimeEnvironmentPayloadIdentity:
    """Distributed identity stored with runtime environment telemetry."""

    rank: Optional[int]
    global_rank: Optional[int]
    local_rank: Optional[int]
    world_size: Optional[int]
    local_world_size: Optional[int]
    node_rank: Optional[int]
    hostname: Optional[str]
    pid: Optional[int]


def _payload_identity(
    meta: TelemetryMeta,
) -> RuntimeEnvironmentPayloadIdentity:
    """Return storage identity for one RuntimeEnvironmentSampler payload."""
    return RuntimeEnvironmentPayloadIdentity(
        rank=_optional_int(meta.rank),
        global_rank=_optional_int(meta.global_rank),
        local_rank=_optional_int(meta.local_rank),
        world_size=_optional_int(meta.world_size),
        local_world_size=_optional_int(meta.local_world_size),
        node_rank=_optional_int(meta.node_rank),
        hostname=_optional_str(meta.hostname),
        pid=_optional_int(meta.pid),
    )


def accepts_sampler(sampler: Optional[str]) -> bool:
    """Return True if this projection writer handles the given sampler."""
    return sampler == SAMPLER_NAME


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the query-friendly runtime environment table."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime_environment (
            id                         INTEGER PRIMARY KEY AUTOINCREMENT,
            recv_ts_ns                 INTEGER NOT NULL,
            rank                       INTEGER,
            global_rank                INTEGER,
            local_rank                 INTEGER,
            world_size                 INTEGER,
            local_world_size           INTEGER,
            node_rank                  INTEGER,
            hostname                   TEXT,
            pid                        INTEGER,
            sample_ts_s                REAL,
            seq                        INTEGER,
            topology                   TEXT,
            distributed_initialized    INTEGER,
            distributed_backend        TEXT,
            training_strategy          TEXT,
            strategy_source            TEXT,
            strategy_confidence        TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_runtime_environment_global_rank
        ON runtime_environment(global_rank, id);
        """
    )


def build_rows(
    envelope: TelemetryEnvelope,
    recv_ts_ns: int,
) -> Dict[str, list[tuple]]:
    """Build SQLite rows from one decoded RuntimeEnvironmentSampler payload."""
    out: Dict[str, list[tuple]] = {"runtime_environment": []}

    sampler = envelope.meta.sampler
    if not accepts_sampler(sampler):
        return out

    identity = _payload_identity(envelope.meta)
    tables = envelope.tables
    if not isinstance(tables, dict):
        return out

    rows = tables.get(TABLE_NAME)
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue

        out["runtime_environment"].append(
            (
                recv_ts_ns,
                identity.rank,
                identity.global_rank,
                identity.local_rank,
                identity.world_size,
                identity.local_world_size,
                identity.node_rank,
                identity.hostname,
                identity.pid,
                _optional_float(row.get("ts")),
                _optional_int(row.get("seq")),
                _optional_str(row.get("topology")),
                _optional_bool_int(row.get("distributed_initialized")),
                _optional_str(row.get("distributed_backend")),
                _optional_str(row.get("training_strategy")),
                _optional_str(row.get("strategy_source")),
                _optional_str(row.get("strategy_confidence")),
            )
        )

    return out


def insert_rows(
    conn: sqlite3.Connection,
    rows_by_table: Dict[str, list[tuple]],
) -> None:
    """Insert runtime environment projection rows into SQLite."""
    rows = rows_by_table.get("runtime_environment", [])
    if rows:
        conn.executemany(
            """
            INSERT INTO runtime_environment(
                recv_ts_ns,
                rank,
                global_rank,
                local_rank,
                world_size,
                local_world_size,
                node_rank,
                hostname,
                pid,
                sample_ts_s,
                seq,
                topology,
                distributed_initialized,
                distributed_backend,
                training_strategy,
                strategy_source,
                strategy_confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
