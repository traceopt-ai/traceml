# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
SQLite projection writer for BatchSizeSampler.

Projects per-step input batch-size-in-bytes samples into the
``batch_size_samples`` table. One row per (global_rank, step).
Payload rows follow BatchSizeSample.to_wire().
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

SAMPLER_NAME = "BatchSizeSampler"
RETENTION_TABLES = ("batch_size_samples",)


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
class BatchSizePayloadIdentity:
    """Distributed identity stored with projected batch-size telemetry."""

    rank: Optional[int]
    global_rank: Optional[int]
    local_rank: Optional[int]
    world_size: Optional[int]
    local_world_size: Optional[int]
    node_rank: Optional[int]
    hostname: Optional[str]
    runtime_pid: Optional[int]


def _payload_identity(
    payload_dict: Dict[str, Any]
) -> BatchSizePayloadIdentity:
    """Return storage identity for one BatchSizeSampler payload."""
    global_rank = _optional_int(payload_dict.get("global_rank"))
    legacy_rank = _optional_int(payload_dict.get("rank"))
    rank = global_rank if global_rank is not None else legacy_rank

    return BatchSizePayloadIdentity(
        rank=rank,
        global_rank=global_rank,
        local_rank=_optional_int(payload_dict.get("local_rank")),
        world_size=_optional_int(payload_dict.get("world_size")),
        local_world_size=_optional_int(payload_dict.get("local_world_size")),
        node_rank=_optional_int(payload_dict.get("node_rank")),
        hostname=_optional_str(payload_dict.get("hostname")),
        runtime_pid=_optional_int(payload_dict.get("pid")),
    )


def accepts_sampler(sampler: Optional[str]) -> bool:
    """Return True if this projection writer handles the given sampler."""
    return sampler == SAMPLER_NAME


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the batch_size_samples projection table and indexes."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS batch_size_samples (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            recv_ts_ns         INTEGER NOT NULL,
            rank               INTEGER,
            global_rank        INTEGER,
            local_rank         INTEGER,
            world_size         INTEGER,
            local_world_size   INTEGER,
            node_rank          INTEGER,
            hostname           TEXT,
            runtime_pid        INTEGER,
            sample_ts_s        REAL,
            seq                INTEGER,
            step               INTEGER,
            bytes_total        INTEGER NOT NULL,
            n_transfers        INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_batch_size_samples_global_rank_step_ts
        ON batch_size_samples(global_rank, step, sample_ts_s, id);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_batch_size_samples_step_rank
        ON batch_size_samples(step, rank, id);
        """
    )


def build_rows(
    payload_dict: Dict[str, Any],
    recv_ts_ns: int,
) -> Dict[str, list[tuple]]:
    """
    Build SQLite projection rows from one decoded BatchSizeSampler payload.

    Returns
    -------
    dict[str, list[tuple]]
        {"batch_size_samples": [...]}
    """
    out: Dict[str, list[tuple]] = {"batch_size_samples": []}

    sampler = payload_dict.get("sampler")
    if not accepts_sampler(str(sampler) if sampler is not None else None):
        return out

    identity = _payload_identity(payload_dict)

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
            bytes_raw = row.get("bytes_total")
            n_raw = row.get("n_transfers")

            seq = int(seq_raw) if isinstance(seq_raw, int) else None
            sample_ts_s = (
                float(ts_raw) if isinstance(ts_raw, (int, float)) else None
            )
            step = int(step_raw) if isinstance(step_raw, int) else None
            bytes_total = (
                int(bytes_raw) if isinstance(bytes_raw, (int, float)) else 0
            )
            n_transfers = int(n_raw) if isinstance(n_raw, int) else 0

            out["batch_size_samples"].append(
                (
                    recv_ts_ns,
                    identity.rank,
                    identity.global_rank,
                    identity.local_rank,
                    identity.world_size,
                    identity.local_world_size,
                    identity.node_rank,
                    identity.hostname,
                    identity.runtime_pid,
                    sample_ts_s,
                    seq,
                    step,
                    bytes_total,
                    n_transfers,
                )
            )

    return out


def insert_rows(
    conn: sqlite3.Connection, rows_by_table: Dict[str, list[tuple]]
) -> None:
    """Insert projection rows into SQLite."""
    rows = rows_by_table.get("batch_size_samples", [])
    if rows:
        conn.executemany(
            """
            INSERT INTO batch_size_samples(
                recv_ts_ns,
                rank,
                global_rank,
                local_rank,
                world_size,
                local_world_size,
                node_rank,
                hostname,
                runtime_pid,
                sample_ts_s,
                seq,
                step,
                bytes_total,
                n_transfers
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
