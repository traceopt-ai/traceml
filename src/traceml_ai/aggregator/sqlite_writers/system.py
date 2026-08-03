# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
SQLite projection writer for SystemSampler.

This module projects TraceML SystemSampler payloads into query-friendly
SQLite tables.

Storage units
-------------
This projection stores raw values wherever possible.

- cpu_percent            : percent
- ram_used_bytes         : bytes
- ram_total_bytes        : bytes
- util                   : percent
- mem_used_bytes         : bytes
- mem_total_bytes        : bytes
- temperature_c          : Celsius
- power_usage_w          : watts
- power_limit_w          : watts

Expected payload shape
----------------------
Canonical telemetry envelope:
{
    "meta": {"sampler": "SystemSampler", ...},
    "body": {
        "tables": {"<table_name>": [SystemSample.to_wire(), ...]}
    }
}
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Optional

from traceml_ai.telemetry.envelope import TelemetryEnvelope, TelemetryMeta

SAMPLER_NAME = "SystemSampler"
RETENTION_TABLES = ("system_samples",)


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
class SystemPayloadIdentity:
    """Distributed identity stored with projected system telemetry."""

    global_rank: Optional[int]
    local_rank: Optional[int]
    world_size: Optional[int]
    local_world_size: Optional[int]
    node_rank: Optional[int]
    hostname: Optional[str]


def _payload_identity(meta: TelemetryMeta) -> SystemPayloadIdentity:
    """
    Return storage identity for one payload.

    System projection tables store explicit distributed identity fields only.
    The legacy generic ``rank`` field is intentionally ignored.
    """
    return SystemPayloadIdentity(
        global_rank=_optional_int(meta.global_rank),
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
    Create query-friendly projection tables for SystemSampler.

    Tables
    ------
    system_samples
        One row per sampled system snapshot. Includes node-level raw fields.

    system_gpu_samples
        One row per GPU within a sampled system snapshot. Summary and live
        views derive GPU rollups from this table instead of storing duplicate
        per-snapshot aggregates in `system_samples`.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS system_samples (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            recv_ts_ns             INTEGER NOT NULL,
            global_rank            INTEGER,
            local_rank             INTEGER,
            world_size             INTEGER,
            local_world_size       INTEGER,
            node_rank              INTEGER,
            hostname               TEXT,
            sample_ts_s            REAL,
            seq                    INTEGER,
            cpu_percent            REAL,
            ram_used_bytes         REAL,
            ram_total_bytes        REAL,
            gpu_available          INTEGER,
            gpu_count              INTEGER
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS system_gpu_samples (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            recv_ts_ns       INTEGER NOT NULL,
            global_rank      INTEGER,
            local_rank       INTEGER,
            world_size       INTEGER,
            local_world_size INTEGER,
            node_rank        INTEGER,
            hostname         TEXT,
            sample_ts_s      REAL,
            seq              INTEGER,
            gpu_idx          INTEGER NOT NULL,
            util             REAL,
            mem_used_bytes   REAL,
            mem_total_bytes  REAL,
            temperature_c    REAL,
            power_usage_w    REAL,
            power_limit_w    REAL
        );
        """
    )
    _ensure_column(
        conn,
        table="system_samples",
        column="global_rank",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="system_samples",
        column="local_rank",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="system_samples",
        column="world_size",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="system_samples",
        column="local_world_size",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="system_samples",
        column="node_rank",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="system_samples",
        column="hostname",
        definition="TEXT",
    )
    _ensure_column(
        conn,
        table="system_gpu_samples",
        column="global_rank",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="system_gpu_samples",
        column="local_rank",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="system_gpu_samples",
        column="world_size",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="system_gpu_samples",
        column="local_world_size",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="system_gpu_samples",
        column="node_rank",
        definition="INTEGER",
    )
    _ensure_column(
        conn,
        table="system_gpu_samples",
        column="hostname",
        definition="TEXT",
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_system_samples_node_ts
        ON system_samples(node_rank, sample_ts_s, id);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_system_gpu_samples_global_gpu_ts
        ON system_gpu_samples(global_rank, gpu_idx, sample_ts_s, id);
        """
    )


def build_rows(
    envelope: TelemetryEnvelope,
    recv_ts_ns: int,
) -> Dict[str, list[tuple]]:
    """
    Build SQLite projection rows from one decoded SystemSampler payload.

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
            "system_samples": [...],
            "system_gpu_samples": [...],
        }

    Notes
    -----
    - Returns empty lists if payload is malformed or belongs to another sampler.
    - Keeps projection logic best-effort and non-throwing.
    - Assumes payload rows follow SystemSample.to_wire().
    """
    out: Dict[str, list[tuple]] = {
        "system_samples": [],
        "system_gpu_samples": [],
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
            cpu_raw = row.get("cpu")
            ram_used_raw = row.get("ram_used")
            ram_total_raw = row.get("ram_total")
            gpu_available_raw = row.get("gpu_available")
            gpu_count_raw = row.get("gpu_count")
            gpus_raw = row.get("gpus")

            seq = int(seq_raw) if isinstance(seq_raw, int) else None
            sample_ts_s = (
                float(ts_raw) if isinstance(ts_raw, (int, float)) else None
            )
            cpu_percent = (
                float(cpu_raw) if isinstance(cpu_raw, (int, float)) else None
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

            if isinstance(gpus_raw, list):
                for gpu_idx, g in enumerate(gpus_raw):
                    if not (isinstance(g, list) and len(g) >= 6):
                        continue

                    util_raw = g[0]
                    mem_used_raw = g[1]
                    mem_total_raw = g[2]
                    temp_raw = g[3]
                    power_raw = g[4]
                    power_limit_raw = g[5]

                    util = (
                        float(util_raw)
                        if isinstance(util_raw, (int, float))
                        else None
                    )
                    mem_used_bytes = (
                        float(mem_used_raw)
                        if isinstance(mem_used_raw, (int, float))
                        else None
                    )
                    mem_total_bytes = (
                        float(mem_total_raw)
                        if isinstance(mem_total_raw, (int, float))
                        else None
                    )
                    temp_c = (
                        float(temp_raw)
                        if isinstance(temp_raw, (int, float))
                        else None
                    )
                    power_w = (
                        float(power_raw)
                        if isinstance(power_raw, (int, float))
                        else None
                    )
                    power_limit_w = (
                        float(power_limit_raw)
                        if isinstance(power_limit_raw, (int, float))
                        else None
                    )

                    out["system_gpu_samples"].append(
                        (
                            recv_ts_ns,
                            identity.global_rank,
                            identity.local_rank,
                            identity.world_size,
                            identity.local_world_size,
                            identity.node_rank,
                            identity.hostname,
                            sample_ts_s,
                            seq,
                            gpu_idx,
                            util,
                            mem_used_bytes,
                            mem_total_bytes,
                            temp_c,
                            power_w,
                            power_limit_w,
                        )
                    )

            out["system_samples"].append(
                (
                    recv_ts_ns,
                    identity.global_rank,
                    identity.local_rank,
                    identity.world_size,
                    identity.local_world_size,
                    identity.node_rank,
                    identity.hostname,
                    sample_ts_s,
                    seq,
                    cpu_percent,
                    ram_used_bytes,
                    ram_total_bytes,
                    gpu_available,
                    gpu_count,
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
    system_rows = rows_by_table.get("system_samples", [])
    if system_rows:
        conn.executemany(
            """
            INSERT INTO system_samples(
                recv_ts_ns,
                global_rank,
                local_rank,
                world_size,
                local_world_size,
                node_rank,
                hostname,
                sample_ts_s,
                seq,
                cpu_percent,
                ram_used_bytes,
                ram_total_bytes,
                gpu_available,
                gpu_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            system_rows,
        )

    gpu_rows = rows_by_table.get("system_gpu_samples", [])
    if gpu_rows:
        conn.executemany(
            """
            INSERT INTO system_gpu_samples(
                recv_ts_ns,
                global_rank,
                local_rank,
                world_size,
                local_world_size,
                node_rank,
                hostname,
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            gpu_rows,
        )
