"""
SQLite projection writer for SystemSampler.

This module projects TraceML SystemSampler payloads into query-friendly
SQLite tables while preserving the original sampler payload in the main
`raw_messages` table.

Design
------
- Keeps sampler-specific SQL logic out of the core SQLite writer
- Accepts already-decoded payload dicts from the main writer
- Produces two query-friendly tables:
    1) system_samples
       One row per system sample with host-level fields and aggregated GPU stats
    2) system_gpu_samples
       One row per GPU within each system sample

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
Envelope:
{
    "rank": int,
    "sampler": "SystemSampler",
    "timestamp": float,
    "tables": {
        "<table_name>": [
            {
                "seq": int,
                "ts": float,
                "cpu": float,
                "ram_used": float,   # bytes
                "ram_total": float,  # bytes
                "gpu_available": bool,
                "gpu_count": int,
                "gpus": [
                    [util, mem_used, mem_total, temperature, power_usage, power_limit],
                    ...
                ]
            },
            ...
        ]
    }
}
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Optional

SAMPLER_NAME = "SystemSampler"


def accepts_sampler(sampler: Optional[str]) -> bool:
    """Return True if this projection writer handles the given sampler."""
    return sampler == SAMPLER_NAME


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Create query-friendly projection tables for SystemSampler.

    Tables
    ------
    system_samples
        One row per sampled system snapshot. Includes host fields plus
        aggregated GPU statistics for the snapshot.

    system_gpu_samples
        One row per GPU within a sampled system snapshot. Useful for detailed
        per-GPU analysis and future imbalance/hotspot queries.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS system_samples (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            recv_ts_ns             INTEGER NOT NULL,
            rank                   INTEGER,
            sample_ts_s            REAL,
            seq                    INTEGER,
            cpu_percent            REAL,
            ram_used_bytes         REAL,
            ram_total_bytes        REAL,
            gpu_available          INTEGER,
            gpu_count              INTEGER,
            gpu_util_avg           REAL,
            gpu_util_peak          REAL,
            gpu_mem_used_avg_bytes REAL,
            gpu_mem_used_peak_bytes REAL,
            gpu_temp_avg_c         REAL,
            gpu_temp_peak_c        REAL,
            gpu_power_avg_w        REAL,
            gpu_power_peak_w       REAL
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_system_samples_rank_ts
        ON system_samples(rank, sample_ts_s, id);
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS system_gpu_samples (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            recv_ts_ns       INTEGER NOT NULL,
            rank             INTEGER,
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
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_system_gpu_samples_rank_gpu_ts
        ON system_gpu_samples(rank, gpu_idx, sample_ts_s, id);
        """
    )


def build_rows(
    payload_dict: Dict[str, Any],
    recv_ts_ns: int,
) -> Dict[str, list[tuple]]:
    """
    Build SQLite projection rows from one decoded SystemSampler payload.

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

            utils: list[float] = []
            mem_useds_bytes: list[float] = []
            temps_c: list[float] = []
            powers_w: list[float] = []

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

                    if util is not None:
                        utils.append(util)
                    if mem_used_bytes is not None:
                        mem_useds_bytes.append(mem_used_bytes)
                    if temp_c is not None:
                        temps_c.append(temp_c)
                    if power_w is not None:
                        powers_w.append(power_w)

                    out["system_gpu_samples"].append(
                        (
                            recv_ts_ns,
                            rank,
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

            gpu_util_avg = sum(utils) / len(utils) if utils else None
            gpu_util_peak = max(utils) if utils else None
            gpu_mem_used_avg_bytes = (
                sum(mem_useds_bytes) / len(mem_useds_bytes)
                if mem_useds_bytes
                else None
            )
            gpu_mem_used_peak_bytes = (
                max(mem_useds_bytes) if mem_useds_bytes else None
            )
            gpu_temp_avg_c = sum(temps_c) / len(temps_c) if temps_c else None
            gpu_temp_peak_c = max(temps_c) if temps_c else None
            gpu_power_avg_w = (
                sum(powers_w) / len(powers_w) if powers_w else None
            )
            gpu_power_peak_w = max(powers_w) if powers_w else None

            out["system_samples"].append(
                (
                    recv_ts_ns,
                    rank,
                    sample_ts_s,
                    seq,
                    cpu_percent,
                    ram_used_bytes,
                    ram_total_bytes,
                    gpu_available,
                    gpu_count,
                    gpu_util_avg,
                    gpu_util_peak,
                    gpu_mem_used_avg_bytes,
                    gpu_mem_used_peak_bytes,
                    gpu_temp_avg_c,
                    gpu_temp_peak_c,
                    gpu_power_avg_w,
                    gpu_power_peak_w,
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
                rank,
                sample_ts_s,
                seq,
                cpu_percent,
                ram_used_bytes,
                ram_total_bytes,
                gpu_available,
                gpu_count,
                gpu_util_avg,
                gpu_util_peak,
                gpu_mem_used_avg_bytes,
                gpu_mem_used_peak_bytes,
                gpu_temp_avg_c,
                gpu_temp_peak_c,
                gpu_power_avg_w,
                gpu_power_peak_w
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            system_rows,
        )

    gpu_rows = rows_by_table.get("system_gpu_samples", [])
    if gpu_rows:
        conn.executemany(
            """
            INSERT INTO system_gpu_samples(
                recv_ts_ns,
                rank,
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            gpu_rows,
        )
