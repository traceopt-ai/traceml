# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""A real process_samples database for the Process renderer tests.

The schema comes from the writer that owns the table rather than a
hand-written approximation, so a column rename in the writer fails these
tests instead of silently drifting past them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

import pytest

from traceml_ai.aggregator.sqlite_writers.process import init_schema

GB = 1_000_000_000.0


@pytest.fixture
def process_db(tmp_path: Path):
    """Return a factory that writes rows and yields the database path."""

    path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(path)
    init_schema(conn)
    conn.commit()
    conn.close()

    def insert(**row: Any) -> None:
        columns = {
            "recv_ts_ns": 0,
            "rank": 0,
            "global_rank": None,
            "world_size": None,
            "node_rank": None,
            "hostname": None,
            "sample_ts_s": None,
            "seq": None,
            "cpu_percent": None,
            "cpu_logical_core_count": None,
            "ram_used_bytes": None,
            "ram_total_bytes": None,
            "gpu_available": 0,
            "gpu_count": None,
            "gpu_device_index": None,
            "gpu_mem_used_bytes": None,
            "gpu_mem_reserved_bytes": None,
            "gpu_mem_total_bytes": None,
        }
        columns.update(row)
        names = ", ".join(columns)
        marks = ", ".join("?" for _ in columns)
        conn = sqlite3.connect(path)
        conn.execute(
            f"INSERT INTO process_samples ({names}) VALUES ({marks})",
            tuple(columns.values()),
        )
        conn.commit()
        conn.close()

    def sample(
        *,
        seq: int,
        rank: int,
        cpu: Optional[float] = 10.0,
        ram: Optional[float] = 2.0 * GB,
        ram_total: Optional[float] = 16.0 * GB,
        gpu_used: Optional[float] = None,
        gpu_reserved: Optional[float] = None,
        gpu_total: Optional[float] = None,
        ts: Optional[float] = None,
    ) -> None:
        insert(
            seq=seq,
            rank=rank,
            sample_ts_s=1_700_000_000.0 + seq if ts is None else ts,
            cpu_percent=cpu,
            ram_used_bytes=ram,
            ram_total_bytes=ram_total,
            gpu_available=1 if gpu_used is not None else 0,
            gpu_mem_used_bytes=gpu_used,
            gpu_mem_reserved_bytes=gpu_reserved,
            gpu_mem_total_bytes=gpu_total,
        )

    return type(
        "ProcessDB",
        (),
        {
            "path": str(path),
            "insert": staticmethod(insert),
            "sample": staticmethod(sample),
        },
    )
