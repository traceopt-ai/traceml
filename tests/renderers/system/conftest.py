# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""A real system_samples database for the System repository tests.

Rows go in through `tests/sqlite_fixtures.insert_system_sample`, which
builds them from the writer's own schema, so a column rename in the writer
fails these tests instead of drifting past them. Mirrors
`tests/renderers/process/conftest.py`; the difference is that System's
rows carry per-GPU children, so the factory takes a per-tick GPU list.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import pytest

from tests.sqlite_fixtures import init_summary_schema, insert_system_sample

GB = 1_000_000_000.0


def gpu(
    idx: int,
    *,
    util: Optional[float] = 100.0,
    power: Optional[float] = 66.0,
    limit: Optional[float] = 70.0,
    temp: Optional[float] = 45.0,
    mem_used: Optional[float] = 6.3 * GB,
    mem_total: Optional[float] = 16.1 * GB,
) -> Mapping[str, Any]:
    """One per-GPU child row."""
    return {
        "gpu_idx": idx,
        "util": util,
        "mem_used_bytes": mem_used,
        "mem_total_bytes": mem_total,
        "temperature_c": temp,
        "power_usage_w": power,
        "power_limit_w": limit,
    }


@pytest.fixture
def system_db(tmp_path: Path) -> Callable[..., str]:
    """Write a run of system samples and return the database path.

    `cadence_s` and `ticks` shape the clock, which is what every
    window-planning assertion in this file turns on. `gaps` inserts extra
    seconds before the given tick indices, which is how a run with missing
    samples is built without changing the rest of the timeline.
    """

    path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(path)
    init_summary_schema(conn)
    conn.commit()
    conn.close()

    def write(
        ticks: int = 20,
        *,
        cadence_s: float = 2.0,
        cpu: Callable[[int], Optional[float]] = lambda seq: 10.0,
        gpus: Optional[Callable[[int], Sequence[Mapping[str, Any]]]] = None,
        hostnames: Sequence[str] = ("box",),
        gaps: Mapping[int, float] = {},
        start_ts: float = 1000.0,
    ) -> str:
        row_id = 0
        conn = sqlite3.connect(path)
        try:
            offset = 0.0
            for seq in range(ticks):
                offset += gaps.get(seq, 0.0)
                for node, host in enumerate(hostnames):
                    row_id += 1
                    insert_system_sample(
                        conn,
                        row_id=row_id,
                        rank=node,
                        ts=start_ts + cadence_s * seq + offset,
                        gpu_available=gpus is not None,
                        gpu_count=len(gpus(seq)) if gpus else 0,
                        world_size=len(hostnames),
                        global_rank=node,
                        node_rank=node,
                        hostname=host,
                        seq=seq,
                        cpu_percent=cpu(seq),
                        ram_used_bytes=2.0 * GB,
                        ram_total_bytes=16.0 * GB,
                        gpu_samples=list(gpus(seq)) if gpus else (),
                    )
            conn.commit()
        finally:
            conn.close()
        return str(path)

    return write
