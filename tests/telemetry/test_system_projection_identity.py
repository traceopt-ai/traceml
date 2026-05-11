# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3

from traceml.aggregator.sqlite_writers import system as system_projection


def _system_payload() -> dict:
    return {
        "rank": 5,
        "global_rank": 5,
        "local_rank": 1,
        "world_size": 8,
        "local_world_size": 4,
        "node_rank": 1,
        "hostname": "worker-1",
        "pid": 12345,
        "sampler": "SystemSampler",
        "timestamp": 123.0,
        "tables": {
            "SystemTable": [
                {
                    "seq": 7,
                    "ts": 123.0,
                    "cpu": 42.0,
                    "ram_used": 4_000.0,
                    "ram_total": 16_000.0,
                    "gpu_available": True,
                    "gpu_count": 1,
                    "gpus": [[70.0, 5_000.0, 10_000.0, 68.0, 120.0, 250.0]],
                }
            ]
        },
    }


def test_system_projection_stores_global_and_local_rank_identity() -> None:
    conn = sqlite3.connect(":memory:")
    system_projection.init_schema(conn)

    rows = system_projection.build_rows(_system_payload(), recv_ts_ns=999)
    system_projection.insert_rows(conn, rows)

    sample = conn.execute(
        """
        SELECT global_rank, local_rank, world_size, local_world_size,
               node_rank, hostname, pid, seq
        FROM system_samples;
        """
    ).fetchone()
    gpu_sample = conn.execute(
        """
        SELECT global_rank, local_rank, world_size, local_world_size,
               node_rank, hostname, pid, gpu_idx
        FROM system_gpu_samples;
        """
    ).fetchone()

    assert sample == (5, 1, 8, 4, 1, "worker-1", 12345, 7)
    assert gpu_sample == (5, 1, 8, 4, 1, "worker-1", 12345, 0)
