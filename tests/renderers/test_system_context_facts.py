# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Context facts in the SYSTEM payload ctx: observed ranks, nodes, clocks.

``ranks_reporting`` follows the card's rule: a rank counts as reporting
while its newest sample is within three ticks of the newest sample
anywhere, so a rank that dies leaves the numerator while ``world_size``
stays the denominator. The rule is on clocks, not seq: two nodes that
start 20 s apart carry a permanent seq offset and must still read N/N.
"""

from __future__ import annotations

from pathlib import Path

from tests.sqlite_fixtures import (
    init_summary_schema,
    insert_process_sample,
    insert_system_sample,
    insert_training_strategy,
    sqlite_database,
)
from traceml_ai.renderers.system.common import SystemMetricsDB
from traceml_ai.renderers.system.dashboard_compute import (
    SystemDashboardComputer,
)

WORLD = 4


def _write_run(path: Path, *, dead_rank_last_seq: int = 9) -> None:
    """Ranks 0-2 reach seq 19; rank 3 stops at ``dead_rank_last_seq``."""
    with sqlite_database(path, init_summary_schema) as conn:
        row_id = 0
        for seq in range(21):
            row_id += 1
            insert_system_sample(
                conn,
                row_id=row_id,
                rank=0,
                ts=100.0 + 2.0 * seq,
                gpu_available=False,
                gpu_count=0,
                world_size=WORLD,
                local_world_size=WORLD,
                hostname="node-a",
                seq=seq,
                cpu_percent=3.0,
                ram_used_bytes=1000.0,
            )
        for rank in range(WORLD):
            last_seq = dead_rank_last_seq if rank == WORLD - 1 else 19
            for seq in range(last_seq + 1):
                row_id += 1
                insert_process_sample(
                    conn,
                    row_id=row_id,
                    rank=rank,
                    ts=100.0 + 2.0 * seq,
                    gpu_available=False,
                    gpu_count=0,
                    global_rank=rank,
                    world_size=WORLD,
                    local_world_size=WORLD,
                    hostname="node-a",
                    seq=seq,
                )
        insert_training_strategy(conn, "ddp")


def test_dead_rank_leaves_the_numerator(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    _write_run(db)
    with SystemMetricsDB(str(db)).connect() as conn:
        facts = SystemMetricsDB(str(db)).fetch_context_facts(conn)
    assert facts["ranks_reporting"] == 3
    assert facts["node_count"] == 1
    assert facts["training_strategy"] == "ddp"
    assert facts["first_data_ts"] == 100.0
    assert facts["last_data_ts"] == 140.0


def test_all_ranks_within_three_ticks_count(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    _write_run(db, dead_rank_last_seq=17)  # 4 s behind at a 2 s tick
    with SystemMetricsDB(str(db)).connect() as conn:
        facts = SystemMetricsDB(str(db)).fetch_context_facts(conn)
    assert facts["ranks_reporting"] == WORLD


def test_facts_ride_on_the_dashboard_ctx(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    _write_run(db)
    payload = SystemDashboardComputer(db_path=str(db)).compute()
    ctx = payload["rollups"]["ctx"]
    assert ctx["world_size"] == WORLD
    assert ctx["ranks_reporting"] == 3
    assert ctx["node_count"] == 1
    assert ctx["training_strategy"] == "ddp"
    assert ctx["last_data_ts"] == 140.0


def test_no_process_data_reports_zero_not_world_size(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    with sqlite_database(db, init_summary_schema) as conn:
        insert_system_sample(
            conn,
            row_id=1,
            rank=0,
            ts=100.0,
            gpu_available=False,
            gpu_count=0,
            world_size=WORLD,
            hostname="node-a",
        )
    with SystemMetricsDB(str(db)).connect() as conn:
        facts = SystemMetricsDB(str(db)).fetch_context_facts(conn)
    assert facts["ranks_reporting"] == 0
    assert facts["training_strategy"] == ""


def test_gpus_observed_sums_over_nodes(tmp_path: Path) -> None:
    """2 nodes x 1 GPU must read 2 GPUs observed, not one node's count."""
    db = tmp_path / "t.db"
    with sqlite_database(db, init_summary_schema) as conn:
        for row_id, (host, rank) in enumerate(
            (("node-a", 0), ("node-b", 1), ("node-a", 0), ("node-b", 1)),
            start=1,
        ):
            insert_system_sample(
                conn,
                row_id=row_id,
                rank=rank,
                ts=100.0 + row_id,
                gpu_available=True,
                gpu_count=1,
                world_size=2,
                local_world_size=1,
                hostname=host,
                seq=row_id,
            )
    with SystemMetricsDB(str(db)).connect() as conn:
        facts = SystemMetricsDB(str(db)).fetch_context_facts(conn)
    assert facts["node_count"] == 2
    assert facts["gpus_observed"] == 2


def test_start_offset_between_nodes_is_not_a_dead_rank(
    tmp_path: Path,
) -> None:
    """Rank 1 started 10 ticks before rank 0 (its seq runs 10 ahead for the
    whole run); both are live at the end, so the strip must read 2/2."""
    db = tmp_path / "t.db"
    with sqlite_database(db, init_summary_schema) as conn:
        row_id = 0
        for rank, first_seq in ((0, 0), (1, 10)):
            for k in range(30):
                row_id += 1
                insert_process_sample(
                    conn,
                    row_id=row_id,
                    rank=rank,
                    ts=100.0 + 2.0 * k,  # same wall clock for both ranks
                    gpu_available=False,
                    gpu_count=0,
                    global_rank=rank,
                    world_size=2,
                    local_world_size=1,
                    hostname=f"node-{rank}",
                    seq=first_seq + k,
                )
    with SystemMetricsDB(str(db)).connect() as conn:
        facts = SystemMetricsDB(str(db)).fetch_context_facts(conn)
    assert facts["ranks_reporting"] == 2
