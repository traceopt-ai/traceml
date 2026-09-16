# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
SQLite reads for the run-context strip.

Context is deliberately cross-domain: which run this is and how much of it
is reporting. It combines the existing tables (system, process, runtime
environment) without owning any of them, so no metric renderer has to
answer questions outside its own domain. Every value is observed, never
configured; each query degrades on its own so an older database that
lacks a table never blanks the strip.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict

# Default sampler cadence; the floor for the per-rank tick estimate.
_TICK_SEC = 2.0


def empty_context() -> Dict[str, Any]:
    """The full key set, all unknown."""
    return {
        "world_size": 0,
        "gpu_count": 0,
        "hostname": "",
        "ranks_reporting": None,
        "node_count": None,
        "gpus_observed": None,
        "training_strategy": "",
        "first_data_ts": None,
        "last_data_ts": None,
    }


class ContextDB:
    """Short-lived read connections plus the context queries."""

    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def fetch_context(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """
        Facts for the context strip: what is reporting, and for how long.

        - world_size / gpu_count / hostname: the newest system sample's
          view of its node (configured world size, the node's GPU count).
        - ranks_reporting: ranks whose newest sample is within three ticks
          of the newest sample anywhere (tick = the fastest rank's cadence,
          floored at the default sampler interval). Time, never seq: seq
          counters carry each rank's start offset.
        - node_count: distinct hostnames that produced system samples.
        - gpus_observed: newest gpu_count per node, summed over nodes.
        - training_strategy: newest recorded strategy, "" when none.
        - first_data_ts / last_data_ts: oldest and newest sample clocks.
        """
        facts = empty_context()
        try:
            row = conn.execute(
                "SELECT world_size, gpu_count, hostname FROM system_samples "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is not None:
                facts["world_size"] = int(row[0] or 0)
                facts["gpu_count"] = int(row[1] or 0)
                facts["hostname"] = str(row[2] or "")
        except sqlite3.Error:
            pass
        try:
            # Per-rank clocks: a rank is reporting while its newest sample is
            # within three ticks of the newest sample anywhere. Time, not
            # seq: seq counters start when each rank starts, so two nodes
            # that came up 20 s apart differ by 10 seq for the whole run.
            # The tick is estimated from the fastest rank's own cadence so a
            # slow sampler is never called dead between its own ticks.
            rows = conn.execute(
                """
                SELECT MIN(sample_ts_s), MAX(sample_ts_s), COUNT(*)
                FROM process_samples
                WHERE sample_ts_s IS NOT NULL
                GROUP BY COALESCE(global_rank, rank)
                """
            ).fetchall()
            if rows:
                newest = max(float(r[1]) for r in rows)
                cadences = [
                    (float(r[1]) - float(r[0])) / (int(r[2]) - 1)
                    for r in rows
                    if int(r[2]) > 1 and float(r[1]) > float(r[0])
                ]
                tick = max(_TICK_SEC, min(cadences)) if cadences else _TICK_SEC
                window = 3.0 * tick
                facts["ranks_reporting"] = sum(
                    1 for r in rows if float(r[1]) >= newest - window
                )
            else:
                facts["ranks_reporting"] = 0
        except sqlite3.Error:
            pass
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT hostname) FROM system_samples "
                "WHERE hostname IS NOT NULL AND hostname != ''"
            ).fetchone()
            facts["node_count"] = int(row[0]) if row else None
            # GPUs observed across the node set: each node's newest
            # gpu_count, summed. One node's count alone under-reports a
            # multi-node run (2 nodes x 1 GPU is 2 GPUs, not 1).
            row = conn.execute(
                """
                SELECT SUM(gpu_count) FROM (
                    SELECT gpu_count FROM system_samples s
                    WHERE id = (
                        SELECT MAX(id) FROM system_samples t
                        WHERE t.hostname IS s.hostname
                    )
                )
                """
            ).fetchone()
            if row and row[0] is not None:
                facts["gpus_observed"] = int(row[0])
        except sqlite3.Error:
            pass
        try:
            row = conn.execute(
                """
                SELECT training_strategy FROM runtime_environment
                WHERE training_strategy IS NOT NULL AND training_strategy != ''
                ORDER BY id DESC LIMIT 1
                """
            ).fetchone()
            facts["training_strategy"] = str(row[0]) if row else ""
        except sqlite3.Error:
            pass
        try:
            sys_row = conn.execute(
                "SELECT MIN(sample_ts_s), MAX(sample_ts_s) FROM system_samples"
            ).fetchone()
            proc_row = conn.execute(
                "SELECT MAX(sample_ts_s) FROM process_samples"
            ).fetchone()
            firsts = [v for v in (sys_row[0],) if v is not None]
            lasts = [v for v in (sys_row[1], proc_row[0]) if v is not None]
            facts["first_data_ts"] = float(min(firsts)) if firsts else None
            facts["last_data_ts"] = float(max(lasts)) if lasts else None
        except sqlite3.Error:
            pass
        return facts
