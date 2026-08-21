# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Shared models and SQLite helpers for system telemetry."""

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Default sampler cadence; the floor for the rank-coverage tick estimate.
_TICK_SEC = 2.0


@dataclass(frozen=True)
class SystemCLISnapshot:
    """Compact CLI snapshot for system telemetry."""

    cpu: float
    ram_used: float
    ram_total: float

    gpu_available: bool
    gpu_count: int

    gpu_util_total: Optional[float]
    gpu_util_skew: Optional[float]
    gpu_mem_used: Optional[float]
    gpu_mem_total: Optional[float]
    gpu_mem_headroom_min: Optional[float]
    gpu_mem_headroom_min_idx: Optional[int]

    gpu_temp_max: Optional[float]
    gpu_power_usage: Optional[float]
    gpu_power_limit: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu": self.cpu,
            "ram_used": self.ram_used,
            "ram_total": self.ram_total,
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count,
            "gpu_util_total": self.gpu_util_total,
            "gpu_mem_used": self.gpu_mem_used,
            "gpu_mem_total": self.gpu_mem_total,
            "gpu_temp_max": self.gpu_temp_max,
            "gpu_power_usage": self.gpu_power_usage,
            "gpu_power_limit": self.gpu_power_limit,
            "gpu_util_skew": self.gpu_util_skew,
            "gpu_mem_headroom_min": self.gpu_mem_headroom_min,
            "gpu_mem_headroom_min_idx": self.gpu_mem_headroom_min_idx,
        }


@dataclass(frozen=True)
class SystemDashboardPayload:
    """Dashboard payload for system telemetry."""

    window_len: int
    gpu_available: bool
    rollups: Dict[str, Any]
    series: Dict[str, List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_len": self.window_len,
            "gpu_available": self.gpu_available,
            "rollups": self.rollups,
            "series": self.series,
        }


class SystemMetricsDB:
    """
    Shared SQLite access helper for system telemetry compute.

    This class centralizes all SQLite reads used by both CLI and dashboard
    compute layers. It keeps the implementation simple and avoids duplicating
    query logic across files.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    node_rank:
        Optional node-rank filter. System telemetry is node-level, so filtered
        reads are restricted to this distributed node identity.

    Notes
    -----
    One short-lived connection per public compute call is preferred here:
    it keeps thread behavior simple and avoids long-lived SQLite state.
    """

    def __init__(
        self,
        db_path: str,
        node_rank: Optional[int] = None,
    ) -> None:
        self._db_path = str(db_path)
        self._node_rank = node_rank

    def connect(self) -> sqlite3.Connection:
        """
        Open a short-lived SQLite read connection.
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def node_rank_filter(self) -> tuple[str, tuple]:
        """
        Return SQL WHERE fragment and bound params for node-rank filtering.
        """
        if self._node_rank is None:
            return "", ()
        return "WHERE node_rank = ?", (int(self._node_rank),)

    def fetch_latest_system_sample(
        self,
        conn: sqlite3.Connection,
    ) -> Optional[sqlite3.Row]:
        """
        Fetch the latest system sample for the configured node filter.
        """
        where_sql, params = self.node_rank_filter()
        sql = f"""
            SELECT *
            FROM system_samples
            {where_sql}
            ORDER BY id DESC
            LIMIT 1;
        """
        return conn.execute(sql, params).fetchone()

    def fetch_recent_system_samples(
        self,
        conn: sqlite3.Connection,
        limit: int,
    ) -> List[sqlite3.Row]:
        """
        Fetch the most recent system samples in ascending time order.

        The inner query limits the read size first, then the outer query
        restores ascending order for downstream time-series compute.
        """
        where_sql, params = self.node_rank_filter()
        sql = f"""
            SELECT *
            FROM (
                SELECT *
                FROM system_samples
                {where_sql}
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC;
        """
        return conn.execute(sql, (*params, int(limit))).fetchall()

    def fetch_context_facts(
        self,
        conn: sqlite3.Connection,
    ) -> Dict[str, Any]:
        """
        Facts for the dashboard's context strip: what is reporting, and how
        long it has been reporting. Every value is observed, never configured.

        - ranks_reporting: ranks whose newest sample is within three ticks
          of the newest sample anywhere (tick = the fastest rank's cadence,
          floored at the default sampler interval). Time, never seq: seq
          counters carry each rank's start offset.
        - node_count: distinct hostnames that produced system samples.
        - gpus_observed: newest gpu_count per node, summed over nodes.
        - training_strategy: newest recorded strategy, "" when none.
        - first_data_ts / last_data_ts: oldest and newest sample clocks.

        Each query degrades to None on its own, so an older database that
        lacks a table never takes the whole strip down.
        """
        facts: Dict[str, Any] = {
            "ranks_reporting": None,
            "node_count": None,
            "gpus_observed": None,
            "training_strategy": "",
            "first_data_ts": None,
            "last_data_ts": None,
        }
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

    def fetch_gpu_rows_for_sample(
        self,
        conn: sqlite3.Connection,
        *,
        global_rank: Optional[int],
        seq: Optional[int],
    ) -> List[sqlite3.Row]:
        """
        Fetch GPU rows for one exact system sample.

        Sample identity is matched by (global_rank, seq), which is unique for
        multi-node jobs because `seq` is monotonic within each worker.
        """
        if seq is None:
            return []

        if global_rank is None:
            sql = """
                SELECT *
                FROM system_gpu_samples
                WHERE global_rank IS NULL
                  AND seq = ?
                ORDER BY gpu_idx ASC;
            """
            params = (int(seq),)
        else:
            sql = """
                SELECT *
                FROM system_gpu_samples
                WHERE global_rank = ?
                  AND seq = ?
                ORDER BY gpu_idx ASC;
            """
            params = (int(global_rank), int(seq))

        return conn.execute(sql, params).fetchall()

    def fetch_gpu_rows_for_samples(
        self,
        conn: sqlite3.Connection,
        sample_keys: List[Tuple[Optional[int], int]],
    ) -> List[sqlite3.Row]:
        """
        Bulk-fetch GPU rows for many samples in one query.

        Parameters
        ----------
        sample_keys:
            List of (global_rank, seq) keys identifying system samples.

        Returns
        -------
        list[sqlite3.Row]
            Matching rows from `system_gpu_samples`.

        Notes
        -----
        This performs one bounded bulk read for the full dashboard window,
        which is faster than issuing one GPU query per sample.
        """
        if not sample_keys:
            return []

        non_null_global_rank_keys = [
            (int(global_rank), int(seq))
            for global_rank, seq in sample_keys
            if global_rank is not None
        ]
        null_global_rank_seqs = [
            int(seq) for global_rank, seq in sample_keys if global_rank is None
        ]

        clauses: List[str] = []
        params: List[Any] = []

        if non_null_global_rank_keys:
            pair_clause = ",".join("(?, ?)" for _ in non_null_global_rank_keys)
            clauses.append(f"(global_rank, seq) IN ({pair_clause})")
            for global_rank, seq in non_null_global_rank_keys:
                params.extend([global_rank, seq])

        if null_global_rank_seqs:
            seq_clause = ",".join("?" for _ in null_global_rank_seqs)
            clauses.append(f"(global_rank IS NULL AND seq IN ({seq_clause}))")
            params.extend(null_global_rank_seqs)

        if not clauses:
            return []

        sql = f"""
            SELECT *
            FROM system_gpu_samples
            WHERE {" OR ".join(clauses)}
            ORDER BY seq ASC, gpu_idx ASC;
        """
        return conn.execute(sql, tuple(params)).fetchall()

    @staticmethod
    def group_gpu_rows_by_global_rank_seq(
        rows: List[sqlite3.Row],
    ) -> Dict[Tuple[Optional[int], int], List[sqlite3.Row]]:
        """
        Group GPU rows by (global_rank, seq) for fast per-sample lookup.

        This avoids repeated scans of the GPU row list during dashboard compute.
        """
        out: Dict[Tuple[Optional[int], int], List[sqlite3.Row]] = {}
        for row in rows:
            seq = row["seq"]
            if seq is None:
                continue
            key = (row["global_rank"], int(seq))
            out.setdefault(key, []).append(row)
        return out
