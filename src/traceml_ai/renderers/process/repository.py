# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""SQLite reads for process telemetry.

Moved here unchanged from ``common.py`` so the read side has one home and
the modules above it can be read without SQL in the way.
"""

import sqlite3
from typing import Dict, List, Optional


class ProcessRepository:
    """
    The only place `process_samples` is read.

    This layer answers questions about rows and nothing else. It does not
    choose dashboard windows, freshness thresholds, cache durations,
    diagnostic thresholds, tile estimators, or disclosure behavior: those
    are decisions about what the numbers MEAN, and they belong to the
    compute layer that calls this one.

    Both the terminal and the dashboard read through here, so a change to
    the table has exactly one place to land.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)

    def connect(self) -> sqlite3.Connection:
        """
        Open a short-lived SQLite connection configured for named-row access.
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def fetch_latest_seq(self, conn: sqlite3.Connection) -> Optional[int]:
        """
        Return the latest sequence number visible in `process_samples`.

        Returns
        -------
        Optional[int]
            Latest seq, or None if the table has no seq-bearing rows.
        """
        row = conn.execute(
            """
            SELECT seq
            FROM process_samples
            WHERE seq IS NOT NULL
            ORDER BY id DESC
            LIMIT 1;
            """
        ).fetchone()
        if row is None or row["seq"] is None:
            return None
        return int(row["seq"])

    def fetch_latest_seq_per_rank(
        self, conn: sqlite3.Connection
    ) -> Dict[int, int]:
        """
        Return the latest seq observed for each rank.

        Returns
        -------
        dict[int, int]
            Mapping rank -> latest seq for that rank.
        """
        rows = conn.execute(
            """
            SELECT rank, MAX(seq) AS max_seq
            FROM process_samples
            WHERE rank IS NOT NULL
              AND seq IS NOT NULL
            GROUP BY rank
            ORDER BY rank ASC;
            """
        ).fetchall()

        out: Dict[int, int] = {}
        for row in rows:
            if row["rank"] is None or row["max_seq"] is None:
                continue
            out[int(row["rank"])] = int(row["max_seq"])
        return out

    def fetch_rows_for_seq_all_ranks(
        self,
        conn: sqlite3.Connection,
        seq: int,
    ) -> List[sqlite3.Row]:
        """
        Fetch all rows for one exact seq across all ranks.

        Parameters
        ----------
        seq:
            Sequence number to read.

        Returns
        -------
        list[sqlite3.Row]
            Rows for that seq ordered by rank then id.
        """
        return conn.execute(
            """
            SELECT *
            FROM process_samples
            WHERE seq = ?
            ORDER BY rank ASC, id ASC;
            """,
            (int(seq),),
        ).fetchall()

    def fetch_committed_seq(self, conn: sqlite3.Connection) -> Optional[int]:
        """
        Return the latest seq completed by all active ranks.

        Semantics
        ---------
        Equivalent to the old in-memory logic:
        committed_seq = min(last_seq_per_rank.values())

        Returns
        -------
        Optional[int]
            Latest globally committed seq, or None if no active ranks exist.
        """
        per_rank = self.fetch_latest_seq_per_rank(conn)
        if not per_rank:
            return None
        return min(per_rank.values())

    def fetch_seq_range_aggregates(
        self,
        conn: sqlite3.Connection,
        start_seq: int,
        end_seq: int,
    ) -> List[sqlite3.Row]:
        """
        Aggregate dashboard history over a contiguous committed seq range.

        This query preserves the previous dashboard semantics:

        - one output row per seq
        - CPU = max(cpu_percent) across ranks
        - RAM = max(ram_used_bytes) across ranks
        - RAM total = max(ram_total_bytes) across ranks
        - GPU candidate chosen from the rank with least headroom
          where headroom = gpu_mem_total_bytes - gpu_mem_reserved_bytes
        - GPU used imbalance = max(gpu_mem_used_bytes) - min(gpu_mem_used_bytes)

        Parameters
        ----------
        start_seq:
            Inclusive sequence lower bound.
        end_seq:
            Inclusive sequence upper bound.

        Returns
        -------
        list[sqlite3.Row]
            One aggregated row per seq, ascending by seq.
        """
        if end_seq < start_seq:
            return []

        return conn.execute(
            """
            WITH seq_rows AS (
                SELECT *
                FROM process_samples
                WHERE seq BETWEEN ? AND ?
            ),
            seq_base AS (
                SELECT
                    seq,
                    MAX(cpu_percent) AS cpu_max,
                    MAX(ram_used_bytes) AS ram_used_max,
                    MAX(ram_total_bytes) AS ram_total,
                    MAX(sample_ts_s) AS sample_ts_s
                FROM seq_rows
                GROUP BY seq
            ),
            gpu_candidates AS (
                SELECT
                    seq,
                    rank,
                    gpu_mem_used_bytes AS gpu_used,
                    gpu_mem_total_bytes AS gpu_total,
                    (gpu_mem_total_bytes - gpu_mem_reserved_bytes) AS gpu_headroom,
                    ROW_NUMBER() OVER (
                        PARTITION BY seq
                        ORDER BY (gpu_mem_total_bytes - gpu_mem_reserved_bytes) ASC,
                                 rank ASC,
                                 id ASC
                    ) AS rn
                FROM seq_rows
                WHERE gpu_available = 1
                  AND gpu_mem_used_bytes IS NOT NULL
                  AND gpu_mem_reserved_bytes IS NOT NULL
                  AND gpu_mem_total_bytes IS NOT NULL
            ),
            gpu_choice AS (
                SELECT
                    seq,
                    rank AS gpu_rank,
                    gpu_used,
                    gpu_total,
                    gpu_headroom
                FROM gpu_candidates
                WHERE rn = 1
            ),
            gpu_imbalance AS (
                SELECT
                    seq,
                    CASE
                        WHEN COUNT(gpu_mem_used_bytes) > 0
                        THEN MAX(gpu_mem_used_bytes) - MIN(gpu_mem_used_bytes)
                        ELSE NULL
                    END AS gpu_used_imbalance
                FROM seq_rows
                WHERE gpu_mem_used_bytes IS NOT NULL
                GROUP BY seq
            )
            SELECT
                b.seq,
                b.sample_ts_s,
                b.cpu_max,
                b.ram_used_max,
                b.ram_total,
                g.gpu_used,
                g.gpu_total,
                g.gpu_headroom,
                g.gpu_rank,
                gi.gpu_used_imbalance
            FROM seq_base b
            LEFT JOIN gpu_choice g
                ON b.seq = g.seq
            LEFT JOIN gpu_imbalance gi
                ON b.seq = gi.seq
            ORDER BY b.seq ASC;
            """,
            (int(start_seq), int(end_seq)),
        ).fetchall()
