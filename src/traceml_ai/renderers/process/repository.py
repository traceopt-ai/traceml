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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from traceml_ai.renderers.shared.run_series import RunSeriesPlan

# The two whole-run metrics this table can serve, as SQL the repository
# owns. They are named constants rather than parameters on a generic
# method: a reader of this class can see every expression it will ever
# evaluate, and no caller can hand it one.
_CPU_CAPACITY_SQL = (
    "cpu_percent / (100.0 * NULLIF(cpu_logical_core_count, 0)) * 100.0"
)
_RSS_SQL = "ram_used_bytes"


@dataclass(frozen=True)
class RunStats:
    """The shape of one metric's history, for planning a read of it."""

    first_ts: float
    last_ts: float
    sample_count: int
    rank_count: int
    max_samples_per_rank: int

    @property
    def span_s(self) -> float:
        return max(0.0, self.last_ts - self.first_ts)

    @property
    def samples_per_rank(self) -> int:
        """The BUSIEST rank's row count, not the average across ranks.

        The stride is applied per rank, so planning it on an average
        overshoots the point budget on any rank with more rows than the
        mean. Measured on a four-rank run where one rank died early:
        planning on the average produced 145 points against a budget of
        120.
        """
        return max(1, self.max_samples_per_rank)


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

    # --- per-rank reads --------------------------------------------------
    def newest_sample_ts(self, conn: sqlite3.Connection) -> Optional[float]:
        """The newest sample clock, read once per tick and reused.

        Every time bound below derives from it. Deriving them separately
        would let the retention pruner delete rows between two statements
        and leave one computation reading a window the other never saw.
        """
        row = conn.execute(
            "SELECT MAX(sample_ts_s) FROM process_samples"
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def fetch_recent_rank_window(
        self,
        conn: sqlite3.Connection,
        window_n: int = 100,
        newest_ts: Optional[float] = None,
        max_age_s: float = 20.0 * 60.0,
    ) -> List[sqlite3.Row]:
        """The last ``window_n`` samples of EVERY rank, newest last.

        Per rank, not globally. A rank that stopped reporting keeps its own
        history instead of being squeezed out by livelier peers, and a rank
        that never reports does not shrink everyone else's window.

        Bounded by time before the partition runs: without it the
        ROW_NUMBER scans every row ever written in order to rank the last
        hundred, and that scan grows with the run.
        """
        floor_ts = None
        if newest_ts is not None:
            floor_ts = newest_ts - max(60.0, float(max_age_s))
        return conn.execute(
            """
            WITH recent AS (
                SELECT * FROM process_samples
                WHERE COALESCE(global_rank, rank) IS NOT NULL
                  AND (? IS NULL OR sample_ts_s >= ?)
            ),
            ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(global_rank, rank)
                        ORDER BY seq DESC, id DESC
                    ) AS rn
                FROM recent
            )
            SELECT * FROM ranked
            WHERE rn <= ?
            ORDER BY COALESCE(global_rank, rank) ASC, seq ASC, id ASC;
            """,
            (floor_ts, floor_ts, int(max(1, window_n))),
        ).fetchall()

    def fetch_rank_latest(self, conn: sqlite3.Connection) -> List[sqlite3.Row]:
        """The newest row of EVERY rank, however long ago it arrived.

        The windowed read above is time-bounded, so a rank silent for
        longer than that bound has no rows in it and would drop out of the
        block entirely. The surface would forget a dead rank a few minutes
        after it died, which is exactly when its death starts to matter.
        This read answers "who has ever reported, and when did each last
        speak".
        """
        return conn.execute(
            """
            SELECT * FROM process_samples
            WHERE id IN (
                SELECT MAX(id) FROM process_samples
                WHERE COALESCE(global_rank, rank) IS NOT NULL
                GROUP BY COALESCE(global_rank, rank)
            )
            ORDER BY COALESCE(global_rank, rank) ASC;
            """
        ).fetchall()

    # --- whole-run reads, one explicit method per metric -----------------
    def cpu_capacity_run_stats(
        self, conn: sqlite3.Connection
    ) -> Optional[RunStats]:
        """The shape of the CPU-capacity history, for planning a read."""
        return self._run_stats(conn, _CPU_CAPACITY_SQL)

    def rss_run_stats(self, conn: sqlite3.Connection) -> Optional[RunStats]:
        """The shape of the RSS history, for planning a read."""
        return self._run_stats(conn, _RSS_SQL)

    def fetch_cpu_capacity_run(
        self, conn: sqlite3.Connection, plan: RunSeriesPlan
    ) -> List[Tuple[int, float, float, float]]:
        """Whole-run CPU capacity per rank, rolled and decimated by ``plan``."""
        return self._run_history(conn, _CPU_CAPACITY_SQL, plan)

    def fetch_rss_run(
        self, conn: sqlite3.Connection, plan: RunSeriesPlan
    ) -> List[Tuple[int, float, float, float]]:
        """Whole-run RSS per rank, rolled and decimated by ``plan``."""
        return self._run_history(conn, _RSS_SQL, plan)

    # --- the shared implementation of the two above ----------------------
    def _run_stats(
        self, conn: sqlite3.Connection, value_sql: str
    ) -> Optional[RunStats]:
        row = conn.execute(
            f"""
            WITH per_rank AS (
                SELECT
                    COUNT(*) AS n,
                    MIN(sample_ts_s) AS lo,
                    MAX(sample_ts_s) AS hi
                FROM process_samples
                WHERE sample_ts_s IS NOT NULL
                  AND ({value_sql}) IS NOT NULL
                  AND COALESCE(global_rank, rank) IS NOT NULL
                GROUP BY COALESCE(global_rank, rank)
            )
            SELECT MIN(lo), MAX(hi), SUM(n), COUNT(*), MAX(n)
            FROM per_rank;
            """
        ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            return None
        return RunStats(
            first_ts=float(row[0]),
            last_ts=float(row[1]),
            sample_count=int(row[2] or 0),
            rank_count=max(1, int(row[3] or 1)),
            max_samples_per_rank=max(1, int(row[4] or 1)),
        )

    def _run_history(
        self,
        conn: sqlite3.Connection,
        value_sql: str,
        plan: RunSeriesPlan,
    ) -> List[Tuple[int, float, float, float]]:
        """Execute one planned whole-run read.

        Private, and the only place a SQL expression is interpolated. The
        two public methods above each pass one of this module's own
        constants, so no expression reaches here from outside the class.
        """
        return [
            (int(r[0]), float(r[1]), float(r[2]), float(r[3]))
            for r in conn.execute(
                f"""
                WITH base AS (
                    SELECT
                        COALESCE(global_rank, rank) AS rank_id,
                        sample_ts_s AS ts,
                        ({value_sql}) AS v,
                        ROW_NUMBER() OVER (
                            PARTITION BY COALESCE(global_rank, rank)
                            ORDER BY sample_ts_s ASC, id ASC
                        ) AS rn
                    FROM process_samples
                    WHERE sample_ts_s IS NOT NULL
                      AND ({value_sql}) IS NOT NULL
                      AND COALESCE(global_rank, rank) IS NOT NULL
                ),
                rolled AS (
                    SELECT
                        rank_id, ts, rn,
                        AVG(v) OVER (
                            PARTITION BY rank_id ORDER BY ts
                            {plan.frame_clause()}
                        ) AS roll_avg,
                        MAX(v) OVER (
                            PARTITION BY rank_id ORDER BY ts
                            {plan.frame_clause()}
                        ) AS roll_max
                    FROM base
                )
                SELECT rank_id, ts, roll_avg, roll_max
                FROM rolled
                WHERE rn % ? = 0 AND rn > ?
                ORDER BY rank_id ASC, ts ASC;
                """,
                (int(plan.stride), int(plan.preceding_rows)),
            ).fetchall()
        ]


__all__ = ["ProcessRepository", "RunStats"]
