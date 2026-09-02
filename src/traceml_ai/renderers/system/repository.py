# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Every SQLite read for system telemetry, and nothing else.

Moved verbatim out of ``common.py`` so the System block has the same three
part shape the Process block already has: this file reads, the compute
layer judges, the card draws. Splitting them is what makes it possible to
say where a number was decided.

The whole-run reads take a ``RunSeriesPlan`` rather than deciding the
window themselves: choosing how much history a chart shows is a dashboard
decision, and this file exists so that decision is visibly somewhere else.
"""

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from traceml_ai.renderers.shared.run_series import RunSeriesPlan

# A row without a clock cannot be placed on a chart or counted toward a
# cadence. `sample_ts_s` is nullable in the writer's schema, so this is a
# real state, not a defensive check: every time-series read requires it.
_HAS_CLOCK_SQL = "sample_ts_s IS NOT NULL"

# An all-zero capacity row is the sampler's NVML failure fallback, not a
# real 0 W observation. One predicate, applied by every query that reads
# power, so the span a chart covers and the values it draws agree about
# which rows exist.
_GPU_REPORTED_SQL = """
    power_usage_w IS NOT NULL
    AND (
        COALESCE(mem_total_bytes, 0) > 0
        OR COALESCE(power_limit_w, 0) > 0
    )
"""


@dataclass(frozen=True)
class RunStats:
    """The shape of one history, for planning a read of it.

    Mirrors ``renderers/process/repository.RunStats``. System has no
    per-rank counts because its dashboard is single-node by construction:
    the computer picks one host and every read is filtered to it.
    """

    first_ts: float
    last_ts: float
    sample_count: int

    @property
    def span_s(self) -> float:
        return max(0.0, self.last_ts - self.first_ts)


class SystemRepository:
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
        hostname: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        """
        Fetch the most recent system samples in ascending time order.

        The inner query limits the read size first, then the outer query
        restores ascending order for downstream time-series compute.
        ``hostname`` narrows the window to one node: system telemetry is
        per machine, and a window that interleaves two hosts is not a
        series of anything.
        """
        where_sql, params = self.node_rank_filter()
        bound: List[Any] = list(params)
        if hostname is not None:
            # IS, not =: a NULL hostname must round-trip as NULL.
            where_sql = (
                f"{where_sql} AND hostname IS ?"
                if where_sql
                else "WHERE hostname IS ?"
            )
            bound.append(str(hostname))
        sql = f"""
            SELECT *
            FROM (
                SELECT *
                FROM system_samples
                {where_sql}
                {'AND' if where_sql else 'WHERE'} {_HAS_CLOCK_SQL}
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC;
        """
        return conn.execute(sql, (*bound, int(limit))).fetchall()

    def _run_scope(self, hostname: Optional[str]) -> Tuple[str, List[Any]]:
        """The node filter every whole-run read shares."""
        where_sql, params = self.node_rank_filter()
        bound: List[Any] = list(params)
        if hostname is not None:
            where_sql = (
                f"{where_sql} AND hostname IS ?"
                if where_sql
                else "WHERE hostname IS ?"
            )
            bound.append(str(hostname))
        return where_sql, bound

    def cpu_run_stats(
        self, conn: sqlite3.Connection, hostname: Optional[str] = None
    ) -> Optional[RunStats]:
        """The shape of the whole-run CPU history, for planning a read."""
        where_sql, bound = self._run_scope(hostname)
        try:
            row = conn.execute(
                "SELECT MIN(sample_ts_s), MAX(sample_ts_s), COUNT(*) "
                f"FROM system_samples {where_sql} "
                f"{'AND' if where_sql else 'WHERE'} {_HAS_CLOCK_SQL}",
                tuple(bound),
            ).fetchone()
        except sqlite3.Error:
            return None
        if not row or row[0] is None or row[1] is None:
            return None
        return RunStats(
            first_ts=float(row[0]),
            last_ts=float(row[1]),
            sample_count=int(row[2] or 0),
        )

    def fetch_cpu_run(
        self,
        conn: sqlite3.Connection,
        plan: RunSeriesPlan,
        hostname: Optional[str] = None,
    ) -> List[Tuple[float, float, float]]:
        """Whole-run host CPU, rolled and decimated by ``plan``.

        The frame comes from the plan, so on SQLite 3.28 and later this is
        a RANGE window measured in seconds rather than a count of rows. The
        two agree only when sampling is perfectly regular; across a gap the
        row frame reaches further back in wall clock than the label claims.
        """
        where_sql, bound = self._run_scope(hostname)
        try:
            rows = conn.execute(
                f"""
                WITH rolled AS (
                    SELECT
                        sample_ts_s AS t,
                        AVG(cpu_percent) OVER w AS a,
                        MAX(cpu_percent) OVER w AS m,
                        ROW_NUMBER() OVER (ORDER BY sample_ts_s) AS rn
                    FROM system_samples
                    {where_sql}
                    {'AND' if where_sql else 'WHERE'} cpu_percent IS NOT NULL
                      AND {_HAS_CLOCK_SQL}
                    WINDOW w AS (
                        ORDER BY sample_ts_s
                        {plan.frame_clause()}
                    )
                )
                SELECT t, a, m FROM rolled
                WHERE rn % ? = 0 AND rn > ?
                ORDER BY t ASC
                """,
                (*bound, int(plan.stride), int(plan.preceding_rows)),
            ).fetchall()
        except sqlite3.Error:
            # Window functions need SQLite >= 3.25; without them the whole
            # run view is simply unavailable and the window view stands.
            return []
        return [(float(r[0]), float(r[1]), float(r[2])) for r in rows]

    def gpu_power_run_stats(
        self, conn: sqlite3.Connection, hostname: Optional[str] = None
    ) -> Optional[RunStats]:
        """The shape of the whole-run power history, over reported rows."""
        where_sql, bound = self._run_scope(hostname)
        try:
            row = conn.execute(
                "SELECT MIN(sample_ts_s), MAX(sample_ts_s), COUNT(*) FROM "
                f"system_gpu_samples {where_sql} "
                f"{'AND' if where_sql else 'WHERE'} {_GPU_REPORTED_SQL} "
                f"AND {_HAS_CLOCK_SQL}",
                tuple(bound),
            ).fetchone()
        except sqlite3.Error:
            return None
        if not row or row[0] is None or row[1] is None:
            return None
        return RunStats(
            first_ts=float(row[0]),
            last_ts=float(row[1]),
            sample_count=int(row[2] or 0),
        )

    def fetch_gpu_power_run(
        self,
        conn: sqlite3.Connection,
        *,
        width_s: float,
        first_ts: float,
        hostname: Optional[str] = None,
    ) -> List[Tuple[int, float, float, float, float]]:
        """Whole-run per-GPU power in fixed-duration buckets.

        This path BUCKETS rather than rolls, so it takes a bucket width
        rather than a ``RunSeriesPlan``: there is no stride, cadence or
        point budget for the shared planner to supply. The width is the
        same duration the rolling charts use, which is the only part of
        the plan that applies here.
        """
        where_sql, bound = self._run_scope(hostname)
        try:
            return [
                (
                    int(r[0]),
                    float(r[1]),
                    float(r[2]),
                    float(r[3]),
                    float(r[4]),
                )
                for r in conn.execute(
                    f"""
                    SELECT
                        gpu_idx,
                        MIN(sample_ts_s),
                        AVG(power_usage_w),
                        MIN(power_usage_w),
                        MAX(power_usage_w)
                    FROM system_gpu_samples
                    {where_sql}
                    {'AND' if where_sql else 'WHERE'} {_GPU_REPORTED_SQL}
                      AND {_HAS_CLOCK_SQL}
                    GROUP BY gpu_idx, CAST((sample_ts_s - ?) / ? AS INTEGER)
                    ORDER BY gpu_idx ASC, 2 ASC
                    """,
                    (*bound, float(first_ts), float(width_s)),
                ).fetchall()
            ]
        except sqlite3.Error:
            return []

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
