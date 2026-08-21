# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Shared models and SQLite helpers for system telemetry."""

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Whole-run charts smooth with a ROLLING window, the way the metrics
# notebook plots them, not with disjoint buckets. Consecutive points then
# share most of their samples, so the line is smooth; disjoint buckets each
# catch a different phase of a fast oscillation and stay jagged (measured:
# mean absolute change per point 0.48 raw, 0.02 rolling over 30 samples).
# The window is a duration, so it means the same thing at any cadence.
_ROLL_MIN_S = 30.0
_ROLL_MAX_S = 300.0
_ROLL_FRACTION = 50.0  # about a fiftieth of the run
_MAX_RUN_POINTS = 120


def choose_window_s(span_s: float) -> float:
    """The rolling window for a run of ``span_s`` seconds, in round steps."""
    if span_s <= 0:
        return _ROLL_MIN_S
    raw = max(_ROLL_MIN_S, min(_ROLL_MAX_S, span_s / _ROLL_FRACTION))
    for step in (30.0, 60.0, 120.0, 300.0):
        if raw <= step:
            return step
    return _ROLL_MAX_S


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
    # Series includes both flat sample arrays and structured whole-run
    # histories, so its values are intentionally heterogeneous.
    series: Dict[str, Any]

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
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC;
        """
        return conn.execute(sql, (*bound, int(limit))).fetchall()

    def fetch_cpu_run_history(
        self,
        conn: sqlite3.Connection,
        hostname: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Whole-run host CPU, decimated to ``buckets`` equal time slices.

        The window series answers "what is CPU doing now"; this answers
        "what has it done over the run", which is the only form that can
        show a drift (the 96-minute reference run climbs about 25%).
        Decimation happens in SQL so the payload stays a fixed size no
        matter how long the run is: one row per slice carrying the slice's
        mean and its max, because a mean alone would erase the spikes that
        a stalling dataloader produces.
        """
        empty: Dict[str, Any] = {
            "t": [],
            "avg": [],
            "max": [],
            "span_s": 0.0,
            "window_s": 0.0,
        }
        where_sql, params = self.node_rank_filter()
        bound: List[Any] = list(params)
        if hostname is not None:
            where_sql = (
                f"{where_sql} AND hostname IS ?"
                if where_sql
                else "WHERE hostname IS ?"
            )
            bound.append(str(hostname))
        try:
            row = conn.execute(
                f"SELECT MIN(sample_ts_s), MAX(sample_ts_s) FROM "
                f"system_samples {where_sql}",
                tuple(bound),
            ).fetchone()
        except sqlite3.Error:
            return empty
        if not row or row[0] is None or row[1] is None:
            return empty
        first, last = float(row[0]), float(row[1])
        span = last - first
        if span <= 0:
            return empty
        window_s = choose_window_s(span)
        count = 0
        try:
            row = conn.execute(
                f"SELECT COUNT(*) FROM system_samples {where_sql}",
                tuple(bound),
            ).fetchone()
            count = int(row[0]) if row else 0
        except sqlite3.Error:
            return empty
        if count < 2:
            return empty
        cadence = span / max(count - 1, 1)
        preceding = max(1, int(round(window_s / max(cadence, 1e-6))) - 1)
        # The first ``preceding`` samples are valid telemetry, but they do not
        # yet cover a complete rolling window and are excluded by the query
        # below. Calculate the stride from only the remaining chart-eligible
        # points: this guarantees the 120-point limit without unnecessarily
        # discarding detail when the eligible series already fits.
        eligible_count = max(0, count - preceding)
        stride = max(
            1,
            (eligible_count + _MAX_RUN_POINTS - 1) // _MAX_RUN_POINTS,
        )
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
                    WINDOW w AS (
                        ORDER BY sample_ts_s
                        ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW
                    )
                )
                SELECT t, a, m FROM rolled
                WHERE rn % ? = 0 AND rn > {preceding}
                ORDER BY t ASC
                """,
                (*bound, stride),
            ).fetchall()
        except sqlite3.Error:
            # Window functions need SQLite >= 3.25; without them the whole
            # run view is simply unavailable and the window view stands.
            return empty
        return {
            "t": [float(r[0]) for r in rows],
            "avg": [float(r[1]) for r in rows],
            "max": [float(r[2]) for r in rows],
            "span_s": span,
            "window_s": window_s,
        }

    def fetch_gpu_power_run_history(
        self,
        conn: sqlite3.Connection,
        hostname: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Whole-run per-GPU power: each bucket's mean, floor and peak.

        Buckets are disjoint, unlike the rolling window the CPU series
        uses, and deliberately: a rolling MIN or MAX holds one extreme
        sample across the whole window and draws square plateaus, while a
        per-bucket extreme varies with the run.

        The display pairs the MEAN with the FLOOR, because that is the
        pair that answers the questions this tool exists for. Sustained
        draw well under the board limit means the GPU is not being fed
        (measured on our own corpus: an input-straggler run averages
        45.9 W against a compute-bound run's 61.9 W), and a floor that
        falls to idle means the GPU went idle inside that window, which is
        what a dataloader stall looks like in watts. The peak is kept for
        callers that want it, but it barely moves on healthy work.
        """
        where_sql, params = self.node_rank_filter()
        bound: List[Any] = list(params)
        if hostname is not None:
            where_sql = (
                f"{where_sql} AND hostname IS ?"
                if where_sql
                else "WHERE hostname IS ?"
            )
            bound.append(str(hostname))
        try:
            row = conn.execute(
                f"SELECT MIN(sample_ts_s), MAX(sample_ts_s) FROM "
                f"system_gpu_samples {where_sql}",
                tuple(bound),
            ).fetchone()
        except sqlite3.Error:
            return []
        if not row or row[0] is None or row[1] is None:
            return []
        first, last = float(row[0]), float(row[1])
        span = last - first
        if span <= 0:
            return []
        width = choose_window_s(span)
        try:
            rows = conn.execute(
                f"""
                SELECT
                    gpu_idx,
                    MIN(sample_ts_s),
                    AVG(power_usage_w),
                    MIN(power_usage_w),
                    MAX(power_usage_w)
                FROM system_gpu_samples
                {where_sql}
                {'AND' if where_sql else 'WHERE'} power_usage_w IS NOT NULL
                GROUP BY gpu_idx, CAST((sample_ts_s - ?) / ? AS INTEGER)
                ORDER BY gpu_idx ASC, 2 ASC
                """,
                (*bound, first, width),
            ).fetchall()
        except sqlite3.Error:
            return []
        by_gpu: Dict[int, Dict[str, Any]] = {}
        for gpu_idx, ts, avg, mn, mx in rows:
            e = by_gpu.setdefault(
                int(gpu_idx),
                {
                    "gpu_idx": int(gpu_idx),
                    "t": [],
                    "avg": [],
                    "min": [],
                    "max": [],
                },
            )
            e["t"].append(float(ts))
            e["avg"].append(float(avg))
            e["min"].append(float(mn))
            e["max"].append(float(mx))
        out = [by_gpu[i] for i in sorted(by_gpu)]
        for e in out:
            e["span_s"] = span
            e["window_s"] = width
        return out

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
