"""Shared models and SQLite helpers for system telemetry."""

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


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
    rank:
        Optional rank filter. When set, all reads are restricted to that rank.

    Notes
    -----
    One short-lived connection per public compute call is preferred here:
    it keeps thread behavior simple and avoids long-lived SQLite state.
    """

    def __init__(self, db_path: str, rank: Optional[int] = None) -> None:
        self._db_path = str(db_path)
        self._rank = rank

    def connect(self) -> sqlite3.Connection:
        """
        Open a short-lived SQLite read connection.
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def rank_filter(self) -> tuple[str, tuple]:
        """
        Return SQL WHERE fragment and bound params for rank filtering.
        """
        if self._rank is None:
            return "", ()
        return "WHERE rank = ?", (int(self._rank),)

    def fetch_latest_system_sample(
        self,
        conn: sqlite3.Connection,
    ) -> Optional[sqlite3.Row]:
        """
        Fetch the latest system sample for the configured rank filter.
        """
        where_sql, params = self.rank_filter()
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
        where_sql, params = self.rank_filter()
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

    def fetch_gpu_rows_for_sample(
        self,
        conn: sqlite3.Connection,
        *,
        rank: Optional[int],
        seq: Optional[int],
    ) -> List[sqlite3.Row]:
        """
        Fetch GPU rows for one exact system sample.

        Sample identity is matched by (rank, seq). This is valid for the
        current SystemSampler because `seq` is monotonic per rank.
        """
        if seq is None:
            return []

        if rank is None:
            sql = """
                SELECT *
                FROM system_gpu_samples
                WHERE rank IS NULL
                  AND seq = ?
                ORDER BY gpu_idx ASC;
            """
            params = (int(seq),)
        else:
            sql = """
                SELECT *
                FROM system_gpu_samples
                WHERE rank = ?
                  AND seq = ?
                ORDER BY gpu_idx ASC;
            """
            params = (int(rank), int(seq))

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
            List of (rank, seq) keys identifying system samples.

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

        non_null_rank_keys = [
            (int(rank), int(seq))
            for rank, seq in sample_keys
            if rank is not None
        ]
        null_rank_seqs = [
            int(seq) for rank, seq in sample_keys if rank is None
        ]

        clauses: List[str] = []
        params: List[Any] = []

        if non_null_rank_keys:
            pair_clause = ",".join("(?, ?)" for _ in non_null_rank_keys)
            clauses.append(f"(rank, seq) IN ({pair_clause})")
            for rank, seq in non_null_rank_keys:
                params.extend([rank, seq])

        if null_rank_seqs:
            seq_clause = ",".join("?" for _ in null_rank_seqs)
            clauses.append(f"(rank IS NULL AND seq IN ({seq_clause}))")
            params.extend(null_rank_seqs)

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
    def group_gpu_rows_by_rank_seq(
        rows: List[sqlite3.Row],
    ) -> Dict[Tuple[Optional[int], int], List[sqlite3.Row]]:
        """
        Group GPU rows by (rank, seq) for fast per-sample lookup.

        This avoids repeated scans of the GPU row list during dashboard compute.
        """
        out: Dict[Tuple[Optional[int], int], List[sqlite3.Row]] = {}
        for row in rows:
            seq = row["seq"]
            if seq is None:
                continue
            key = (row["rank"], int(seq))
            out.setdefault(key, []).append(row)
        return out
