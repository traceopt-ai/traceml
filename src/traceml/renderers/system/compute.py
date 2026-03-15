import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SystemCLISnapshot:
    """
    Compact CLI snapshot for system telemetry.

    Output schema is intentionally unchanged so existing rendering code can
    continue to work without modification.
    """

    cpu: float
    ram_used: float
    ram_total: float

    gpu_available: bool
    gpu_count: int

    gpu_util_total: Optional[float]
    gpu_mem_used: Optional[float]
    gpu_mem_total: Optional[float]

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
        }


@dataclass(frozen=True)
class SystemDashboardPayload:
    """
    Dashboard payload for system telemetry.

    Output schema is intentionally unchanged so existing rendering code can
    continue to work without modification.
    """

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


class SystemMetricsComputer:
    """
    Compute system-level metrics directly from SQLite projection tables.

    Data source
    -----------
    Reads from:
    - system_samples
    - system_gpu_samples

    Output compatibility
    --------------------
    The output dict structure is intentionally kept identical to the previous
    in-memory implementation so existing CLI/dashboard rendering code can be
    reused without modification.

    Design goals
    ------------
    - Keep compute fast with bounded SQL reads
    - Preserve stale-cache behavior to avoid UI flicker / blanking
    - Keep logic easy to read and maintain
    - Avoid dependence on RemoteDBStore / raw sampler payloads

    Notes
    -----
    - Memory values remain in raw bytes because that is what existing renderers
      expect from this compute layer.
    - GPU rows are matched to system samples using (rank, seq), which is valid
      for the current SystemSampler because `seq` is monotonic per rank.
    """

    def __init__(
        self,
        db_path: str,
        rank: Optional[int] = None,
        stale_ttl_s: Optional[float] = 30.0,
    ) -> None:
        self._db_path = str(db_path)
        self._rank = rank

        self._last_ok_cli: Optional[Dict[str, Any]] = None
        self._last_ok_dash: Optional[Dict[str, Any]] = None
        self._last_ok_cli_ts: float = 0.0
        self._last_ok_dash_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )

    def compute_cli(self) -> Dict[str, Any]:
        """
        Compute the latest CLI snapshot.

        Returns cached values on transient failure if they are still within the
        configured stale TTL. Otherwise returns the default empty payload.
        """
        try:
            with self._connect() as conn:
                out = self._compute_cli_impl(conn)
        except Exception:
            return self._return_stale_cli()

        self._last_ok_cli = out
        self._last_ok_cli_ts = time.time()
        return out

    def compute_dashboard(self, window_n: int = 100) -> Dict[str, Any]:
        """
        Compute dashboard rollups + short series over the latest window.

        Returns cached values on transient failure or empty window if they are
        still within the configured stale TTL. Otherwise returns the default
        empty payload.
        """
        try:
            with self._connect() as conn:
                out = self._compute_dashboard_impl(
                    conn, window_n=max(1, int(window_n))
                )
        except Exception as e:
            return self._return_stale_dash(
                f"STALE (exception: {type(e).__name__})"
            )

        if out.get("window_len", 0) == 0 and self._last_ok_dash is not None:
            return self._return_stale_dash("STALE (empty window)")

        self._last_ok_dash = out
        self._last_ok_dash_ts = time.time()
        return out

    # ------------------------------------------------------------------
    # CLI compute
    # ------------------------------------------------------------------

    def _compute_cli_impl(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        latest = self._fetch_latest_system_sample(conn)
        if latest is None:
            return self._empty_cli_snapshot()

        gpu_rows = self._fetch_gpu_rows_for_sample(
            conn,
            rank=latest["rank"],
            seq=latest["seq"],
        )

        if gpu_rows:
            util_total = 0.0
            mem_used_total = 0.0
            mem_total_total = 0.0
            temp_max = 0.0
            power_total = 0.0
            power_limit_total = 0.0

            for g in gpu_rows:
                util_total += float(g["util"] or 0.0)
                mem_used_total += float(g["mem_used_bytes"] or 0.0)
                mem_total_total += float(g["mem_total_bytes"] or 0.0)

                temp_val = float(g["temperature_c"] or 0.0)
                if temp_val > temp_max:
                    temp_max = temp_val

                power_total += float(g["power_usage_w"] or 0.0)
                power_limit_total += float(g["power_limit_w"] or 0.0)
        else:
            util_total = None
            mem_used_total = None
            mem_total_total = None
            temp_max = None
            power_total = None
            power_limit_total = None

        return SystemCLISnapshot(
            cpu=float(latest["cpu_percent"] or 0.0),
            ram_used=float(latest["ram_used_bytes"] or 0.0),
            ram_total=float(latest["ram_total_bytes"] or 0.0),
            gpu_available=bool(latest["gpu_available"] or False),
            gpu_count=int(latest["gpu_count"] or 0),
            gpu_util_total=util_total,
            gpu_mem_used=mem_used_total,
            gpu_mem_total=mem_total_total,
            gpu_temp_max=temp_max,
            gpu_power_usage=power_total,
            gpu_power_limit=power_limit_total,
        ).to_dict()

    def _return_stale_cli(self) -> Dict[str, Any]:
        now = time.time()
        if self._last_ok_cli is not None:
            if (
                self._stale_ttl_s is None
                or (now - self._last_ok_cli_ts) <= self._stale_ttl_s
            ):
                return self._last_ok_cli
        return self._empty_cli_snapshot()

    def _empty_cli_snapshot(self) -> Dict[str, Any]:
        return SystemCLISnapshot(
            cpu=0.0,
            ram_used=0.0,
            ram_total=0.0,
            gpu_available=False,
            gpu_count=0,
            gpu_util_total=None,
            gpu_mem_used=None,
            gpu_mem_total=None,
            gpu_temp_max=None,
            gpu_power_usage=None,
            gpu_power_limit=None,
        ).to_dict()

    # ------------------------------------------------------------------
    # Dashboard compute
    # ------------------------------------------------------------------

    def _compute_dashboard_impl(
        self,
        conn: sqlite3.Connection,
        window_n: int,
    ) -> Dict[str, Any]:
        samples = self._fetch_recent_system_samples(conn, limit=window_n)
        if not samples:
            return self._empty_dashboard_payload()

        last = samples[-1]
        gpu_available = bool(last["gpu_available"] or False)

        cpu_hist = np.array(
            [float(r["cpu_percent"] or 0.0) for r in samples],
            dtype=np.float64,
        )
        ram_used_hist = np.array(
            [float(r["ram_used_bytes"] or 0.0) for r in samples],
            dtype=np.float64,
        )
        ram_total = float(last["ram_total_bytes"] or 0.0)

        # Bulk-fetch all GPU rows for this window once, then group in Python.
        gpu_rows = self._fetch_gpu_rows_for_samples(
            conn,
            sample_keys=[
                (sample["rank"], sample["seq"])
                for sample in samples
                if sample["seq"] is not None
            ],
        )
        gpu_rows_by_key = self._group_gpu_rows_by_rank_seq(gpu_rows)

        n = len(samples)
        gpu_avg = np.zeros(n, dtype=np.float64)
        gpu_delta = np.zeros(n, dtype=np.float64)
        gpu_mem_worst = np.zeros(n, dtype=np.float64)
        temp_max = np.zeros(n, dtype=np.float64)

        max_gpu_capacity = 0.0

        for i, sample in enumerate(samples):
            key = (sample["rank"], sample["seq"])
            rows = gpu_rows_by_key.get(key, [])

            if rows:
                utils = [float(g["util"] or 0.0) for g in rows]
                mem_useds = [float(g["mem_used_bytes"] or 0.0) for g in rows]
                mem_totals = [float(g["mem_total_bytes"] or 0.0) for g in rows]
                temps = [float(g["temperature_c"] or 0.0) for g in rows]

                gpu_avg[i] = sum(utils) / float(len(utils))
                gpu_delta[i] = max(utils) - min(utils)
                gpu_mem_worst[i] = max(mem_useds)
                temp_max[i] = max(temps)

                if i == n - 1 and mem_totals:
                    max_gpu_capacity = max(mem_totals)
            else:
                gpu_avg[i] = 0.0
                gpu_delta[i] = 0.0
                gpu_mem_worst[i] = 0.0
                temp_max[i] = 0.0

        cpu_p50 = float(np.percentile(cpu_hist, 50)) if cpu_hist.size else 0.0
        cpu_p95 = float(np.percentile(cpu_hist, 95)) if cpu_hist.size else 0.0

        ram_p95 = (
            float(np.percentile(ram_used_hist, 95))
            if ram_used_hist.size
            else 0.0
        )

        gpu_p50 = float(np.percentile(gpu_avg, 50)) if gpu_avg.size else 0.0
        gpu_p95 = float(np.percentile(gpu_avg, 95)) if gpu_avg.size else 0.0

        delta_p95 = (
            float(np.percentile(gpu_delta, 95)) if gpu_delta.size else 0.0
        )

        mem_p95 = (
            float(np.percentile(gpu_mem_worst, 95))
            if gpu_mem_worst.size
            else 0.0
        )
        temp_p95 = float(np.percentile(temp_max, 95)) if temp_max.size else 0.0

        temp_now = float(temp_max[-1]) if temp_max.size else 0.0
        temp_status = (
            "Hot" if temp_now >= 85 else "Warm" if temp_now >= 80 else "OK"
        )

        rollups = {
            "gpu_available": gpu_available,
            "cpu": {
                "now": float(cpu_hist[-1]),
                "p50": cpu_p50,
                "p95": cpu_p95,
            },
            "ram": {
                "now": float(ram_used_hist[-1]),
                "p95": ram_p95,
                "total": ram_total,
                "headroom": max(ram_total - float(ram_used_hist[-1]), 0.0),
            },
            "gpu_util": {
                "now": float(gpu_avg[-1]),
                "p50": gpu_p50,
                "p95": gpu_p95,
            },
            "gpu_delta": {
                "now": float(gpu_delta[-1]),
                "p95": delta_p95,
            },
            "gpu_mem": {
                "now": float(gpu_mem_worst[-1]),
                "p95": mem_p95,
                "headroom": max(
                    max_gpu_capacity - float(gpu_mem_worst[-1]), 0.0
                ),
            },
            "temp": {
                "now": temp_now,
                "p95": temp_p95,
                "status": temp_status,
            },
        }

        return SystemDashboardPayload(
            window_len=len(samples),
            gpu_available=gpu_available,
            rollups=rollups,
            series={
                "cpu": cpu_hist.astype(float).tolist(),
                "gpu_avg": (
                    gpu_avg.astype(float).tolist() if gpu_available else []
                ),
            },
        ).to_dict()

    def _return_stale_dash(self, msg: str) -> Dict[str, Any]:
        now = time.time()
        if self._last_ok_dash is not None:
            if (
                self._stale_ttl_s is None
                or (now - self._last_ok_dash_ts) <= self._stale_ttl_s
            ):
                cached = self._last_ok_dash
                rollups = dict(cached.get("rollups", {}))
                rollups["status"] = msg
                return {
                    "window_len": cached.get("window_len", 0),
                    "gpu_available": cached.get("gpu_available", False),
                    "rollups": rollups,
                    "series": cached.get("series", {"cpu": [], "gpu_avg": []}),
                }

        return {
            "window_len": 0,
            "gpu_available": False,
            "rollups": {"status": "No fresh system data"},
            "series": {"cpu": [], "gpu_avg": []},
        }

    def _empty_dashboard_payload(self) -> Dict[str, Any]:
        return SystemDashboardPayload(
            window_len=0,
            gpu_available=False,
            rollups={},
            series={"cpu": [], "gpu_avg": []},
        ).to_dict()

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """
        Open a short-lived read connection.

        One connection per public compute call keeps the implementation simple,
        avoids cross-thread SQLite issues, and reduces connection churn
        compared to opening a new connection inside every helper.
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _rank_filter(self) -> tuple[str, tuple]:
        if self._rank is None:
            return "", ()
        return "WHERE rank = ?", (int(self._rank),)

    def _fetch_latest_system_sample(
        self,
        conn: sqlite3.Connection,
    ) -> Optional[sqlite3.Row]:
        where_sql, params = self._rank_filter()
        sql = f"""
            SELECT *
            FROM system_samples
            {where_sql}
            ORDER BY id DESC
            LIMIT 1;
        """
        return conn.execute(sql, params).fetchone()

    def _fetch_recent_system_samples(
        self,
        conn: sqlite3.Connection,
        limit: int,
    ) -> List[sqlite3.Row]:
        where_sql, params = self._rank_filter()
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

    def _fetch_gpu_rows_for_sample(
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

    def _fetch_gpu_rows_for_samples(
        self,
        conn: sqlite3.Connection,
        sample_keys: List[Tuple[Optional[int], int]],
    ) -> List[sqlite3.Row]:
        """
        Bulk-fetch GPU rows for many samples in one query.

        Parameters
        ----------
        sample_keys:
            List of (rank, seq) keys identifying samples.

        Returns
        -------
        list[sqlite3.Row]
            Matching rows from `system_gpu_samples`.
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

    def _group_gpu_rows_by_rank_seq(
        self,
        rows: List[sqlite3.Row],
    ) -> Dict[Tuple[Optional[int], int], List[sqlite3.Row]]:
        """
        Group GPU rows by (rank, seq) for fast per-sample lookup during
        dashboard compute.
        """
        out: Dict[Tuple[Optional[int], int], List[sqlite3.Row]] = {}
        for row in rows:
            seq = row["seq"]
            if seq is None:
                continue
            key = (row["rank"], int(seq))
            out.setdefault(key, []).append(row)
        return out
