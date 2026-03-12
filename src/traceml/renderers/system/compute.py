import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SystemCLISnapshot:
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

    Stability
    ---------
    - On exception or transient empty compute, returns previous payload
      if it is still within `stale_ttl_s`.
    - If no cached payload is available, returns the same empty/default
      payload shape as before.
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
        try:
            out = self._compute_cli_impl()
        except Exception as e:
            return self._return_stale_cli(
                f"STALE (exception: {type(e).__name__})"
            )

        self._last_ok_cli = out
        self._last_ok_cli_ts = time.time()
        return out

    def _return_stale_cli(self, msg: str) -> Dict[str, Any]:
        now = time.time()
        if self._last_ok_cli is not None:
            if (
                self._stale_ttl_s is None
                or (now - self._last_ok_cli_ts) <= self._stale_ttl_s
            ):
                return self._last_ok_cli

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

    def _compute_cli_impl(self) -> Dict[str, Any]:
        latest = self._fetch_latest_system_sample()
        if latest is None:
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

        gpu_rows = self._fetch_gpu_rows_for_sample(
            rank=latest["rank"],
            seq=latest["seq"],
            sample_ts_s=latest["sample_ts_s"],
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

        snap = SystemCLISnapshot(
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
        )
        return snap.to_dict()

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def compute_dashboard(self, window_n: int = 100) -> Dict[str, Any]:
        try:
            out = self._compute_dashboard_impl(window_n=window_n)
        except Exception as e:
            return self._return_stale_dash(
                f"STALE (exception: {type(e).__name__})"
            )

        if out.get("window_len", 0) == 0 and self._last_ok_dash is not None:
            return self._return_stale_dash("STALE (empty window)")

        self._last_ok_dash = out
        self._last_ok_dash_ts = time.time()
        return out

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

        return SystemDashboardPayload(
            window_len=0,
            gpu_available=False,
            rollups={"status": "No fresh system data"},
            series={"cpu": [], "gpu_avg": []},
        ).to_dict()

    def _compute_dashboard_impl(self, window_n: int) -> Dict[str, Any]:
        samples = self._fetch_recent_system_samples(limit=int(window_n))
        if not samples:
            return SystemDashboardPayload(
                window_len=0,
                gpu_available=False,
                rollups={},
                series={"cpu": [], "gpu_avg": []},
            ).to_dict()

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

        n = len(samples)
        gpu_avg = np.zeros(n, dtype=np.float64)
        gpu_delta = np.zeros(n, dtype=np.float64)
        gpu_mem_worst = np.zeros(n, dtype=np.float64)
        temp_max = np.zeros(n, dtype=np.float64)

        max_gpu_capacity = 0.0

        for i, sample in enumerate(samples):
            gpu_rows = self._fetch_gpu_rows_for_sample(
                rank=sample["rank"],
                seq=sample["seq"],
                sample_ts_s=sample["sample_ts_s"],
            )

            if gpu_rows:
                utils = [float(g["util"] or 0.0) for g in gpu_rows]
                mem_useds = [
                    float(g["mem_used_bytes"] or 0.0) for g in gpu_rows
                ]
                mem_totals = [
                    float(g["mem_total_bytes"] or 0.0) for g in gpu_rows
                ]
                temps = [float(g["temperature_c"] or 0.0) for g in gpu_rows]

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

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """
        Open a short-lived read connection.

        A fresh connection per compute call keeps the class simple and avoids
        cross-thread SQLite issues.
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _rank_filter(self) -> tuple[str, tuple]:
        if self._rank is None:
            return "", ()
        return "WHERE rank = ?", (int(self._rank),)

    def _fetch_latest_system_sample(self) -> Optional[sqlite3.Row]:
        where_sql, params = self._rank_filter()
        sql = f"""
            SELECT *
            FROM system_samples
            {where_sql}
            ORDER BY id DESC
            LIMIT 1;
        """
        with self._connect() as conn:
            return conn.execute(sql, params).fetchone()

    def _fetch_recent_system_samples(self, limit: int) -> List[sqlite3.Row]:
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
        with self._connect() as conn:
            return conn.execute(sql, (*params, int(limit))).fetchall()

    def _fetch_gpu_rows_for_sample(
        self,
        *,
        rank: Optional[int],
        seq: Optional[int],
        sample_ts_s: Optional[float],
    ) -> List[sqlite3.Row]:
        """
        Fetch GPU rows for one exact system sample.

        The system projection schema does not currently store a direct
        foreign-key link from `system_gpu_samples` to `system_samples`, so
        the sample identity is matched by:
        - rank
        - seq
        - sample_ts_s
        """
        if seq is None or sample_ts_s is None:
            return []

        if rank is None:
            sql = """
                SELECT *
                FROM system_gpu_samples
                WHERE rank IS NULL
                  AND seq = ?
                  AND sample_ts_s = ?
                ORDER BY gpu_idx ASC;
            """
            params = (int(seq), float(sample_ts_s))
        else:
            sql = """
                SELECT *
                FROM system_gpu_samples
                WHERE rank = ?
                  AND seq = ?
                  AND sample_ts_s = ?
                ORDER BY gpu_idx ASC;
            """
            params = (int(rank), int(seq), float(sample_ts_s))

        with self._connect() as conn:
            return conn.execute(sql, params).fetchall()
