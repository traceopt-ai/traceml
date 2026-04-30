"""Dashboard compute for system telemetry."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

from .common import SystemDashboardPayload, SystemMetricsDB


class SystemDashboardComputer:
    """Compute dashboard rollups and short time-series."""

    def __init__(
        self,
        db_path: str,
        rank: Optional[int] = None,
        stale_ttl_s: Optional[float] = 30.0,
    ) -> None:
        self._db = SystemMetricsDB(db_path=db_path, rank=rank)
        self._last_ok: Optional[Dict[str, Any]] = None
        self._last_ok_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )

    def compute(self, window_n: int = 100) -> Dict[str, Any]:
        """
        Compute dashboard rollups plus short series over the latest window.

        Returns cached values on transient failure or empty window if they are
        still within the configured stale TTL. Otherwise returns the default
        empty payload.
        """
        try:
            with self._db.connect() as conn:
                out = self._compute_impl(conn, window_n=max(1, int(window_n)))
        except Exception as e:
            return self._return_stale(f"STALE (exception: {type(e).__name__})")

        if out.get("window_len", 0) == 0 and self._last_ok is not None:
            return self._return_stale("STALE (empty window)")

        self._last_ok = out
        self._last_ok_ts = time.time()
        return out

    def _compute_impl(self, conn, window_n: int) -> Dict[str, Any]:
        """
        Compute the dashboard payload from recent SQLite rows.

        Notes
        -----
        - `sample_ts_s` is used as the canonical sample timestamp.
        - `x_time` is emitted as ISO-8601 UTC strings so the UI can plot a
          real time axis instead of sample indices or relative negative values.
        """
        samples = self._db.fetch_recent_system_samples(conn, limit=window_n)
        if not samples:
            return self._empty_payload()

        last = samples[-1]
        gpu_available = bool(last["gpu_available"] or False)

        ts_hist = np.array(
            [float(r["sample_ts_s"] or 0.0) for r in samples],
            dtype=np.float64,
        )
        cpu_hist = np.array(
            [float(r["cpu_percent"] or 0.0) for r in samples],
            dtype=np.float64,
        )
        ram_used_hist = np.array(
            [float(r["ram_used_bytes"] or 0.0) for r in samples],
            dtype=np.float64,
        )
        ram_total = float(last["ram_total_bytes"] or 0.0)

        gpu_rows = self._db.fetch_gpu_rows_for_samples(
            conn,
            sample_keys=[
                (sample["rank"], sample["seq"])
                for sample in samples
                if sample["seq"] is not None
            ],
        )
        gpu_rows_by_key = self._db.group_gpu_rows_by_rank_seq(gpu_rows)

        n = len(samples)
        gpu_avg = np.zeros(n, dtype=np.float64)
        gpu_delta = np.zeros(n, dtype=np.float64)
        gpu_mem_worst = np.zeros(n, dtype=np.float64)
        gpu_mem_headroom_min = np.zeros(n, dtype=np.float64)
        temp_max = np.zeros(n, dtype=np.float64)

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

                headrooms = [
                    max(mt - mu, 0.0)
                    for mu, mt in zip(mem_useds, mem_totals)
                    if mt > 0.0
                ]
                gpu_mem_headroom_min[i] = min(headrooms) if headrooms else 0.0
            else:
                gpu_avg[i] = 0.0
                gpu_delta[i] = 0.0
                gpu_mem_worst[i] = 0.0
                gpu_mem_headroom_min[i] = 0.0
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
                "headroom": float(gpu_mem_headroom_min[-1]),
            },
            "temp": {
                "now": temp_now,
                "p95": temp_p95,
                "status": temp_status,
            },
        }

        x_time = [self._format_time_iso(ts) for ts in ts_hist.tolist()]

        return SystemDashboardPayload(
            window_len=len(samples),
            gpu_available=gpu_available,
            rollups=rollups,
            series={
                "x_time": x_time,
                "cpu": cpu_hist.astype(float).tolist(),
                "gpu_avg": (
                    gpu_avg.astype(float).tolist() if gpu_available else []
                ),
            },
        ).to_dict()

    def _format_time_iso(self, ts_s: float) -> str:
        """
        Convert one UNIX timestamp in seconds to an ISO-8601 UTC string.

        Returns an empty string on invalid input so callers can safely degrade.
        """
        try:
            if ts_s <= 0.0:
                return ""
            return datetime.fromtimestamp(
                float(ts_s), tz=timezone.utc
            ).isoformat()
        except Exception:
            return ""

    def _return_stale(self, msg: str) -> Dict[str, Any]:
        """
        Return the last known good payload when it is still within TTL.

        Adds a human-readable status string into `rollups["status"]`.
        """
        now = time.time()
        if self._last_ok is not None:
            if (
                self._stale_ttl_s is None
                or (now - self._last_ok_ts) <= self._stale_ttl_s
            ):
                cached = self._last_ok
                rollups = dict(cached.get("rollups", {}))
                rollups["status"] = msg
                return {
                    "window_len": cached.get("window_len", 0),
                    "gpu_available": cached.get("gpu_available", False),
                    "rollups": rollups,
                    "series": cached.get(
                        "series",
                        {"x_time": [], "cpu": [], "gpu_avg": []},
                    ),
                }

        return {
            "window_len": 0,
            "gpu_available": False,
            "rollups": {"status": "No fresh system data"},
            "series": {"x_time": [], "cpu": [], "gpu_avg": []},
        }

    def _empty_payload(self) -> Dict[str, Any]:
        """
        Return an empty dashboard payload with the full expected schema.
        """
        return SystemDashboardPayload(
            window_len=0,
            gpu_available=False,
            rollups={},
            series={"x_time": [], "cpu": [], "gpu_avg": []},
        ).to_dict()
