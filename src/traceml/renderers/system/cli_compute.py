"""CLI compute for system telemetry."""

import time
from typing import Any, Dict, Optional

from .common import SystemCLISnapshot, SystemMetricsDB


class SystemCLIComputer:
    """Compute the latest CLI system telemetry snapshot."""

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

    def compute(self) -> Dict[str, Any]:
        """
        Compute the latest CLI snapshot.

        Returns cached values on transient failure if they are still within the
        configured stale TTL. Otherwise returns the default empty payload.
        """
        try:
            with self._db.connect() as conn:
                out = self._compute_impl(conn)
        except Exception:
            return self._return_stale()

        self._last_ok = out
        self._last_ok_ts = time.time()
        return out

    def _compute_impl(self, conn) -> Dict[str, Any]:
        latest = self._db.fetch_latest_system_sample(conn)
        if latest is None:
            return self._empty_snapshot()

        gpu_rows = self._db.fetch_gpu_rows_for_sample(
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

            util_min: Optional[float] = None
            util_max: Optional[float] = None
            headroom_min: Optional[float] = None
            headroom_min_idx: Optional[int] = None

            for idx, g in enumerate(gpu_rows):
                util = float(g["util"] or 0.0)
                mem_used = float(g["mem_used_bytes"] or 0.0)
                mem_total = float(g["mem_total_bytes"] or 0.0)

                util_total += util
                mem_used_total += mem_used
                mem_total_total += mem_total

                util_min = util if util_min is None else min(util_min, util)
                util_max = util if util_max is None else max(util_max, util)

                if mem_total > 0.0:
                    headroom = max(mem_total - mem_used, 0.0)
                    if headroom_min is None or headroom < headroom_min:
                        headroom_min = headroom
                        headroom_min_idx = (
                            int(g["gpu_idx"])
                            if g["gpu_idx"] is not None
                            else idx
                        )

                temp_val = float(g["temperature_c"] or 0.0)
                if temp_val > temp_max:
                    temp_max = temp_val

                power_total += float(g["power_usage_w"] or 0.0)
                power_limit_total += float(g["power_limit_w"] or 0.0)

            gpu_util_skew = (
                util_max - util_min
                if util_min is not None and util_max is not None
                else None
            )
        else:
            util_total = None
            mem_used_total = None
            mem_total_total = None
            temp_max = None
            power_total = None
            power_limit_total = None
            gpu_util_skew = None
            headroom_min = None
            headroom_min_idx = None

        return SystemCLISnapshot(
            cpu=float(latest["cpu_percent"] or 0.0),
            ram_used=float(latest["ram_used_bytes"] or 0.0),
            ram_total=float(latest["ram_total_bytes"] or 0.0),
            gpu_available=bool(latest["gpu_available"] or False),
            gpu_count=int(latest["gpu_count"] or 0),
            gpu_util_total=util_total,
            gpu_util_skew=gpu_util_skew,
            gpu_mem_used=mem_used_total,
            gpu_mem_total=mem_total_total,
            gpu_mem_headroom_min=headroom_min,
            gpu_mem_headroom_min_idx=headroom_min_idx,
            gpu_temp_max=temp_max,
            gpu_power_usage=power_total,
            gpu_power_limit=power_limit_total,
        ).to_dict()

    def _return_stale(self) -> Dict[str, Any]:
        now = time.time()
        if self._last_ok is not None:
            if (
                self._stale_ttl_s is None
                or (now - self._last_ok_ts) <= self._stale_ttl_s
            ):
                return self._last_ok
        return self._empty_snapshot()

    def _empty_snapshot(self) -> Dict[str, Any]:
        return SystemCLISnapshot(
            cpu=0.0,
            ram_used=0.0,
            ram_total=0.0,
            gpu_available=False,
            gpu_count=0,
            gpu_util_total=None,
            gpu_util_skew=None,
            gpu_mem_used=None,
            gpu_mem_total=None,
            gpu_mem_headroom_min=None,
            gpu_mem_headroom_min_idx=None,
            gpu_temp_max=None,
            gpu_power_usage=None,
            gpu_power_limit=None,
        ).to_dict()
