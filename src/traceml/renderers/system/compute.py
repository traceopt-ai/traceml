import time
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    Compute system-level metrics from the system telemetry table.

    Stability:
    - last-good cache: on exception or transient empty compute, return previous payload
      with "STALE (...)" status embedded inside rollups (no schema change).
    - optional TTL: after stale_ttl_s seconds without a successful compute, return empty.

    NOTE: Output dict structure remains the same keys as your current code.
    """

    def __init__(self, table: Any, *, stale_ttl_s: Optional[float] = 30.0) -> None:
        self._table: Sequence[Any] = table or ()
        self._last_ok_cli: Optional[Dict[str, Any]] = None
        self._last_ok_dash: Optional[Dict[str, Any]] = None
        self._last_ok_cli_ts: float = 0.0
        self._last_ok_dash_ts: float = 0.0
        self._stale_ttl_s: Optional[float] = float(stale_ttl_s) if stale_ttl_s is not None else None

    # ---------
    # CLI
    # ---------

    def compute_cli(self) -> Dict[str, Any]:
        try:
            out = self._compute_cli_impl()
        except Exception as e:
            return self._return_stale_cli(f"STALE (exception: {type(e).__name__})")

        # If output is a valid dict, accept as "ok"
        self._last_ok_cli = out
        self._last_ok_cli_ts = time.time()
        return out

    def _return_stale_cli(self, msg: str) -> Dict[str, Any]:
        now = time.time()
        if self._last_ok_cli is not None:
            if self._stale_ttl_s is None or (now - self._last_ok_cli_ts) <= self._stale_ttl_s:
                # can't add new fields; return cached dict as-is
                return self._last_ok_cli
        # fallback: your original empty snapshot
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
        if not self._table:
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

        latest = self._table[-1]
        gpus = _gpus_safe(latest)

        if gpus:
            util_total = 0.0
            mem_used_total = 0.0
            mem_total_total = 0.0
            temp_max = 0.0
            power_total = 0.0
            power_limit_total = 0.0

            for g in gpus:
                util_total += g[0]
                mem_used_total += g[1]
                mem_total_total += g[2]
                if g[3] > temp_max:
                    temp_max = g[3]
                power_total += g[4]
                power_limit_total += g[5]
        else:
            util_total = mem_used_total = mem_total_total = None
            temp_max = power_total = power_limit_total = None

        snap = SystemCLISnapshot(
            cpu=float(latest.get("cpu", 0.0) or 0.0),
            ram_used=float(latest.get("ram_used", 0.0) or 0.0),
            ram_total=float(latest.get("ram_total", 0.0) or 0.0),
            gpu_available=bool(latest.get("gpu_available", False)),
            gpu_count=int(latest.get("gpu_count", 0) or 0),
            gpu_util_total=util_total,
            gpu_mem_used=mem_used_total,
            gpu_mem_total=mem_total_total,
            gpu_temp_max=temp_max,
            gpu_power_usage=power_total,
            gpu_power_limit=power_limit_total,
        )
        return snap.to_dict()

    # -------------
    # Dashboard
    # -------------

    def compute_dashboard(self, window_n: int = 100) -> Dict[str, Any]:
        try:
            out = self._compute_dashboard_impl(window_n=window_n)
        except Exception as e:
            return self._return_stale_dash(f"STALE (exception: {type(e).__name__})")

        # If transiently empty window, return stale so UI doesn't blank
        if out.get("window_len", 0) == 0 and self._last_ok_dash is not None:
            return self._return_stale_dash("STALE (empty window)")

        self._last_ok_dash = out
        self._last_ok_dash_ts = time.time()
        return out

    def _return_stale_dash(self, msg: str) -> Dict[str, Any]:
        now = time.time()
        if self._last_ok_dash is not None:
            if self._stale_ttl_s is None or (now - self._last_ok_dash_ts) <= self._stale_ttl_s:
                # can't change schema, but we CAN safely annotate rollups with a status string
                cached = self._last_ok_dash
                rollups = dict(cached.get("rollups", {}))
                rollups["status"] = msg  # does not break existing keys; add is optional
                return {
                    "window_len": cached.get("window_len", 0),
                    "gpu_available": cached.get("gpu_available", False),
                    "rollups": rollups,
                    "series": cached.get("series", {"cpu": [], "gpu_avg": []}),
                }

        # too stale or no cache
        payload = SystemDashboardPayload(
            window_len=0,
            gpu_available=False,
            rollups={"status": "No fresh system data"},
            series={"cpu": [], "gpu_avg": []},
        )
        return payload.to_dict()

    def _compute_dashboard_impl(self, window_n: int) -> Dict[str, Any]:
        window = _last_n_fast(self._table, int(window_n))
        if not window:
            payload = SystemDashboardPayload(
                window_len=0,
                gpu_available=False,
                rollups={},
                series={"cpu": [], "gpu_avg": []},
            )
            return payload.to_dict()

        last = window[-1]
        gpu_available = bool(last.get("gpu_available", False))

        # Build histories using numpy arrays (faster percentiles)
        cpu_hist = np.fromiter((float(r.get("cpu", 0.0) or 0.0) for r in window), dtype=np.float64)
        ram_used_hist = np.fromiter((float(r.get("ram_used", 0.0) or 0.0) for r in window), dtype=np.float64)
        ram_total = float(last.get("ram_total", 0.0) or 0.0)

        n = len(window)
        gpu_avg = np.zeros(n, dtype=np.float64)
        gpu_delta = np.zeros(n, dtype=np.float64)
        gpu_mem_worst = np.zeros(n, dtype=np.float64)
        temp_max = np.zeros(n, dtype=np.float64)

        for i, r in enumerate(window):
            gpus = _gpus_safe(r)
            if gpus:
                # util avg + util delta
                utils = [g[0] for g in gpus]
                umin = utils[0]
                umax = utils[0]
                usum = 0.0
                for u in utils:
                    usum += u
                    if u < umin:
                        umin = u
                    if u > umax:
                        umax = u
                gpu_avg[i] = usum / float(len(utils))
                gpu_delta[i] = umax - umin

                # worst mem_used
                mmax = gpus[0][1]
                tmax = gpus[0][3]
                for g in gpus:
                    if g[1] > mmax:
                        mmax = g[1]
                    if g[3] > tmax:
                        tmax = g[3]
                gpu_mem_worst[i] = mmax
                temp_max[i] = tmax
            else:
                gpu_avg[i] = 0.0
                gpu_delta[i] = 0.0
                gpu_mem_worst[i] = 0.0
                temp_max[i] = 0.0

        # capacity: latest sample max mem_total across GPUs
        last_gpus = _gpus_safe(last)
        max_gpu_capacity = 0.0
        for g in last_gpus:
            if g[2] > max_gpu_capacity:
                max_gpu_capacity = g[2]

        # percentiles (numpy)
        cpu_p50 = float(np.percentile(cpu_hist, 50)) if cpu_hist.size else 0.0
        cpu_p95 = float(np.percentile(cpu_hist, 95)) if cpu_hist.size else 0.0

        ram_p95 = float(np.percentile(ram_used_hist, 95)) if ram_used_hist.size else 0.0

        gpu_p50 = float(np.percentile(gpu_avg, 50)) if gpu_avg.size else 0.0
        gpu_p95 = float(np.percentile(gpu_avg, 95)) if gpu_avg.size else 0.0

        delta_p95 = float(np.percentile(gpu_delta, 95)) if gpu_delta.size else 0.0

        mem_p95 = float(np.percentile(gpu_mem_worst, 95)) if gpu_mem_worst.size else 0.0
        temp_p95 = float(np.percentile(temp_max, 95)) if temp_max.size else 0.0

        temp_now = float(temp_max[-1]) if temp_max.size else 0.0
        temp_status = "Hot" if temp_now >= 85 else "Warm" if temp_now >= 80 else "OK"

        rollups = {
            "gpu_available": gpu_available,
            "cpu": {"now": float(cpu_hist[-1]), "p50": cpu_p50, "p95": cpu_p95},
            "ram": {
                "now": float(ram_used_hist[-1]),
                "p95": ram_p95,
                "total": ram_total,
                "headroom": max(ram_total - float(ram_used_hist[-1]), 0.0),
            },
            "gpu_util": {"now": float(gpu_avg[-1]), "p50": gpu_p50, "p95": gpu_p95},
            "gpu_delta": {"now": float(gpu_delta[-1]), "p95": delta_p95},
            "gpu_mem": {
                "now": float(gpu_mem_worst[-1]),
                "p95": mem_p95,
                "headroom": max(max_gpu_capacity - float(gpu_mem_worst[-1]), 0.0),
            },
            "temp": {"now": temp_now, "p95": temp_p95, "status": temp_status},
        }

        payload = SystemDashboardPayload(
            window_len=len(window),
            gpu_available=gpu_available,
            rollups=rollups,
            series={
                "cpu": cpu_hist.astype(float).tolist(),
                "gpu_avg": gpu_avg.astype(float).tolist() if gpu_available else [],
            },
        )
        return payload.to_dict()


# ----------------------------
# Helpers
# ----------------------------

def _last_n_fast(table: Sequence[Any], n: int) -> List[Any]:
    """
    Return last n records efficiently for list/deque-like sequences.
    - For deque: reversed + islice is efficient.
    - For list: slicing is fastest.
    """
    if not table or n <= 0:
        return []
    try:
        # list path (fast slice)
        if isinstance(table, list):
            return table[-n:] if len(table) > n else table[:]
        # deque/sequence path
        return list(islice(reversed(table), n))[::-1]
    except Exception:
        # safest fallback
        try:
            return list(table[-n:]) if len(table) > n else list(table)
        except Exception:
            return []


def _gpus_safe(rec: Dict[str, Any]) -> List[List[float]]:
    """
    Defensive GPU extraction.
    Ensures each GPU row has 6 numeric fields:
      [util, mem_used, mem_total, temperature, power, power_limit]
    Malformed entries are ignored.
    """
    raw = rec.get("gpus", []) or []
    out: List[List[float]] = []
    for g in raw:
        if not g or not isinstance(g, (list, tuple)) or len(g) < 6:
            continue
        try:
            out.append([
                float(g[0] or 0.0),
                float(g[1] or 0.0),
                float(g[2] or 0.0),
                float(g[3] or 0.0),
                float(g[4] or 0.0),
                float(g[5] or 0.0),
            ])
        except Exception:
            continue
    return out