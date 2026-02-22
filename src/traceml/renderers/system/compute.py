"""
System metrics computation layer.

This module contains *pure aggregation logic* for system-level telemetry.
No rendering, formatting, UI, Rich, or NiceGUI dependencies.

It provides two explicit entrypoints so we avoid unnecessary compute:
- compute_cli():   minimal latest-sample snapshot for CLI panels
- compute_dashboard(window_n): rolling-window rollups + timeseries arrays for dashboard

The output schemas are stable and intentionally compact:
- No raw table objects are returned (to keep payloads lightweight and serializable).
"""

from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple



@dataclass(frozen=True)
class SystemCLISnapshot:
    """
    Minimal snapshot used by CLI panel.

    Note: `gpu_*` totals are totals across all visible GPUs on this node.
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
    Rolling-window dashboard payload.

    - rollups: a dict matching  dashboard semantics
    - series: arrays used for plotting
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
    Compute system-level metrics from the system telemetry table.

    Parameters
    ----------
    table:
        A list/deque-like sequence of system telemetry dicts.
        Each entry may contain:
          cpu, ram_used, ram_total, gpu_available, gpu_count, gpus (wire list).
    """

    def __init__(self, table: Any) -> None:
        self._table: Sequence[Any] = table or ()

    def compute_cli(self) -> Dict[str, Any]:
        """
        Compute minimal latest-sample snapshot for CLI.

        Returns
        -------
        dict
            Flattened snapshot
        """
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
        gpus = _gpus(latest)

        if gpus:
            util_total = sum(g[0] for g in gpus)
            mem_used_total = sum(g[1] for g in gpus)
            mem_total_total = sum(g[2] for g in gpus)
            temp_max = max(g[3] for g in gpus)
            power_total = sum(g[4] for g in gpus)
            power_limit_total = sum(g[5] for g in gpus)
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

    def compute_dashboard(self, window_n: int = 100) -> Dict[str, Any]:
        """
        Compute rolling-window dashboard payload.

        Semantics preserved from your current dashboard implementation:
        - CPU: now/p50/p95 over time
        - GPU Util (avg): now/p50/p95 over time
        - GPU skew: max(util)-min(util) per sample; now/p95 over time
        - RAM: now/p95/total/headroom
        - GPU mem: worst GPU mem_used per sample; now/p95/headroom
        - Temp: max GPU temp per sample; now/p95/status
        - Series: cpu history + avg gpu util history (for plotting)

        Returns
        -------
        dict
            {"window_len", "gpu_available", "rollups", "series"}.
        """
        window = _last_n(self._table, int(window_n))
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

        cpu_hist = [float(r.get("cpu", 0.0) or 0.0) for r in window]
        ram_used_hist = [float(r.get("ram_used", 0.0) or 0.0) for r in window]
        ram_total = float(last.get("ram_total", 0.0) or 0.0)

        gpu_avg_hist: List[float] = []
        gpu_delta_hist: List[float] = []
        gpu_mem_hist: List[float] = []
        temp_hist: List[float] = []

        for r in window:
            utils = _gpu_utils(r)
            mems = _gpu_mems(r)
            temps = _gpu_temps(r)

            if utils:
                gpu_avg_hist.append(sum(utils) / len(utils))
                gpu_delta_hist.append(max(utils) - min(utils))
            else:
                gpu_avg_hist.append(0.0)
                gpu_delta_hist.append(0.0)

            # Worst GPU mem_used per sample (conservative)
            gpu_mem_hist.append(max((m for m, _ in mems), default=0.0))
            temp_hist.append(max(temps, default=0.0))

        # Capacity: from latest sample, max mem_total across GPUs (same as your current code)
        max_gpu_capacity = max((m for _, m in _gpu_mems(last)), default=0.0)

        rollups = {
            "gpu_available": gpu_available,
            "cpu": {
                "now": cpu_hist[-1],
                "p50": _percentile(cpu_hist, 50),
                "p95": _percentile(cpu_hist, 95),
            },
            "ram": {
                "now": ram_used_hist[-1],
                "p95": _percentile(ram_used_hist, 95),
                "total": ram_total,
                "headroom": max(ram_total - ram_used_hist[-1], 0.0),
            },
            "gpu_util": {
                "now": gpu_avg_hist[-1],
                "p50": _percentile(gpu_avg_hist, 50),
                "p95": _percentile(gpu_avg_hist, 95),
            },
            "gpu_delta": {
                "now": gpu_delta_hist[-1],
                "p95": _percentile(gpu_delta_hist, 95),
            },
            "gpu_mem": {
                "now": gpu_mem_hist[-1],
                "p95": _percentile(gpu_mem_hist, 95),
                "headroom": max(max_gpu_capacity - gpu_mem_hist[-1], 0.0),
            },
            "temp": {
                "now": temp_hist[-1],
                "p95": _percentile(temp_hist, 95),
                "status": "Hot" if temp_hist[-1] >= 85 else "Warm" if temp_hist[-1] >= 80 else "OK",
            },
        }

        payload = SystemDashboardPayload(
            window_len=len(window),
            gpu_available=gpu_available,
            rollups=rollups,
            series={
                "cpu": cpu_hist,
                "gpu_avg": gpu_avg_hist if gpu_available else [],
            },
        )
        return payload.to_dict()




def _last_n(table: Sequence[Any], n: int) -> List[Any]:
    """
    Return the last n records efficiently (works for list/deque-like sequences).

    Cost: O(n) for deque/list.
    """
    if not table or n <= 0:
        return []
    if hasattr(table, "__reversed__"):
        # Deque supports reversed() efficiently
        return list(islice(reversed(table), n))[::-1]
    return list(table[-n:]) if len(table) > n else list(table)


def _percentile(vals: Iterable[float], p: float) -> float:
    """
    Simple percentile computation (no numpy dependency).

    Parameters
    ----------
    vals:
        Iterable of numeric values.
    p:
        Percentile in [0, 100].

    Returns
    -------
    float
        The percentile value, or 0.0 if input is empty.
    """
    xs = sorted(float(v) for v in vals if v is not None)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    # linear interpolation
    return xs[f] * (c - k) + xs[c] * (k - f)


def _gpus(rec: Dict[str, Any]) -> List[List[float]]:
    """
    Return GPU wire list:
    Each GPU entry is [util, mem_used, mem_total, temperature, power, power_limit].
    """
    return rec.get("gpus", []) or []


def _gpu_utils(rec: Dict[str, Any]) -> List[float]:
    return [g[0] for g in _gpus(rec) if g]


def _gpu_mems(rec: Dict[str, Any]) -> List[Tuple[float, float]]:
    return [(g[1], g[2]) for g in _gpus(rec) if g]


def _gpu_temps(rec: Dict[str, Any]) -> List[float]:
    return [g[3] for g in _gpus(rec) if g]
