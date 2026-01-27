"""
System metrics computation layer.

This module contains *pure aggregation logic* for system-level telemetry.
No rendering, formatting, or UI dependencies.

Responsibilities
----------------
- Consume the `system` table schema
- Compute presentation-ready snapshots (latest sample)
- Compute aggregated summaries over the full run
"""

from typing import Dict, Any, List
import numpy as np

class SystemMetricsComputer:
    """
    Compute system-level metrics from the system telemetry table.

    Parameters
    ----------
    table : deque[Any]
        The in-memory system table containing sampled telemetry records.
    """

    def __init__(self, table):
        self._table = table

    def compute_snapshot(self) -> Dict[str, Any]:
        """
        Compute a presentation-friendly snapshot from the latest system sample.

        Returns
        -------
        Dict[str, Any]
            Flattened system metrics suitable for rendering layers.
        """
        if not self._table:
            return {
                "cpu": 0.0,
                "ram_used": 0.0,
                "ram_total": 0.0,
                "gpu_available": False,
                "gpu_count": 0,
                "gpu_util_total": None,
                "gpu_mem_used": None,
                "gpu_mem_total": None,
                "gpu_temp_max": None,
                "gpu_power_usage": None,
                "gpu_power_limit": None,
            }

        latest = self._table[-1]
        gpus: List[List[float]] = latest.get("gpus", []) or []

        if gpus:
            # Per-GPU wire format:
            # [util, mem_used, mem_total, temperature, power, power_limit]
            util_total = sum(g[0] for g in gpus)
            mem_used_total = sum(g[1] for g in gpus)
            mem_total_total = sum(g[2] for g in gpus)
            temp_max = max(g[3] for g in gpus)
            power_total = sum(g[4] for g in gpus)
            power_limit_total = sum(g[5] for g in gpus)
        else:
            util_total = mem_used_total = mem_total_total = None
            temp_max = power_total = power_limit_total = None

        return {
            "cpu": latest.get("cpu", 0.0),
            "ram_used": latest.get("ram_used", 0.0),
            "ram_total": latest.get("ram_total", 0.0),
            "gpu_available": latest.get("gpu_available", False),
            "gpu_count": latest.get("gpu_count", 0),
            "gpu_util_total": util_total,
            "gpu_mem_used": mem_used_total,
            "gpu_mem_total": mem_total_total,
            "gpu_temp_max": temp_max,
            "gpu_power_usage": power_total,
            "gpu_power_limit": power_limit_total,
        }


    def compute_summary(self) -> Dict[str, Any]:
        """
        Compute aggregated statistics over the entire run.

        Returns
        -------
        Dict[str, Any]
            Summary statistics for logging or reporting.
        """
        if not self._table:
            return {"error": "no data", "total_samples": 0}

        cpu_vals = [x.get("cpu", 0.0) for x in self._table]
        ram_vals = [x.get("ram_used", 0.0) for x in self._table]
        ram_total = self._table[-1].get("ram_total", 0.0)

        summary = {
            "total_samples": len(self._table),
            "cpu_avg_percent": round(float(np.mean(cpu_vals)), 2),
            "cpu_p95_percent": round(float(np.percentile(cpu_vals, 95)), 2),
            "ram_avg_used": round(float(np.mean(ram_vals)), 2),
            "ram_peak_used": round(float(np.max(ram_vals)), 2),
            "ram_total": ram_total,
            "gpu_available": self._table[-1].get("gpu_available", False),
            "gpu_total_count": self._table[-1].get("gpu_count", 0),
        }

        util_totals = []
        mem_used_totals = []
        mem_total_totals = []
        max_single_gpu_mem = []
        temp_max_vals = []

        for x in self._table:
            gpu_raw = x.get("gpu_raw", {}) or {}
            if not gpu_raw:
                continue

            util_totals.append(sum(v["util"] for v in gpu_raw.values()))
            mem_used_totals.append(sum(v["mem_used"] for v in gpu_raw.values()))
            mem_total_totals.append(sum(v["mem_total"] for v in gpu_raw.values()))
            max_single_gpu_mem.append(max(v["mem_used"] for v in gpu_raw.values()))
            temp_max_vals.append(max(v["temperature"] for v in gpu_raw.values()))

        if summary["gpu_available"] and util_totals:
            summary.update(
                {
                    "gpu_util_total_avg": round(float(np.mean(util_totals)), 2),
                    "gpu_util_total_peak": round(float(np.max(util_totals)), 2),
                    "gpu_mem_total_p95": round(
                        float(np.percentile(mem_used_totals, 95)), 2
                    ),
                    "gpu_mem_total_peak": round(float(np.max(mem_used_totals)), 2),
                    "gpu_mem_total_capacity": round(
                        float(np.mean(mem_total_totals)), 2
                    ),
                    "gpu_mem_single_peak": round(
                        float(np.max(max_single_gpu_mem)), 2
                    ),
                    "gpu_temp_peak": round(float(np.max(temp_max_vals)), 2),
                }
            )

        return summary