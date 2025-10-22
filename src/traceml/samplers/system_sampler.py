from collections import deque
import psutil
from typing import Dict, Any, Optional, Deque
import numpy as np
from .base_sampler import BaseSampler
from traceml.loggers.error_log import setup_error_logger, get_error_logger


from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetCount,
    NVMLError,
)



class SystemSampler(BaseSampler):
    """
    Sampler that tracks CPU RAM and GPU usage over time using psutil.
    Collects usage percentages periodically and exposes live snapshots
    and statistical summaries.
    Keeps per-GPU history internally, and exposes live metrics
    (min/max/avg/imbalance) as well as a summary (global peak, lowest non-zero,
    average, variance).
    """

    def __init__(self, table=None, max_snapshots: int = 10_000):
        super().__init__()
        setup_error_logger()
        self.logger = get_error_logger("SystemSampler")
        self._table = table or []

        self._init_cpu()
        self._init_ram()
        self._init_gpu()

    def _init_cpu(self):
        # Initialize psutil.cpu_percent for non-blocking calls
        self.cpu_logical_core_count = 0
        try:
            psutil.cpu_percent(interval=None)
            self.cpu_logical_core_count = psutil.cpu_count(logical=True)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: psutil.cpu_percent initial call failed: {e}"
            )

    def _init_ram(self):
        self.ram_total_memory = 0
        try:
            self.ram_total_memory = psutil.virtual_memory().total
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: psutil.virtual_memory initial call failed: {e}"
            )

    def _init_gpu(self):
        # GPU setup
        self.gpu_available = False
        self.gpu_count = 0
        try:
            nvmlInit()
            self.gpu_count = nvmlDeviceGetCount()
            self.gpu_available = self.gpu_count > 0
        except NVMLError as e:
            self.logger.error(f"[TraceML] WARNING: GPU not available: {e}")

    def _sample_cpu(self):
        """Sample CPU usage and update history."""
        try:
            return psutil.cpu_percent(interval=None)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: psutil.cpu_percent initial call failed: {e}"
            )
            return 0.0

    def _sample_ram(self):
        """Sample RAM usage and update history."""
        try:
            mem = psutil.virtual_memory()
            return float(mem.used)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: psutil.virtual_memory initial call failed: {e}"
            )
            return  0.0

    def _sample_gpu(self):
        """
        Sample GPU usage and update histories. Returns dict per-GPU metrics:
        {gpu_id: {"util": ..., "mem_used": ..., "mem_total": ...}}
        """
        if not self.gpu_available:
            return {}

        gpu_info: Dict[int, Dict[str, float]] = {}
        for i in range(self.gpu_count):
            try:
                handle = nvmlDeviceGetHandleByIndex(i)
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                gpu_info[i] = {
                    "util": float(util.gpu),
                    "mem_used": float(mem.used),
                    "mem_total": float(mem.total),
                }

            except Exception as e:
                self.logger.error(f"[TraceML] GPU {i} sampling failed: {e}")
                gpu_info[i] = {"util": 0.0, "mem_used": 0.0, "mem_total": 0.0}

        return gpu_info


    def sample(self) -> Dict[str, Any]:
        """
        Take one system snapshot and return backward-compatible dict.
        """
        try:
            cpu = self._sample_cpu()
            ram_used = self._sample_ram()
            gpu_raw = self._sample_gpu()

            self._table.append({
                "cpu_percent": cpu,
                "ram_used": ram_used,
                "ram_total": self.ram_total_memory,
                "gpu_available": self.gpu_available,
                "gpu_count": self.gpu_count,
                "gpu_raw": gpu_raw,
            })

            # backward-compatible
            return self.snapshot_dict(
                self.make_snapshot(
                    ok=True,
                    message="sampled successfully",
                    source="system",
                    data={},
                )
            )

        except Exception as e:
            self.logger.error(f"[TraceML] System sampling error: {e}")
            return self.snapshot_dict(
                self.make_snapshot(
                    ok=False,
                    message=f"sampling failed: {e}",
                    source="system",
                    data=None,
                )
            )

    def get_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics from the table list.
        Each entry in the list is a sample dict.
        """
        if not self._table:
            return {"error": "no data", "total_samples": 0}

        # Extract lists of values
        cpu_vals = [x.get("cpu_percent", 0.0) for x in self._table]
        ram_vals = [x.get("ram_used", 0.0) for x in self._table]
        ram_total = self._table[-1].get("ram_total", 0.0)

        gpu_util_avg = [x.get("gpu_util_avg") for x in self._table if x.get("gpu_util_avg") is not None]
        gpu_util_max = [x.get("gpu_util_max") for x in self._table if x.get("gpu_util_max") is not None]
        gpu_mem_sum_used = [x.get("gpu_mem_sum_used") for x in self._table if x.get("gpu_mem_sum_used") is not None]
        gpu_mem_max_used = [x.get("gpu_mem_max_used") for x in self._table if x.get("gpu_mem_max_used") is not None]
        gpu_mem_total = [x.get("gpu_mem_total") for x in self._table if x.get("gpu_mem_total") is not None]

        gpu_available = self._table[-1].get("gpu_available", False)
        gpu_count = self._table[-1].get("gpu_count", 0)

        summary = {
            "total_samples": len(self._table),
            "cpu_average_percent": round(float(np.mean(cpu_vals)), 2),
            "cpu_peak_percent": round(float(np.max(cpu_vals)), 2),
            "cpu_logical_core_count": self.cpu_logical_core_count,
            "ram_average_used": round(float(np.mean(ram_vals)), 2),
            "ram_peak_used": round(float(np.max(ram_vals)), 2),
            "ram_total": ram_total,
            "gpu_available": gpu_available,
            "gpu_total_count": gpu_count,
        }

        # Temporary backward-compatible GPU summary
        if gpu_available and gpu_util_avg:
            summary.update({
                "gpu_average_util_percent": round(float(np.mean(gpu_util_avg)), 2),
                "gpu_peak_util_percent": round(float(np.max(gpu_util_max)), 2),
                "gpu_memory_peak_used": round(float(np.max(gpu_mem_max_used)), 2),
                "gpu_memory_average_used": round(float(np.mean(gpu_mem_sum_used)), 2),
                "gpu_memory_total": round(float(np.mean(gpu_mem_total)), 2),
            })

        return summary
