from dataclasses import dataclass
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


@dataclass
class Snapshot:
    cpu_percent: float
    ram_used: float
    ram_total: float
    gpu_available: bool
    gpu_count: int
    gpu_util_avg: Optional[float] = None
    gpu_util_max: Optional[float] = None
    gpu_mem_sum_used: Optional[float] = None
    gpu_mem_max_used: Optional[float] = None
    gpu_mem_total: Optional[float] = None


class SystemSampler(BaseSampler):
    """
    Sampler that tracks CPU RAM and GPU usage over time using psutil.
    Collects usage percentages periodically and exposes live snapshots
    and statistical summaries.
    Keeps per-GPU history internally, and exposes live metrics
    (min/max/avg/imbalance) as well as a summary (global peak, lowest non-zero,
    average, variance).
    """

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

    def __init__(self):
        super().__init__()
        setup_error_logger()
        self.logger = get_error_logger("SystemSampler")

        self._init_cpu()
        self._init_ram()
        self._init_gpu()

        self.cpu_history: Deque[float] = deque(maxlen=10_000)
        self.ram_history: Deque[float] = deque(maxlen=10_000)
        self.gpu_util_avg_history: Deque[float] = deque(maxlen=10_000)

        self.gpu_mem_sum_history: Deque[float] = deque(maxlen=10_000)
        self.gpu_mem_max_history: Deque[float] = deque(maxlen=10_000)
        self.gpu_mem_min_history: Deque[float] = deque(maxlen=10_000)
        self.gpu_mem_total_history: Deque[float] = deque(maxlen=10_000)

        self.latest: Optional[Snapshot] = None

    def _sample_cpu(self):
        """Sample CPU usage and update history."""
        try:
            cpu_usage = psutil.cpu_percent(interval=None)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: psutil.cpu_percent initial call failed: {e}"
            )
            cpu_usage = 0.0
        self.cpu_history.append(cpu_usage)

    def _sample_ram(self):
        """Sample RAM usage and update history."""
        try:
            mem = psutil.virtual_memory()
            ram_used = float(mem.used)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: psutil.virtual_memory initial call failed: {e}"
            )
            ram_used = 0.0
        self.ram_history.append(ram_used)

    def _sample_gpu(self):
        """Sample GPU usage and update histories. Returns dict."""
        if not self.gpu_available:
            return

        gpu_utils, gpu_mem_used, gpu_mem_total = [], [], []
        for i in range(self.gpu_count):
            try:
                handle = nvmlDeviceGetHandleByIndex(i)
                util = nvmlDeviceGetUtilizationRates(handle)
                meminfo = nvmlDeviceGetMemoryInfo(handle)

                util_pct = float(util.gpu)
                used_memory = float(meminfo.used)
                total_memory = float(meminfo.total)

            except Exception as e:
                self.logger.error(f"[TraceML] GPU {i} sampling failed: {e}")
                util_pct = 0.0
                used_memory = 0.0
                total_memory = 0.0

            gpu_utils.append(util_pct)
            gpu_mem_used.append(used_memory)
            gpu_mem_total.append(total_memory)

        util_arr = np.array(gpu_utils)
        mem_used_arr = np.array(gpu_mem_used)
        mem_total_arr = np.array(gpu_mem_total)

        avg_util = float(np.mean(util_arr))

        sum_mem_used = float(np.sum(mem_used_arr)) if mem_used_arr.size else 0.0
        max_mem_used = float(np.max(mem_used_arr)) if mem_used_arr.size else 0.0
        min_mem_used = float(np.min(mem_used_arr)) if mem_used_arr.size else 0.0

        # Update aggregated histories
        self.gpu_util_avg_history.append(avg_util)
        self.gpu_mem_sum_history.append(sum_mem_used)
        self.gpu_mem_max_history.append(max_mem_used)
        self.gpu_mem_min_history.append(min_mem_used)
        self.gpu_mem_total_history.append(float(np.sum(mem_total_arr)))

    def _generate_snapshot(self):
        """Convert current sample dict into Snapshot object."""
        return Snapshot(
            cpu_percent=self.cpu_history[-1],
            ram_used=self.ram_history[-1],
            ram_total=float(self.ram_total_memory),
            gpu_available=self.gpu_available,
            gpu_count=self.gpu_count,
            gpu_util_avg=(
                float(self.gpu_util_avg_history[-1]) if self.gpu_available else None
            ),
            gpu_mem_sum_used=(
                float(self.gpu_mem_sum_history[-1]) if self.gpu_available else None
            ),
            gpu_mem_max_used=(
                float(self.gpu_mem_max_history[-1]) if self.gpu_available else None
            ),
            gpu_mem_total=(
                float(self.gpu_mem_total_history[-1]) if self.gpu_available else None
            ),
        )

    def sample(self) -> Dict[str, Any]:
        """
        Poll current CPU, RAM and GPU for usage and return it as a dict.
        This method is non-blocking.

        Returns:
            Dict[str, Any]: Includes "error" key if sampling fails.
        """
        try:
            self._sample_cpu()
            self._sample_ram()
            self._sample_gpu()

            self.latest = self._generate_snapshot()

            snap = self.make_snapshot(
                ok=True,
                message="sampled successfully",
                source="system",
                data=self.latest.__dict__,
            )
            return self.snapshot_dict(snap)

        except Exception as e:
            self.logger.error(f"[TraceML] System sampling error: {e}")
            self.latest = None
            snap = self.make_snapshot(
                ok=False,
                message=f"sampling failed: {e}",
                source="system",
                data=None,
            )
            return self.snapshot_dict(snap)

    def _get_cpu_summary(self) -> Dict[str, Any]:
        cpu_values = [s for s in self.cpu_history]
        cpu_avg = float(np.mean(cpu_values)) if cpu_values else 0.0
        cpu_peak = float(np.max(cpu_values)) if cpu_values else 0.0
        return {
            "total_samples": len(self.cpu_history),
            "cpu_average_percent": round(cpu_avg, 2),
            "cpu_logical_core_count": self.cpu_logical_core_count,
            "cpu_peak_percent": round(cpu_peak, 2),
        }

    def _get_ram_summary(self) -> Dict[str, Any]:
        ram_values = [s for s in self.ram_history]
        ram_avg_used = float(np.mean(ram_values)) if ram_values else 0.0
        ram_peak_used = float(np.max(ram_values)) if ram_values else 0.0
        return {
            "ram_average_used": round(ram_avg_used, 2),
            "ram_peak_used": round(ram_peak_used, 2),
            "ram_total": self.ram_total_memory,
        }

    def _get_gpu_summary(self) -> Dict[str, Any]:
        if not self.gpu_available:
            return {"is_GPU_available": self.gpu_available}

        util_arr = (
            np.array(self.gpu_util_avg_history, dtype=float)
            if self.gpu_util_avg_history
            else np.array([], dtype=float)
        )
        average_gpu_util = float(np.mean(util_arr)) if util_arr.size else 0.0
        peak_gpu_util = float(np.max(util_arr)) if util_arr.size else 0.0

        sum_gpu_memory = [s for s in self.gpu_mem_sum_history]
        avg_mem = float(np.mean(sum_gpu_memory))

        peak_gpu_memory = [s for s in self.gpu_mem_max_history]
        peak_mem = float(np.mean(peak_gpu_memory))

        total_mem = [s for s in self.gpu_mem_total_history]
        total_mem = float(np.mean(total_mem))

        summary = {
            "is_GPU_available": self.gpu_available,
            "gpu_total_count": self.gpu_count,
            "gpu_average_util_percent": round(average_gpu_util, 2),
            "gpu_peak_util_percent": round(peak_gpu_util, 2),
            "gpu_memory_peak_used": round(peak_mem, 2),
            "gpu_memory_average_used": round(avg_mem, 2),
            "gpu_memory_total": round(float(total_mem), 2),
        }

        return summary

    def get_summary(self) -> Dict[str, Any]:
        """
        Summarize history across CPU, RAM, and GPU.
        """
        try:
            summary = {}
            summary.update(self._get_cpu_summary())
            summary.update(self._get_ram_summary())
            summary.update(self._get_gpu_summary())
            return summary

        except Exception as e:
            self.logger.error(f"[TraceML] System summary calculation error: {e}")
            return {
                "error": str(e),
                "total_system_samples": 0,
            }
