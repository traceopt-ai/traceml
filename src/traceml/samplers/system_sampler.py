from dataclasses import dataclass, field
from collections import deque
import psutil
from typing import List, Dict, Any, Optional, Deque
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
class CPUSample:
    percent: float


@dataclass
class RAMSample:
    percent: float
    used: float


@dataclass
class GPUSample:
    util_percent: float
    mem_used: float


@dataclass
class PerGPUState:
    total_mem: float
    util: Deque[float] = field(default_factory=lambda: deque(maxlen=10_000))
    mem_used: Deque[float] = field(default_factory=lambda: deque(maxlen=10_000))


@dataclass
class Snapshot:
    cpu_percent: float
    ram_used: float
    ram_total: float
    gpu_available: bool
    gpu_count: int
    gpu_util_avg: Optional[float] = None
    gpu_mem_used_total: Optional[float] = None
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
            self.gpus: Dict[int, PerGPUState] = {}
            # Initialize per-GPU with known total mem (if accessible now)
            for i in range(self.gpu_count):
                try:
                    handle = nvmlDeviceGetHandleByIndex(i)
                    total_memory = float(nvmlDeviceGetMemoryInfo(handle).total)
                except Exception:
                    total_memory = 0.0
                self.gpus[i] = PerGPUState(total_mem=total_memory)
        except NVMLError as e:
            self.logger.error(f"[TraceML] WARNING: GPU not available: {e}")

    def __init__(self):
        super().__init__()
        setup_error_logger()
        self.logger = get_error_logger("SystemSampler")

        self._init_cpu()
        self._init_ram()
        self._init_gpu()

        self.cpu_history: Deque[CPUSample] = deque(maxlen=10_000)
        self.ram_history: Deque[RAMSample] = deque(maxlen=10_000)
        if self.gpu_available:
            self.gpu_util_avg_history: Deque[float] = deque(maxlen=10_000)
            self.gpu_mem_peak_used_history: Deque[float] = deque(maxlen=10_000)
            self.gpu_mem_total_avg_history: Deque[float] = deque(maxlen=10_000)
        self.latest: Optional[Snapshot] = None

    def _sample_cpu(self):
        """Sample CPU usage and update history."""
        cpu_usage = psutil.cpu_percent(interval=None)
        self.cpu_history.append(CPUSample(percent=cpu_usage))
        return {"cpu_percent": round(cpu_usage, 2)}

    def _sample_ram(self):
        """Sample RAM usage and update history."""
        mem = psutil.virtual_memory()
        ram_percent_used = float(mem.percent)
        ram_used = float(mem.used)
        self.ram_history.append(RAMSample(percent=ram_percent_used, used=ram_used))
        return {"ram_used": round(ram_used, 2)}

    def _sample_gpu(self):
        """Sample GPU usage and update histories. Returns empty dict if no GPUs."""
        if not self.gpu_available:
            return {}

        gpu_utils, gpu_mem_used, gpu_mem_total = [], [], []
        for i in range(self.gpu_count):
            try:
                handle = nvmlDeviceGetHandleByIndex(i)
                util = nvmlDeviceGetUtilizationRates(handle)
                meminfo = nvmlDeviceGetMemoryInfo(handle)

                util_pct = float(util.gpu)
                used_memory = float(meminfo.used)
                total_memory = float(meminfo.total)

                if i not in self.gpus:
                    self.gpus[i] = PerGPUState(total_mem=total_memory)
                if self.gpus[i].total_mem == 0.0 and total_memory > 0.0:
                    self.gpus[i].total_mem = total_memory

                self.gpus[i].util.append(util_pct)
                self.gpus[i].mem_used.append(used_memory)

                gpu_utils.append(util_pct)
                gpu_mem_used.append(used_memory)
                gpu_mem_total.append(total_memory)

            except Exception as e:
                self.logger.error(f"[TraceML] GPU {i} sampling failed: {e}")

        if not gpu_utils:
            return {}

        util_arr = np.array(gpu_utils)
        mem_used_arr = np.array(gpu_mem_used)
        mem_total_arr = np.array(gpu_mem_total)

        avg_util = float(np.mean(util_arr))
        max_util = float(np.max(util_arr))
        nonzero_utils = util_arr[util_arr > 0]
        min_nonzero_util = float(np.min(nonzero_utils)) if nonzero_utils.size else 0.0
        imbalance_util = (max_util / min_nonzero_util) if min_nonzero_util > 0 else None

        highest_mem = float(np.max(mem_used_arr)) if mem_used_arr.size else 0.0
        nonzero_mem = mem_used_arr[mem_used_arr > 0]
        lowest_nonzero_mem = float(np.min(nonzero_mem)) if nonzero_mem.size else 0.0

        count_high_pressure = sum(
            1
            for used, total in zip(mem_used_arr, mem_total_arr)
            if total > 0 and (used / total) > 0.9
        )

        # Update aggregated histories
        self.gpu_util_avg_history.append(avg_util)
        self.gpu_mem_peak_used_history.append(highest_mem)
        self.gpu_mem_total_avg_history.append(float(np.mean(mem_total_arr)))

        return {
            "gpu_total_count": self.gpu_count,
            "gpu_util_avg_percent": round(avg_util, 2),
            "gpu_util_min_nonzero_percent": round(min_nonzero_util, 2),
            "gpu_util_max_percent": round(max_util, 2),
            "gpu_util_imbalance_ratio": round(imbalance_util, 2)
            if imbalance_util
            else None,
            "gpu_memory_highest_used": round(highest_mem, 2),
            "gpu_memory_lowest_nonzero_used": round(lowest_nonzero_mem, 2),
            "gpu_count_high_pressure": count_high_pressure,
        }

    def _generate_snapshot(self, current_sample):
        """Convert current sample dict into Snapshot object."""
        return Snapshot(
            cpu_percent=float(current_sample.get("cpu_percent", 0.0)),
            ram_used=float(current_sample.get("ram_used", 0.0)),
            ram_total=float(self.ram_total_memory),
            gpu_available=self.gpu_available,
            gpu_count=self.gpu_count,
            gpu_util_avg=float(current_sample.get("gpu_util_avg_percent", 0.0))
            if self.gpu_available
            else None,
            gpu_mem_used_total=float(current_sample.get("gpu_memory_highest_used", 0.0))
            if self.gpu_available
            else None,
            gpu_mem_total=float(self.gpu_mem_total_avg_history[-1])
            if self.gpu_available
            else None,
        )

    def sample(self) -> Dict[str, Any]:
        """
        Poll current CPU, RAM and GPU for usage and return it as a dict.
        This method is non-blocking.

        Returns:
            Dict[str, Any]: Includes "error" key if sampling fails.
        """
        try:
            current_sample: Dict[str, Any] = {}
            current_sample.update(self._sample_cpu())
            current_sample.update(self._sample_ram())
            current_sample.update(self._sample_gpu())

            self.latest = self._generate_snapshot(current_sample)

            snap = self.make_snapshot(
                ok=True,
                message="sampled successfully",
                source="system",
                data=self.latest.__dict__,  # or asdict(self.latest)
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
        cpu_values = [s.percent for s in self.cpu_history]
        cpu_avg = float(np.mean(cpu_values)) if cpu_values else 0.0
        cpu_peak = float(np.max(cpu_values)) if cpu_values else 0.0
        return {
            "total_samples": len(self.cpu_history),
            "cpu_average_percent": round(cpu_avg, 2),
            "cpu_logical_core_count": self.cpu_logical_core_count,
            "cpu_peak_percent": round(cpu_peak, 2),
        }

    def _get_ram_summary(self) -> Dict[str, Any]:
        ram_values = [s.used for s in self.ram_history]
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

        all_mem_usages: List[float] = []
        nonzero_mem_usages: List[float] = []

        for state in self.gpus.values():
            all_mem_usages.extend(state.mem_used)
            nonzero_mem_usages.extend([u for u in state.mem_used if u > 0])

        all_mem_arr = (
            np.array(all_mem_usages, dtype=float)
            if all_mem_usages
            else np.array([], dtype=float)
        )
        nonzero_mem_arr = (
            np.array(nonzero_mem_usages, dtype=float)
            if nonzero_mem_usages
            else np.array([], dtype=float)
        )

        global_peak = float(np.max(all_mem_arr)) if all_mem_arr.size else 0.0
        global_min_nonzero = (
            float(np.min(nonzero_mem_arr)) if nonzero_mem_arr.size else 0.0
        )
        avg_mem = float(np.mean(all_mem_arr)) if all_mem_arr.size else 0.0

        util_arr = (
            np.array(self.gpu_util_avg_history, dtype=float)
            if self.gpu_util_avg_history
            else np.array([], dtype=float)
        )
        average_gpu_util = float(np.mean(util_arr)) if util_arr.size else 0.0
        peak_gpu_util = float(np.max(util_arr)) if util_arr.size else 0.0

        # Try to read total GPU memory (use first GPU as baseline)
        try:
            handle = nvmlDeviceGetHandleByIndex(0)
            total_gpu_mem = nvmlDeviceGetMemoryInfo(handle).total
        except Exception:
            total_gpu_mem = 0

        summary = {
            "is_GPU_available": self.gpu_available,
            "gpu_total_count": self.gpu_count,
            "gpu_average_util_percent": round(average_gpu_util, 2),
            "gpu_peak_util_percent": round(peak_gpu_util, 2),
            "gpu_memory_global_peak_used": round(global_peak, 2),
            "gpu_memory_global_lowest_nonzero_used": round(
                global_min_nonzero, 2
            ),
            "gpu_memory_average_used": round(avg_mem, 2),
            "gpu_memory_global_total": round(total_gpu_mem, 2),
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
