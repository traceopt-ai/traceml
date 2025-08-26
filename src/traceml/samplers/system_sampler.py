from dataclasses import dataclass, field
from collections import deque
import psutil
import sys
from typing import List, Dict, Any, Optional, Deque
from .base_sampler import BaseSampler
import numpy as np

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
    available: float
    total: float


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
    ram_percent: float
    ram_used: float
    ram_available: float
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
    Keeps per-GPU history internally, and exposes distilled live metrics
    (min/max/avg/imbalance) as well as a summary (global peak, lowest non-zero,
    average, variance).
    """

    def __init__(self):
        super().__init__()

        # Initialize psutil.cpu_percent for non-blocking calls
        try:
            psutil.cpu_percent(interval=None)
        except Exception as e:
            print(
                f"[TraceML] WARNING: psutil.cpu_percent initial call failed: {e}",
                file=sys.stderr,
            )

        self.cpu_history: Deque[CPUSample] = deque(maxlen=10_000)
        self.ram_history: Deque[RAMSample] = deque(maxlen=10_000)

        # Aggregate GPU histories (not per-GPU)
        self.gpu_util_avg_history: Deque[float] = deque(maxlen=10_000)
        self.gpu_mem_peak_used_history: Deque[float] = deque(maxlen=10_000)
        self.gpu_mem_total_avg_history: Deque[float] = deque(maxlen=10_000)

        self.gpus: Dict[int, PerGPUState] = {}

        # GPU setup
        self.gpu_available = False
        self.gpu_count = 0
        self._nvml_inited = False
        try:
            nvmlInit()
            self._nvml_inited = True
            self.gpu_count = nvmlDeviceGetCount()
            self.gpu_available = self.gpu_count > 0
            # Initialize per-GPU with known total mem (if accessible now)
            for i in range(self.gpu_count):
                try:
                    handle = nvmlDeviceGetHandleByIndex(i)
                    total_memory = float(nvmlDeviceGetMemoryInfo(handle).total) / (
                        1024**2
                    )
                except Exception:
                    total_memory = 0.0
                self.gpus[i] = PerGPUState(total_mem=total_memory)
        except NVMLError as e:
            print(f"[TraceML] WARNING: GPU not available: {e}", file=sys.stderr)

        self.latest: Optional[Snapshot] = None

    def sample(self) -> Dict[str, Any]:
        """
        Poll current CPU, RAM and GPU for usage and return it as a dict.
        This method is non-blocking.

        Returns:
            Dict[str, Any]: Includes "error" key if sampling fails.
        """
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=None)

            # RAM usage
            mem = psutil.virtual_memory()
            ram_percent_used = float(mem.percent)
            ram_used = float(mem.used) / (1024**2)
            ram_available = float(mem.available) / (1024**2)
            ram_total = float(mem.total) / (1024**2)

            # Store raw samples for summary calculation
            self.cpu_history.append(CPUSample(percent=cpu_usage))
            self.ram_history.append(
                RAMSample(
                    percent=ram_percent_used,
                    used=ram_used,
                    available=ram_available,
                    total=ram_total,
                )
            )

            current_sample: Dict[str, Any] = {
                "cpu_percent": round(cpu_usage, 2),
                "ram_percent_used": round(ram_percent_used, 2),
                "ram_used": round(ram_used, 2),
                "ram_available": round(ram_available, 2),
                "ram_total": round(ram_total, 2),
            }

            if self.gpu_available:
                gpu_utils: List[float] = []
                gpu_mem_used: List[float] = []
                gpu_mem_total: List[float] = []

                for i in range(self.gpu_count):
                    try:
                        handle = nvmlDeviceGetHandleByIndex(i)
                        util = nvmlDeviceGetUtilizationRates(handle)
                        meminfo = nvmlDeviceGetMemoryInfo(handle)

                        util_pct = float(util.gpu)
                        used_memory = float(meminfo.used) / (1024**2)
                        total_memory = float(meminfo.total) / (1024**2)

                        if i not in self.gpus:
                            self.gpus[i] = PerGPUState(total_mem=total_memory)
                        if self.gpus[i].total_mem == 0.0 and total_memory > 0.0:
                            self.gpus[i].total_mem = total_memory

                        self.gpus[i].util.append(util_pct)
                        self.gpus[i].mem_used.append(used_memory)

                        gpu_utils.append(util_pct)
                        gpu_mem_used.append(used_memory)
                        gpu_mem_total.append(total_memory)

                    except NVMLError as e:
                        print(
                            f"[TraceML] NVML read failed for GPU {i}: {e}",
                            file=sys.stderr,
                        )
                    except Exception as e:
                        print(
                            f"[TraceML] Unexpected error reading GPU {i}: {e}",
                            file=sys.stderr,
                        )

                util_arr = (
                    np.array(gpu_utils, dtype=float)
                    if gpu_utils
                    else np.array([], dtype=float)
                )
                mem_used_arr = (
                    np.array(gpu_mem_used, dtype=float)
                    if gpu_mem_used
                    else np.array([], dtype=float)
                )
                mem_total_arr = (
                    np.array(gpu_mem_total, dtype=float)
                    if gpu_mem_total
                    else np.array([], dtype=float)
                )

                avg_util = float(np.mean(util_arr)) if util_arr.size else 0.0
                max_util = float(np.max(util_arr)) if util_arr.size else 0.0
                nonzero_utils = util_arr[util_arr > 0]
                min_nonzero_util = (
                    float(np.min(nonzero_utils)) if nonzero_utils.size else 0.0
                )
                imbalance_util = (
                    (max_util / min_nonzero_util) if min_nonzero_util > 0 else None
                )

                highest_mem = float(np.max(mem_used_arr)) if mem_used_arr.size else 0.0
                nonzero_mem = mem_used_arr[mem_used_arr > 0]
                lowest_nonzero_mem = (
                    float(np.min(nonzero_mem)) if nonzero_mem.size else 0.0
                )

                # Count GPUs under high pressure (>90% of its own total)
                count_high_pressure = 0
                for used, total in zip(mem_used_arr, mem_total_arr):
                    if total > 0 and (used / total) > 0.9:
                        count_high_pressure += 1

                # Update aggregated histories
                self.gpu_util_avg_history.append(avg_util)
                self.gpu_mem_peak_used_history.append(highest_mem)
                if mem_total_arr.size:
                    self.gpu_mem_total_avg_history.append(float(np.mean(mem_total_arr)))

                current_sample.update(
                    {
                        "gpu_total_count": self.gpu_count,
                        "gpu_util_avg_percent": round(avg_util, 2),
                        "gpu_util_min_nonzero_percent": (
                            round(min_nonzero_util, 2) if min_nonzero_util else 0.0
                        ),
                        "gpu_util_max_percent": round(max_util, 2),
                        "gpu_util_imbalance_ratio": (
                            round(imbalance_util, 2)
                            if imbalance_util is not None
                            else None
                        ),
                        "gpu_memory_highest_used": round(highest_mem, 2),
                        "gpu_memory_lowest_nonzero_used": (
                            round(lowest_nonzero_mem, 2) if lowest_nonzero_mem else 0.0
                        ),
                        "gpu_count_high_pressure": count_high_pressure,
                    }
                )

                # Maintain a typed Snapshot for internal use
            self.latest = Snapshot(
                cpu_percent=float(current_sample["cpu_percent"]),
                ram_percent=float(current_sample["ram_percent_used"]),
                ram_used=float(current_sample["ram_used"]),
                ram_available=float(current_sample["ram_available"]),
                ram_total=float(current_sample["ram_total"]),
                gpu_available=self.gpu_available,
                gpu_count=self.gpu_count,
                gpu_util_avg=(
                    float(current_sample.get("gpu_util_avg_percent", 0.0))
                    if self.gpu_available
                    else None
                ),
                gpu_mem_used_total=(
                    float(current_sample.get("gpu_memory_highest_used", 0.0))
                    if self.gpu_available
                    else None
                ),
                gpu_mem_total=(
                    float(self.gpu_mem_total_avg_history[-1])
                    if (self.gpu_available and self.gpu_mem_total_avg_history)
                    else None
                ),
            )

            # Return envelope via BaseSampler helpers
            snap = self.make_snapshot(
                ok=True,
                message="sampled successfully",
                source="system",
                data=self.latest.__dict__,  # or asdict(self.latest)
            )
            return self.snapshot_dict(snap)

        except Exception as e:
            print(f"[TraceML] System sampling error: {e}", file=sys.stderr)
            self.latest = None
            snap = self.make_snapshot(
                ok=False,
                message=f"sampling failed: {e}",
                source="system",
                data=None,
            )
            return self.snapshot_dict(snap)

    def get_summary(self) -> Dict[str, Any]:
        """
        Summarize history across CPU, RAM, and GPU.
        """
        try:
            # CPU
            cpu_values = [s.percent for s in self.cpu_history]
            cpu_avg = float(np.mean(cpu_values)) if cpu_values else 0.0
            cpu_peak = float(np.max(cpu_values)) if cpu_values else 0.0

            # RAM
            ram_percent_values = [s.percent for s in self.ram_history]
            ram_used_values = [s.used for s in self.ram_history]
            ram_avail_values = [s.available for s in self.ram_history]
            ram_total_values = [s.total for s in self.ram_history]

            ram_avg_pct = (
                float(np.mean(ram_percent_values)) if ram_percent_values else 0.0
            )
            ram_peak_pct = (
                float(np.max(ram_percent_values)) if ram_percent_values else 0.0
            )

            summary: Dict[str, Any] = {
                "total_system_samples": len(self.cpu_history),
                "cpu_average_percent": round(cpu_avg, 2),
                "cpu_peak_percent": round(cpu_peak, 2),
                "ram_average_percent_used": round(ram_avg_pct, 2),
                "ram_peak_percent_used": round(ram_peak_pct, 2),
            }

            # GPU summary
            if self.gpu_available and self.gpus:
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

                summary.update(
                    {
                        "gpu_total_count": self.gpu_count,
                        "gpu_average_util_percent": round(average_gpu_util, 2),
                        "gpu_peak_util_percent": round(peak_gpu_util, 2),
                        "gpu_memory_global_peak_used": round(global_peak, 2),
                        "gpu_memory_global_lowest_nonzero_used": round(
                            global_min_nonzero, 2
                        ),
                        "gpu_memory_average_used": round(avg_mem, 2),
                    }
                )

            return summary

        except Exception as e:
            print(f"[TraceML] System summary calculation error: {e}", file=sys.stderr)
            return {
                "error": str(e),
                "total_system_samples": 0,
            }
