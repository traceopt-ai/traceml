from dataclasses import dataclass
from collections import deque
import psutil
import os
import sys
from typing import Dict, Any, Optional, Deque
from .base_sampler import BaseSampler

from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetCount,
    NVMLError,
)


@dataclass
class ProcessCPUSample:
    percent: float


@dataclass
class ProcessRAMSample:
    used: float


@dataclass
class ProcessGPUMemSample:
    used: float


@dataclass
class ProcessSnapshot:
    process_cpu_percent: float
    process_ram: float
    process_gpu_memory: Optional[float] = None


class ProcessSampler(BaseSampler):
    """
    Sampler that tracks CPU and RAM usage of the current Python process
    (or a specified PID) over time using psutil.
    """

    def __init__(self, pid: Optional[int] = None):
        super().__init__()

        # Monitor current process by default
        try:
            self.pid = pid or os.getpid()
            self.process = psutil.Process(self.pid)
        except Exception as e:
            print(
                f"[TraceML] WARNING: Failed to attach to process {pid or os.getpid()}: {e}",
                file=sys.stderr,
            )
            self.pid = None
            self.process = None

        # Sampling history
        self.cpu_history: Deque[ProcessCPUSample] = deque(maxlen=10_000)
        self.ram_history: Deque[ProcessRAMSample] = deque(maxlen=10_000)
        self.gpu_mem_history: Deque[ProcessGPUMemSample] = deque(maxlen=10_000)

        self.gpu_available = False
        self.gpu_count = 0
        # CPU usage measurement
        try:
            self.process.cpu_percent(interval=None)
        except Exception as e:
            print(
                f"[TraceML] WARNING: process.cpu_percent() initial call failed: {e}",
                file=sys.stderr,
            )
        # GPU Tracking
        try:
            nvmlInit()
            self.gpu_count = nvmlDeviceGetCount()
            self.gpu_available = True
        except NVMLError as e:
            print(
                f"[TraceML] WARNING: NVML GPU support unavailable: {e}", file=sys.stderr
            )

        # Latest snapshot
        self.latest: Optional[ProcessSnapshot] = None

    def _get_process_gpu_memory(self) -> Optional[float]:
        """Return the GPU memory (in MB) used by this process, or None if unavailable."""
        if not self.gpu_available:
            return None

        try:
            for i in range(self.gpu_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                procs = nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in procs:
                    if proc.pid == self.pid:
                        return proc.usedGpuMemory / (1024**2)
        except NVMLError as e:
            print(f"[TraceML] NVML GPU memory read failed: {e}", file=sys.stderr)
        except Exception as e:
            print(
                f"[TraceML] Unexpected error reading GPU memory: {e}", file=sys.stderr
            )
        return None

    def sample(self) -> Dict[str, Any]:
        """
        Sample current CPU, RAM and GPU usage of the monitored process.
        Returns:
            envelope dict via BaseSampler helpers
        """
        try:
            cpu_usage = self.process.cpu_percent(interval=None)
            ram_usage = self.process.memory_info().rss / (1024**2)
            gpu_mem_usage = self._get_process_gpu_memory()

            # Append to histories
            self.cpu_history.append(ProcessCPUSample(percent=cpu_usage))
            self.ram_history.append(ProcessRAMSample(used=ram_usage))
            if gpu_mem_usage is not None:
                self.gpu_mem_history.append(ProcessGPUMemSample(used=gpu_mem_usage))

            # Create snapshot
            self.latest = ProcessSnapshot(
                process_cpu_percent=round(cpu_usage, 2),
                process_ram=round(ram_usage, 2),
                process_gpu_memory=(
                    round(gpu_mem_usage, 2) if gpu_mem_usage is not None else None
                ),
            )

            snap = self.make_snapshot(
                ok=True,
                message="sampled successfully",
                source="process",
                data=self.latest.__dict__,
            )
            return self.snapshot_dict(snap)

        except Exception as e:
            print(f"[TraceML] Process sampling error: {e}", file=sys.stderr)
            self.latest = None
            snap = self.make_snapshot(
                ok=False,
                message=f"sampling failed: {e}",
                source="process",
                data=None,
            )
            return self.snapshot_dict(snap)

    def get_summary(self) -> Dict[str, Any]:
        """
        Summarize history for process CPU, RAM, and GPU memory.
        """
        try:
            cpu_values = [s.percent for s in self.cpu_history]
            ram_values = [s.used for s in self.ram_history]
            gpu_mem_values = [s.used for s in self.gpu_mem_history]

            summary: Dict[str, Any] = {
                "total_process_samples": len(cpu_values),
                "cpu_average_percent": (
                    round(float(sum(cpu_values) / len(cpu_values)), 2)
                    if cpu_values
                    else 0.0
                ),
                "cpu_peak_percent": round(max(cpu_values), 2) if cpu_values else 0.0,
                "ram_average": (
                    round(float(sum(ram_values) / len(ram_values)), 2)
                    if ram_values
                    else 0.0
                ),
                "ram_peak": round(max(ram_values), 2) if ram_values else 0.0,
            }
            if gpu_mem_values:
                summary.update(
                    {
                        "gpu_average_memory": round(
                            float(sum(gpu_mem_values) / len(gpu_mem_values)), 2
                        ),
                        "gpu_peak_memory": round(max(gpu_mem_values), 2),
                    }
                )
            return summary

        except Exception as e:
            print(f"[TraceML] Process summary error: {e}", file=sys.stderr)
            return {
                "error": str(e),
                "total_process_samples": 0,
            }
