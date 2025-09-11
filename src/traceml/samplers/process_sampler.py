from dataclasses import dataclass
from collections import deque
import psutil
import torch
import os
from typing import Dict, Any, Optional, Deque
from .base_sampler import BaseSampler
from traceml.loggers.error_log import setup_error_logger, get_error_logger

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
        setup_error_logger()
        self.logger = get_error_logger("ProcessSampler")

        # Monitor current process by default
        try:
            self.pid = pid or os.getpid()
            self.process = psutil.Process(self.pid)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: Failed to attach to process {pid or os.getpid()}: {e}"
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
            self.logger.error(
                f"[TraceML] WARNING: process.cpu_percent() initial call failed: {e}"
            )

        # GPU Tracking
        try:
            nvmlInit()
            self.gpu_count = nvmlDeviceGetCount()
            self.gpu_available = True
        except NVMLError as e:
            self.logger.error(f"[TraceML] WARNING: NVML GPU support unavailable: {e}")

        # Latest snapshot
        self.latest: Optional[ProcessSnapshot] = None

    def _get_process_gpu_memory(self) -> Optional[float]:
        """Return the GPU memory (in MB) used by this process, or None if unavailable."""
        if not self.gpu_available:
            return None

        total_memory = 0
        try:
            for i in range(self.gpu_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                procs = nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in procs:
                    if proc.pid == self.pid:
                        total_memory += proc.usedGpuMemory / (1024**2)
            if total_memory > 0:
                return total_memory
        except NVMLError as e:
            self.logger.error(f"[TraceML] NVML GPU memory read failed: {e}")
        except Exception as e:
            self.logger.error(f"[TraceML] Unexpected error reading GPU memory: {e}")

        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 2)
        except Exception as e:
            self.logger.error(f"[TraceML] Torch GPU memory read failed: {e}")

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
            self.logger.error(f"[TraceML] Process sampling error: {e}")
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
        Values are reported in percentages instead of absolute MB.
        """
        try:
            cpu_values = [s.percent for s in self.cpu_history]
            ram_values = [s.used for s in self.ram_history]
            gpu_mem_values = [s.used for s in self.gpu_mem_history]

            # Get system RAM total
            total_ram = psutil.virtual_memory().total / (1024 ** 2)  # MB

            # Get GPU total (use NVML first, fallback to torch)
            total_gpu = None
            try:
                from pynvml import nvmlDeviceGetMemoryInfo
                if self.gpu_available:
                    handle = nvmlDeviceGetHandleByIndex(0)
                    total_gpu = nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 2)
            except Exception:
                try:
                    if torch.cuda.is_available():
                        total_gpu = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                except Exception:
                    total_gpu = None

            summary: Dict[str, Any] = {
                "total_process_samples": len(cpu_values),
                "process_average_cpu_percent": (
                    round(float(sum(cpu_values) / len(cpu_values)), 2)
                    if cpu_values else 0.0
                ),
                "process_peak_cpu_percent": round(max(cpu_values), 2) if cpu_values else 0.0,
            }

            if ram_values and total_ram:
                summary.update(
                    {
                        "process_average_ram_percent": round(
                            float(sum(ram_values) / len(ram_values)) / total_ram * 100, 2
                        ),
                        "process_peak_ram_percent": round(max(ram_values) / total_ram * 100, 2),
                    }
                )

            if gpu_mem_values and total_gpu:
                summary.update(
                    {
                        "process_average_gpu_percent": round(
                            float(sum(gpu_mem_values) / len(gpu_mem_values)) / total_gpu * 100, 2
                        ),
                        "process_peak_gpu_percent": round(max(gpu_mem_values) / total_gpu * 100, 2),
                    }
                )

            return summary

        except Exception as e:
            self.logger.error(f"[TraceML] Process summary error: {e}")
            return {
                "error": str(e),
                "total_process_samples": 0,
            }

