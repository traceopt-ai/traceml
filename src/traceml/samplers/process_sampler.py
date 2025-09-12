from dataclasses import dataclass
from collections import deque
import psutil
import torch
import os
from typing import Dict, Any, Optional, Deque, Union
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

    def _init_process(self) -> None:
        """Attach to target process (default: current pid)."""
        try:
            self.pid = os.getpid()
            self.process = psutil.Process(self.pid)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: Failed to attach to process {os.getpid()}: {e}"
            )
            self.pid = None
            self.process = None

    def _warmup_cpu(self) -> None:
        # CPU usage measurement
        try:
            self.process.cpu_percent(interval=None)
        except Exception as e:
            self.logger.error( f"[TraceML] WARNING: process.cpu_percent() initial call failed: {e}")

    def _init_gpu(self) -> None:
        self.gpu_available = False
        self.gpu_count = 0
        try:
            nvmlInit()
            self.gpu_count = nvmlDeviceGetCount()
            self.gpu_available = True
        except NVMLError as e:
            self.logger.error(f"[TraceML] WARNING: NVML GPU support unavailable: {e}")


    def __init__(self):
        super().__init__()
        setup_error_logger()
        self.logger = get_error_logger("ProcessSampler")

        # Sampling history
        self.cpu_history: Deque[ProcessCPUSample] = deque(maxlen=10_000)
        self.ram_history: Deque[ProcessRAMSample] = deque(maxlen=10_000)
        self.gpu_mem_history: Deque[ProcessGPUMemSample] = deque(maxlen=10_000)
        # Initiate
        self._init_process()
        self._warmup_cpu()
        self._init_gpu()
        # Latest snapshot
        self.latest: Optional[ProcessSnapshot] = None

    def _sample_cpu(self) -> ProcessCPUSample:
        sample = ProcessCPUSample(percent=self.process.cpu_percent(interval=None))
        self.cpu_history.append(sample)
        return sample

    def _sample_ram(self) -> ProcessRAMSample:
        sample = ProcessRAMSample(used=self.process.memory_info().rss)
        self.ram_history.append(sample)
        return sample

    def _sample_gpu_memory(self) -> int:
        """Return the GPU memory used by this process, or None if unavailable."""
        if not self.gpu_available:
            return None

        total_memory = 0
        try:
            for i in range(self.gpu_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                procs = nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in procs:
                    if proc.pid == self.pid:
                        total_memory += proc.usedGpuMemory
            if total_memory > 0:
                return total_memory
        except NVMLError as e:
            self.logger.error(f"[TraceML] NVML GPU memory read failed: {e}")
        except Exception as e:
            self.logger.error(f"[TraceML] Unexpected error reading GPU memory: {e}")

        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated()
        except Exception as e:
            self.logger.error(f"[TraceML] Torch GPU memory read failed: {e}")
        return None


    def _sample_gpu(self) -> Optional[ProcessGPUMemSample]:
        gpu_memory_usage = self._sample_gpu_memory()
        if gpu_memory_usage is not None:
            sample = ProcessGPUMemSample(used=gpu_memory_usage)
            self.gpu_mem_history.append(sample)
            return sample
        return None


    def sample(self) -> Dict[str, Any]:
        """
        Sample current CPU, RAM and GPU usage of the monitored process.
        Returns:
            envelope dict via BaseSampler helpers
        """
        try:
            cpu_sample = self._sample_cpu()
            ram_sample = self._sample_ram()
            gpu_sample = self._sample_gpu()

            # Create snapshot
            self.latest = ProcessSnapshot(
                process_cpu_percent=round(cpu_sample.percent, 2),
                process_ram=round(ram_sample.used, 2),
                process_gpu_memory=round(gpu_sample.used, 2) if gpu_sample is not None else None,
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


    def _get_cpu_summary(self) -> Dict[str, Any]:
        cpu_values = [s.percent for s in self.cpu_history]
        if not cpu_values:
            return {
                "total_process_samples": 0
            }
        avg_cpu = float(sum(cpu_values)/len(cpu_values))
        peak_cpu = float(max(cpu_values))
        return {
            "total_process_samples": len(cpu_values),
            "process_average_cpu_percent": avg_cpu,
            "process_peak_cpu_percent": peak_cpu,
        }

    def _get_ram_summary(self) -> Dict[str, Any]:
        

    def get_summary(self) -> Dict[str, Any]:
        """
        Summarize history for process CPU, RAM, and GPU memory.
        Values are reported in percentages instead of absolute MB.
        """
        try:
            summary: Dict[str, Any] = {}
            summary.update(self._get_cpu_summary())

            ram_values = [s.used for s in self.ram_history]
            gpu_mem_values = [s.used for s in self.gpu_mem_history]

            # Get system RAM total
            total_ram = psutil.virtual_memory().total / (1024**2)  # MB

            # Get GPU total (use NVML first, fallback to torch)
            total_gpu = None
            try:
                from pynvml import nvmlDeviceGetMemoryInfo

                if self.gpu_available:
                    handle = nvmlDeviceGetHandleByIndex(0)
                    total_gpu = nvmlDeviceGetMemoryInfo(handle).total / (1024**2)
            except Exception:
                try:
                    if torch.cuda.is_available():
                        total_gpu = torch.cuda.get_device_properties(0).total_memory / (
                            1024**2
                        )
                except Exception:
                    total_gpu = None

            if ram_values and total_ram:
                summary.update(
                    {
                        "process_average_ram_percent": round(
                            float(sum(ram_values) / len(ram_values)) / total_ram * 100,
                            2,
                        ),
                        "process_peak_ram_percent": round(
                            max(ram_values) / total_ram * 100, 2
                        ),
                    }
                )

            if gpu_mem_values and total_gpu:
                summary.update(
                    {
                        "process_average_gpu_percent": round(
                            float(sum(gpu_mem_values) / len(gpu_mem_values))
                            / total_gpu
                            * 100,
                            2,
                        ),
                        "process_peak_gpu_percent": round(
                            max(gpu_mem_values) / total_gpu * 100, 2
                        ),
                    }
                )

            return summary

        except Exception as e:
            self.logger.error(f"[TraceML] Process summary error: {e}")
            return {
                "error": str(e),
                "total_process_samples": 0,
            }
