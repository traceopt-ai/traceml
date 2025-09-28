from dataclasses import dataclass
from collections import deque
import psutil
import torch
import os
from typing import Dict, Any, Optional, Deque
from .base_sampler import BaseSampler
from traceml.loggers.error_log import setup_error_logger, get_error_logger


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
            self.cpu_count = psutil.cpu_count(logical=True)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: process.cpu_percent() initial call failed: {e}"
            )

    def _init_gpu(self) -> None:
        self.gpu_available = False
        self.gpu_count = 0
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            self.gpu_available = True

    def __init__(self):
        super().__init__()
        setup_error_logger()
        self.logger = get_error_logger("ProcessSampler")

        # Sampling history
        self.cpu_history: Deque[float] = deque(maxlen=10_000)
        self.ram_history: Deque[float] = deque(maxlen=10_000)
        self.gpu_mem_history: Deque[float] = deque(maxlen=10_000)
        # Initiate
        self._init_process()
        self._warmup_cpu()
        self._init_gpu()
        # Latest snapshot
        self.latest: Optional[ProcessSnapshot] = None

    def _sample_cpu(self):
        try:
            cpu_percent = self.process.cpu_percent(interval=None)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: Failed to sample CPU usage from process CPU usage: {e}"
            )
            cpu_percent = 0.0
        self.cpu_history.append(float(cpu_percent))

    def _sample_ram(self):
        try:
            ram_percent = self.process.memory_info().rss
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: Failed to sample RAM usage from process RAM usage: {e}"
            )
            ram_percent = 0.0
        self.ram_history.append(float(ram_percent))

    def _sample_gpu_memory(self):
        """Return the GPU memory used by this process, or None if unavailable."""
        if not self.gpu_available:
            return None
        try:
            return torch.cuda.memory_allocated()
        except Exception as e:
            self.logger.error(f"[TraceML] Torch GPU memory read failed: {e}")
        return 0.0

    def _sample_gpu(self):
        gpu_memory_usage = self._sample_gpu_memory()
        if gpu_memory_usage is not None:
            self.gpu_mem_history.append(float(gpu_memory_usage))

    def sample(self) -> Dict[str, Any]:
        """
        Sample current CPU, RAM and GPU usage of the monitored process.
        Returns:
            envelope dict via BaseSampler helpers
        """
        try:
            self._sample_cpu()
            self._sample_ram()
            self._sample_gpu()

            # Create snapshot
            self.latest = ProcessSnapshot(
                process_cpu_percent=round(self.cpu_history[-1], 2),
                process_ram=round(self.ram_history[-1], 2),
                process_gpu_memory=(
                    round(self.gpu_mem_history[-1], 2) if self.gpu_available else None
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

    def _get_cpu_summary(self) -> Dict[str, Any]:
        cpu_values = [s for s in self.cpu_history]
        if not cpu_values:
            return {
                "total_samples": 0,
            }
        avg_cpu = float(sum(cpu_values) / len(cpu_values))
        peak_cpu = float(max(cpu_values))
        return {
            "total_samples": len(cpu_values),
            "average_cpu_percent": round(avg_cpu, 2),
            "peak_cpu_percent": round(peak_cpu, 2),
            "cpu_logical_core_count": self.cpu_count,
        }

    def _get_ram_summary(self) -> Dict[str, Any]:
        ram_values = [s for s in self.ram_history]
        if not ram_values:
            return {}
        total_ram = psutil.virtual_memory().total
        avg_ram = float(sum(ram_values) / len(ram_values))
        peak_ram = float(max(ram_values))
        return {
            "average_ram": round(avg_ram, 2),
            "peak_ram": round(peak_ram, 2),
            "total_ram": total_ram,
        }

    def _get_gpu_memory(self) -> Dict[str, Any]:
        gpu_mem_values = [s for s in self.gpu_mem_history]
        if not gpu_mem_values:
            return {"is_GPU_available": self.gpu_available}

        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        return {
            "is_GPU_available": self.gpu_available,
            "average_gpu_memory_used": float(sum(gpu_mem_values) / len(gpu_mem_values)),
            "peak_gpu_memory_used": max(gpu_mem_values),
            "total_gpu_memory_used": total_gpu_memory,
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Summarize history for process CPU, RAM, and GPU memory.
        Values are reported in percentages instead of absolute MB.
        """
        try:
            summary: Dict[str, Any] = {}
            summary.update(self._get_cpu_summary())
            summary.update(self._get_ram_summary())
            summary.update(self._get_gpu_memory())

            return summary

        except Exception as e:
            self.logger.error(f"[TraceML] Process summary error: {e}")
            return {
                "error": str(e),
                "total_process_samples": 0,
            }
