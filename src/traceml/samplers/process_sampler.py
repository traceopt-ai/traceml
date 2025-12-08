import psutil
import torch
import os
import time
from .base_sampler import BaseSampler
from traceml.loggers.error_log import get_error_logger


class ProcessSampler(BaseSampler):
    """
    Sampler that tracks CPU and RAM usage of the current Python process
    (or a specified PID) over time using psutil.
    """

    def __init__(self) -> None:
        self.sampler_name = "ProcessSampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)
        self.db.create_table("process")

        # Initiate
        self._init_process()
        self._init_ram()
        self._warmup_cpu()
        self._init_gpu()

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

    def _init_ram(self) -> None:
        self.ram_total = 0.0
        try:
            self.ram_total = psutil.virtual_memory().total
        except Exception as e:
            self.logger.error(f"[TraceML] WARNING: psutil failed to allocate RAM: {e}")

    def _warmup_cpu(self) -> None:
        # CPU usage measurement
        try:
            if self.process:
                self.process.cpu_percent(interval=None)
            self.cpu_count = psutil.cpu_count(logical=True) or 0
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: process.cpu_percent() initial call failed: {e}"
            )
            self.cpu_count = 0

    def _init_gpu(self) -> None:
        self.gpu_available = False
        self.gpu_count = 0
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            self.gpu_available = True

    def _sample_cpu(self):
        try:
            cpu_percent = float(self.process.cpu_percent(interval=None))
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: Failed to sample CPU usage from process CPU usage: {e}"
            )
            cpu_percent = 0.0
        return cpu_percent

    def _sample_ram(self):
        try:
            ram_percent = float(self.process.memory_info().rss)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: Failed to sample RAM usage from process RAM usage: {e}"
            )
            ram_percent = 0.0
        return ram_percent

    def _sample_gpu(self):
        """
        Return per-GPU memory information
        """
        if not self.gpu_available:
            return {}

        gpu_info = {}
        try:
            for i in range(self.gpu_count):
                try:
                    # Current device context
                    with torch.cuda.device(i):
                        used = torch.cuda.memory_allocated(i)
                        reserved = torch.cuda.memory_reserved(i)
                        total = torch.cuda.get_device_properties(i).total_memory

                    gpu_info[i] = {
                        "used": float(used),
                        "reserved": float(reserved),
                        "total": float(total),
                    }

                except Exception as e:
                    self.logger.error(
                        f"[TraceML] GPU {i} process memory read failed: {e}"
                    )
                    gpu_info[i] = {"used": 0.0, "reserved": 0.0, "total": 0.0}

        except Exception as outer:
            self.logger.error(f"[TraceML] Iterating GPUs failed: {outer}")
            return {}

        return gpu_info

    def sample(self):
        """
        Sample current CPU, RAM and GPU usage of the monitored process.
        Returns:
            envelope dict via BaseSampler helpers
        """
        try:
            cpu_pct = self._sample_cpu()
            ram_rss = self._sample_ram()
            gpu_mem = self._sample_gpu()

            record = {
                "timestamp": time.time(),
                "pid": self.pid,
                "cpu_logical_core_count": self.cpu_count,
                "cpu_percent": cpu_pct,
                "ram_used": ram_rss,
                "ram_total": self.ram_total,
                "gpu_available": self.gpu_available,
                "gpu_count": self.gpu_count,
                "gpu_raw": gpu_mem,
            }
            self.db.add_record("process", record)

        except Exception as e:
            self.logger.error(f"[TraceML] Process sampling error: {e}")
