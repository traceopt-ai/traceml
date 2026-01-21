import psutil
import torch
import os
import time
from .base_sampler import BaseSampler
from traceml.distributed import get_ddp_info
from traceml.loggers.error_log import get_error_logger



class ProcessSampler(BaseSampler):
    """
    Sampler that tracks CPU and RAM usage of the current Python process
    and (by default) GPU memory for the device this process is using.
    DDP-friendly: reads only local_rank GPU instead of iterating all CUDA devices.
    """

    def __init__(self) -> None:
        self.sampler_name = "ProcessSampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)

        # Initiate
        self._init_process()
        self._init_ram()
        self._warmup_cpu()
        self._init_gpu()

        self.is_ddp, self.local_rank, self.world_size = get_ddp_info()
        self.rank = int(os.environ.get("RANK", "-1"))
        self.device_index = self._resolve_device_index()

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


    def _resolve_device_index(self) -> int:
        """
        Pick the single GPU device this process should sample.
        DDP: use LOCAL_RANK.
        Otherwise: use current_device if CUDA initialized, else 0.
        """
        if not self.gpu_available:
            return -1

        # If DDP detected, local_rank is the canonical device index
        if self.is_ddp and self.local_rank is not None and self.local_rank >= 0:
            return int(self.local_rank)

        # Non-DDP: prefer current device if CUDA context exists
        try:
            return int(torch.cuda.current_device())
        except Exception:
            return 0


    def _sample_cpu(self):
        try:
            return float(self.process.cpu_percent(interval=None))
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: Failed to sample CPU usage from process CPU usage: {e}"
            )
            return 0.0


    def _sample_ram(self):
        try:
            return float(self.process.memory_info().rss)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: Failed to sample RAM usage from process RAM usage: {e}"
            )
            return 0.0

    def _sample_gpu(self):
        """
        Return per-GPU memory information
        """
        if not self.gpu_available:
            return None, 0.0, 0.0, 0.0

        i = self.device_index
        if i is None or i < 0:
            return None, 0.0, 0.0, 0.0
        try:
            # Current device context
            with torch.cuda.device(i):
                used = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                total = torch.cuda.get_device_properties(i).total_memory
            return i, used, reserved, total
        except Exception as e:
            self.logger.error(
                f"[TraceML] GPU {i} process memory read failed: {e}"
            )
            return None, 0.0, 0.0, 0.0


    def sample(self):
        """
        Sample current CPU, RAM and GPU usage of the monitored process.
        Returns:
            envelope dict via BaseSampler helpers
        """
        try:
            cpu_pct = self._sample_cpu()
            ram_rss = self._sample_ram()
            gpu_i, gpu_used, gpu_reserved, gpu_total = self._sample_gpu()

            record = {
                "timestamp": time.time(),
                "pid": self.pid,
                "cpu_logical_core_count": self.cpu_count,
                "cpu_percent": cpu_pct,
                "ram_used": ram_rss,
                "ram_total": self.ram_total,
                "gpu_available": self.gpu_available,
                "gpu_count": self.gpu_count,
                "gpu_device_index": gpu_i,
                "gpu_mem_used": gpu_used,
                "gpu_mem_reserved": gpu_reserved,
                "gpu_mem_total": gpu_total,
            }
            self.db.add_record("process", record)

        except Exception as e:
            self.logger.error(f"[TraceML] Process sampling error: {e}")
