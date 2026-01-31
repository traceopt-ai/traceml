import os
import time
from typing import Optional

import psutil
import torch

from traceml.samplers.base_sampler import BaseSampler
from traceml.loggers.error_log import get_error_logger
from traceml.samplers.schema.process import ProcessSample, ProcessGPUMetrics


class ProcessSampler(BaseSampler):
    """
    Process-level telemetry sampler.

    This sampler tracks resource usage attributed to the *current Python process*:
    - CPU utilization (process-relative)
    - Resident memory usage (RSS)
    - GPU memory usage for the single CUDA device used by this process

    Scope
    -----
    - Single-process, single-node
    - Single GPU per process

    Guarantees
    -----------------
    - Sampling failures never propagate
    - Partial data is acceptable
    - Overhead is negligible compared to training
    """

    def __init__(self) -> None:
        self.name = "Process"
        self.sampler_name = self.name+"Sampler"
        self.table_name = self.name+"Table"
        super().__init__(sampler_name=self.sampler_name)
        self.sample_idx = 0

        self.logger = get_error_logger(self.sampler_name)

        # Initiate
        self._init_process()
        self._init_ram()
        self._warmup_cpu()
        self._init_gpu()


    def _init_process(self) -> None:
        """
        Attach to the current Python process via psutil.

        Failure degrades sampling gracefully (CPU/RAM reported as zero).
        """
        try:
            self.pid = os.getpid()
            self.process = psutil.Process(self.pid)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: Failed to attach to process {os.getpid()}: {e}"
            )
            self.pid = -1
            self.process = None

    def _init_ram(self) -> None:
        """
        Total system RAM.
        """
        self.ram_total = 0.0
        try:
            self.ram_total = psutil.virtual_memory().total
        except Exception as e:
            self.logger.error(f"[TraceML] WARNING: psutil failed to allocate RAM: {e}")

    def _warmup_cpu(self) -> None:
        """
        Warm up process CPU sampling.

        psutil requires an initial call to avoid blocking
        on the first real measurement.
        """
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
        """
        Detect GPU availability
        """
        self.gpu_available = torch.cuda.is_available()
        if not self.gpu_available:
            self.gpu_count = 0
            return
        # Number of visible GPUs (after CUDA_VISIBLE_DEVICES)
        self.gpu_count = torch.cuda.device_count()
        self.device_index = None

    def _sample_cpu(self):
        """Return process CPU utilization as a percentage."""
        try:
            return (
                float(self.process.cpu_percent(interval=None)) if self.process else 0.0
            )
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: Failed to sample CPU usage from process CPU usage: {e}"
            )
            return 0.0

    def _sample_ram(self):
        """Return process resident memory (RSS) in bytes."""
        try:
            return float(self.process.memory_info().rss) if self.process else 0.0
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: Failed to sample RAM usage from process RAM usage: {e}"
            )
            return 0.0

    def _ensure_cuda_device(self):
        if not self.gpu_available or self.device_index is not None:
            return

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        self.device_index = local_rank

    def _sample_gpu(self) -> Optional[ProcessGPUMetrics]:
        """
        Sample GPU memory usage for the device used by this process.

        Returns
        -------
        Optional[ProcessGPUMetrics]
            GPU metrics if available, otherwise None.
        """
        if not self.gpu_available:
            return None

        self._ensure_cuda_device()
        i = self.device_index
        try:
            # Current device context
            with torch.cuda.device(i):
                used = float(torch.cuda.memory_allocated(i))
                reserved = float(torch.cuda.memory_reserved(i))
                total = float(torch.cuda.get_device_properties(i).total_memory)

            return ProcessGPUMetrics(
                device_index=i,
                mem_used=used,
                mem_reserved=reserved,
                mem_total=total,
            )

        except Exception as e:
            self.logger.error(f"[TraceML] GPU {i} process memory read failed: {e}")
            return None

    def sample(self):
        """
        Collect a single process-level telemetry snapshot.

        The sample is converted to its wire representation and stored
        in the local database. All exceptions are caught and logged.
        """
        self.sample_idx += 1
        try:
            sample = ProcessSample(
                sample_idx=self.sample_idx,
                timestamp=time.time(),
                pid=self.pid,
                cpu_percent=self._sample_cpu(),
                cpu_logical_core_count=self.cpu_count,
                ram_used=self._sample_ram(),
                ram_total=self.ram_total,
                gpu_available=self.gpu_available,
                gpu_count=self.gpu_count,
                gpu=self._sample_gpu(),
            )
            self.db.add_record(self.table_name, sample.to_wire())

        except Exception as e:
            self.logger.error(f"[TraceML] Process sampling error: {e}")
