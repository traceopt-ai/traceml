from __future__ import annotations

import os
import time
from typing import Optional

import psutil
import torch
import torch.distributed as dist

from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.runtime_context import resolve_runtime_context
from traceml.samplers.schema.process import ProcessGPUMetrics, ProcessSample


class ProcessSampler(BaseSampler):
    """
    Process-level telemetry sampler.

    This sampler tracks resource usage attributed to the current Python process:
    - CPU utilization (process-relative)
    - Resident memory usage (RSS)
    - GPU memory usage for the single CUDA device used by this process

    DDP notes
    ---------
    In DistributedDataParallel, touching CUDA before `dist.init_process_group()`
    has completed can lead to hangs during setup. This sampler therefore keeps
    GPU sampling lazy and only touches CUDA when it is safe to do so.

    Guarantees
    ----------
    - Sampling failures never propagate
    - Partial data is acceptable
    - Overhead is negligible compared to training
    """

    def __init__(self) -> None:
        super().__init__(
            sampler_name="ProcessSampler",
            table_name="ProcessTable",
        )

        self.sample_idx = 0

        self._init_process()
        self._init_ram()
        self._warmup_cpu()
        self._init_ddp_context()
        self._init_gpu_state()

    def _init_process(self) -> None:
        """
        Attach to the current Python process via psutil.
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
        Read total system RAM.
        """
        self.ram_total = 0.0
        try:
            self.ram_total = float(psutil.virtual_memory().total)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: psutil RAM query failed: {e}"
            )

    def _warmup_cpu(self) -> None:
        """
        Warm up process CPU sampling.
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

    def _init_ddp_context(self) -> None:
        """
        Detect whether the current process was launched as part of a
        multi-process run.

        We do not rely on `dist.is_initialized()` here because distributed
        initialization may not have completed yet.
        """
        ctx = resolve_runtime_context()

        self.local_rank = ctx.local_rank
        self.world_size = ctx.world_size
        self.rank = ctx.rank
        self.is_ddp_intended = ctx.is_ddp_intended

        if self.is_ddp_intended and self.local_rank == -1:
            self.local_rank = 0

    def _init_gpu_state(self) -> None:
        """
        Initialize GPU-related state without touching CUDA.
        """
        self.gpu_available: Optional[bool] = None
        self.gpu_count: int = 0
        self.device_index: Optional[int] = None

    def _sample_cpu(self) -> float:
        """
        Return process CPU utilization as a percentage.
        """
        try:
            return (
                float(self.process.cpu_percent(interval=None))
                if self.process
                else 0.0
            )
        except Exception as e:
            self.logger.error(f"[TraceML] WARNING: CPU sample failed: {e}")
            return 0.0

    def _sample_ram(self) -> float:
        """
        Return process resident memory (RSS) in bytes.
        """
        try:
            return (
                float(self.process.memory_info().rss) if self.process else 0.0
            )
        except Exception as e:
            self.logger.error(f"[TraceML] WARNING: RAM sample failed: {e}")
            return 0.0

    def _cuda_safe_to_touch(self) -> bool:
        """
        Return True if safe to call torch.cuda.* in this process.
        """
        if not self.is_ddp_intended:
            return True
        if not dist.is_available():
            return False
        return dist.is_initialized()

    def _ensure_cuda_device(self) -> None:
        """
        Ensure a CUDA device is selected for this rank.
        """
        if self.device_index is not None:
            return

        device_index = (
            self.local_rank
            if self.is_ddp_intended and self.local_rank != -1
            else 0
        )

        torch.cuda.set_device(device_index)
        self.device_index = device_index

        if self.gpu_available is None:
            self.gpu_available = bool(torch.cuda.is_available())
            self.gpu_count = (
                int(torch.cuda.device_count()) if self.gpu_available else 0
            )

    def _sample_gpu(self) -> Optional[ProcessGPUMetrics]:
        """
        Sample GPU memory usage for the device used by this process.
        """
        if not self._cuda_safe_to_touch():
            return None

        try:
            self._ensure_cuda_device()
            if not self.gpu_available:
                return None

            i = int(self.device_index or 0)

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
            self.logger.error(f"[TraceML] GPU process memory read failed: {e}")
            return None

    def sample(self) -> None:
        """
        Collect a single process-level telemetry snapshot.
        """
        self.sample_idx += 1
        try:
            gpu_metrics = self._sample_gpu()

            gpu_available = (
                bool(self.gpu_available)
                if self.gpu_available is not None
                else False
            )
            gpu_count = int(self.gpu_count) if self.gpu_available else 0

            sample = ProcessSample(
                sample_idx=self.sample_idx,
                timestamp=time.time(),
                pid=self.pid,
                cpu_percent=self._sample_cpu(),
                cpu_logical_core_count=self.cpu_count,
                ram_used=self._sample_ram(),
                ram_total=self.ram_total,
                gpu_available=gpu_available,
                gpu_count=gpu_count,
                gpu=gpu_metrics,
            )
            self._add_record(sample.to_wire())

        except Exception as e:
            self.logger.error(f"[TraceML] Process sampling error: {e}")
