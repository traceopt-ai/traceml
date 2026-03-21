import os
import time
from typing import Optional

import psutil
import torch
import torch.distributed as dist

from traceml.loggers.error_log import get_error_logger
from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.schema.process import ProcessGPUMetrics, ProcessSample


class ProcessSampler(BaseSampler):
    """
    Process-level telemetry sampler.

    This sampler tracks resource usage attributed to the *current Python process*:
    - CPU utilization (process-relative)
    - Resident memory usage (RSS)
    - GPU memory usage for the single CUDA device used by this process

    DDP notes
    ------------------
    In DistributedDataParallel (DDP), touching CUDA (creating a context, setting
    device, querying CUDA properties) before `dist.init_process_group()` has
    completed can lead to hangs during DDP/NCCL initialization.

    To avoid this, this sampler:
    - Never calls `torch.cuda.*` during __init__.
    - In multi-process (WORLD_SIZE > 1), skips GPU sampling until
      `dist.is_initialized()` becomes True.

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
        self.sampler_name = self.name + "Sampler"
        self.table_name = self.name + "Table"
        super().__init__(sampler_name=self.sampler_name)

        self.sample_idx = 0
        self.logger = get_error_logger(self.sampler_name)

        # Initialize non-CUDA state
        self._init_process()
        self._init_ram()
        self._warmup_cpu()
        self._init_ddp_context()
        self._init_gpu_state()

    def _init_process(self) -> None:
        """Attach to the current Python process via psutil."""
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
        """Read total system RAM."""
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

    def _init_ddp_context(self) -> None:
        """
        Detect  if in a multi-process launch.

        We intentionally do not use `dist.is_initialized()` here for gating
        CUDA usage during initialization because `init_process_group()` may not
        have run yet. Instead, we use environment variables that launchers set.
        """
        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "-1"))

        # "Intended DDP" if launcher indicates multiple processes.
        self.is_ddp_intended = self.world_size > 1

        # Normalize local_rank for common cases.
        if self.is_ddp_intended and self.local_rank == -1:
            # torchrun sets LOCAL_RANK; mp.spawn may not.
            # We avoid torch.cuda.device_count() here (CUDA touch).
            self.local_rank = 0

    def _init_gpu_state(self) -> None:
        """
        Initialize GPU-related state WITHOUT touching CUDA.

        Keep GPU sampling "lazy": only after DDP init (if applicable) do we
        call torch.cuda.* and resolve device indices.
        """
        self.gpu_available: Optional[bool] = (
            None  # Unknown until we safely touch CUDA
        )
        self.gpu_count: int = 0
        self.device_index: Optional[int] = None

    def _sample_cpu(self) -> float:
        """Return process CPU utilization as a percentage."""
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
        """Return process resident memory (RSS) in bytes."""
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

        - Single-process: safe (no DDP/NCCL init to interfere with).
        - Multi-process: safe only after dist.init_process_group() completes.
        """
        if not self.is_ddp_intended:
            return True
        # If distributed isn't even available, treat as not safe to touch CUDA here
        # (but in practice torchrun uses dist). This is conservative.
        if not dist.is_available():
            return False
        return dist.is_initialized()

    def _ensure_cuda_device(self) -> None:
        """
        Ensure a CUDA device is selected for this rank.

        Called only when `_cuda_safe_to_touch()` is True.
        """
        if self.device_index is not None:
            return

        # Decide device index.
        # - In DDP, use LOCAL_RANK (torchrun maps each proc to one GPU).
        # - Otherwise default to device 0.
        i = (
            self.local_rank
            if self.is_ddp_intended and self.local_rank != -1
            else 0
        )

        torch.cuda.set_device(i)
        self.device_index = i

        # Populate gpu availability/count lazily (now that touching CUDA is safe).
        if self.gpu_available is None:
            self.gpu_available = bool(torch.cuda.is_available())
            self.gpu_count = (
                int(torch.cuda.device_count()) if self.gpu_available else 0
            )

    def _sample_gpu(self) -> Optional[ProcessGPUMetrics]:
        """
        Sample GPU memory usage for the device used by this process.

        Returns None if:
        - CUDA isn't safe to touch yet (DDP init not complete), or
        - CUDA isn't available, or
        - an error occurs (logged and degraded gracefully).
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
            # If CUDA sampling fails once, we still keep CPU/RAM sampling alive.
            self.logger.error(f"[TraceML] GPU process memory read failed: {e}")
            return None

    def sample(self) -> None:
        """
        Collect a single process-level telemetry snapshot.

        The sample is converted to its wire representation and stored
        in the local database. All exceptions are caught and logged.
        """
        self.sample_idx += 1
        try:
            gpu_metrics = self._sample_gpu()

            # If we haven't safely touched CUDA yet, keep gpu_available/gpu_count stable.
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
            self.db.add_record(self.table_name, sample.to_wire())

        except Exception as e:
            self.logger.error(f"[TraceML] Process sampling error: {e}")
