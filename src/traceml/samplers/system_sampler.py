"""
System-level telemetry sampler.

This module defines a lightweight sampler that periodically captures
host and GPU resource metrics during training or inference.

Collected metrics include:
- CPU utilization
- RAM usage
- Per-GPU utilization, memory, temperature, and power

The sampler is designed to be:
- Non-intrusive (never raises, never blocks training)
- Robust to partial failures (e.g. missing GPUs)
- Cheap enough to run continuously
"""

import time
from typing import List

import psutil
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetPowerManagementLimit,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetCount,
    NVMLError,
    NVML_TEMPERATURE_GPU,
)

from traceml.loggers.error_log import get_error_logger
from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.schema.system import SystemSample, GPUMetrics


class SystemSampler(BaseSampler):
    """
    Periodic sampler for system resource metrics.

    This sampler captures a point-in-time snapshot of system state and
    stores it in the local TraceML database. It is intended to be invoked
    repeatedly by the runtime (e.g., once per step or on a timer).

    Guarantees
    ----------
    - Sampling failures are logged and never propagated
    - Partial data is acceptable (e.g., some GPUs failing)
    - Ordering of GPU metrics is stable (index == GPU id)
    """

    def __init__(self) -> None:
        """
        Initialize the system sampler.

        Performs one-time setup for CPU, RAM, and GPU sampling. Failures
        during initialization are logged but do not prevent the sampler
        from operating in a degraded mode.
        """
        self.name = "System"
        self.sampler_name = self.name+"Sampler"
        self.table_name = self.name+"Table"
        super().__init__(sampler_name=self.sampler_name)

        # We are sampling and sending only latest values
        self.sender.max_rows_per_flush = 1
        self.sample_idx = 0

        # Dedicated error logger (ERROR-level only)
        self.logger = get_error_logger(self.sampler_name)

        # One-time initialization
        self._init_cpu()
        self._init_ram()
        self._init_gpu()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_cpu(self) -> None:
        """
        Prepare CPU sampling via psutil.

        psutil.cpu_percent() requires an initial warm-up call to avoid
        blocking on the first real measurement.
        """
        self.cpu_logical_core_count = 0
        try:
            psutil.cpu_percent(interval=None)
            self.cpu_logical_core_count = psutil.cpu_count(logical=True)
        except Exception as e:
            self.logger.error(f"[TraceML] CPU sampler initialization failed: {e}")

    def _init_ram(self) -> None:
        """
        Prepare RAM sampling.

        Stores total system memory for reuse across samples.
        """
        self.ram_total_memory = 0.0
        try:
            self.ram_total_memory = float(psutil.virtual_memory().total)
        except Exception as e:
            self.logger.error(f"[TraceML] RAM sampler initialization failed: {e}")

    def _init_gpu(self) -> None:
        """
        Prepare GPU sampling via NVML.

        GPU sampling is optional. If NVML initialization fails, the sampler
        continues operating with CPU and RAM metrics only.
        """
        self.gpu_available = False
        self.gpu_count = 0

        try:
            nvmlInit()
            self.gpu_count = nvmlDeviceGetCount()
            self.gpu_available = self.gpu_count > 0
        except NVMLError as e:
            self.logger.error(
                f"[TraceML] NVML initialization failed (GPU unavailable): {e}"
            )

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_cpu(self) -> float:
        """
        Sample current CPU utilization.

        Returns
        -------
        float
            CPU usage percentage in the range [0, 100].
            Returns 0.0 on failure.
        """
        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception as e:
            self.logger.error(f"[TraceML] CPU sampling failed: {e}")
            return 0.0

    def _sample_ram(self) -> float:
        """
        Sample current RAM usage.

        Returns
        -------
        float
            Used system memory in bytes.
            Returns 0.0 on failure.
        """
        try:
            return float(psutil.virtual_memory().used)
        except Exception as e:
            self.logger.error(f"[TraceML] RAM sampling failed: {e}")
            return 0.0

    def _sample_gpus(self) -> List[GPUMetrics]:
        """
        Sample all available GPUs.

        Returns
        -------
        List[GPUMetrics]
            Ordered list of per-GPU metrics. The list index corresponds
            to the GPU index on the host.

        Notes
        -----
        - Failure to sample one GPU does not affect others.
        - A zeroed placeholder is inserted to preserve index alignment.
        """
        if not self.gpu_available:
            return []

        gpus: List[GPUMetrics] = []

        for gpu_id in range(self.gpu_count):
            try:
                handle = nvmlDeviceGetHandleByIndex(gpu_id)

                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                power_limit = nvmlDeviceGetPowerManagementLimit(handle) / 1000.0

                gpus.append(
                    GPUMetrics(
                        util=float(util.gpu),
                        mem_used=float(mem.used),
                        mem_total=float(mem.total),
                        temperature=float(temp),
                        power_usage=float(power),
                        power_limit=float(power_limit),
                    )
                )

            except Exception as e:
                self.logger.error(f"[TraceML] GPU {gpu_id} sampling failed: {e}")

                # Preserve GPU ordering even on failure
                gpus.append(
                    GPUMetrics(
                        util=0.0,
                        mem_used=0.0,
                        mem_total=0.0,
                        temperature=0.0,
                        power_usage=0.0,
                        power_limit=0.0,
                    )
                )

        return gpus

    def sample(self) -> None:
        """
        Collect a single system snapshot and store it in the database.

        This method is safe to call frequently. All exceptions are caught
        and logged to ensure sampling never interferes with training.
        """
        self.sample_idx += 1
        try:
            sample = SystemSample(
                sample_idx=self.sample_idx,
                timestamp=time.time(),
                cpu_percent=self._sample_cpu(),
                ram_used=self._sample_ram(),
                ram_total=self.ram_total_memory,
                gpu_available=self.gpu_available,
                gpu_count=self.gpu_count,
                gpus=self._sample_gpus(),
            )

            # Store the wire representation to keep DB and transport
            # independent of Python-specific objects.
            self.db.add_record(self.table_name, sample.to_wire())

        except Exception as e:
            # Absolute safety net: this should never crash the runtime
            self.logger.error(f"[TraceML] SystemSampler failed to collect sample: {e}")
