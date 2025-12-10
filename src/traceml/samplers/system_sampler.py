import psutil
import time
from typing import Dict
from .base_sampler import BaseSampler
from traceml.loggers.error_log import get_error_logger
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


class SystemSampler(BaseSampler):
    """
    Sampler that tracks CPU RAM and GPU usage over time using psutil.
    Collects usage percentages periodically and exposes live snapshots
    and statistical summaries.
    Keeps per-GPU history internally, and exposes live metrics
    (min/max/avg/imbalance) as well as a summary (global peak, lowest non-zero,
    average, variance).
    """

    def __init__(self) -> None:
        self.sampler_name = "SystemSampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)
        self.db.create_table("system")

        self._init_cpu()
        self._init_ram()
        self._init_gpu()

    def _init_cpu(self):
        # Initialize psutil.cpu_percent for non-blocking calls
        self.cpu_logical_core_count = 0
        try:
            psutil.cpu_percent(interval=None)
            self.cpu_logical_core_count = psutil.cpu_count(logical=True)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: psutil.cpu_percent initial call failed: {e}"
            )

    def _init_ram(self):
        self.ram_total_memory = 0
        try:
            self.ram_total_memory = psutil.virtual_memory().total
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: psutil.virtual_memory initial call failed: {e}"
            )

    def _init_gpu(self):
        # GPU setup
        self.gpu_available = False
        self.gpu_count = 0
        try:
            nvmlInit()
            self.gpu_count = nvmlDeviceGetCount()
            self.gpu_available = self.gpu_count > 0
        except NVMLError as e:
            self.logger.error(f"[TraceML] WARNING: GPU not available: {e}")

    def _sample_cpu(self):
        """Sample CPU usage and update history."""
        try:
            return psutil.cpu_percent(interval=None)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: psutil.cpu_percent initial call failed: {e}"
            )
            return 0.0

    def _sample_ram(self):
        """Sample RAM usage and update history."""
        try:
            mem = psutil.virtual_memory()
            return float(mem.used)
        except Exception as e:
            self.logger.error(
                f"[TraceML] WARNING: psutil.virtual_memory initial call failed: {e}"
            )
            return 0.0

    def _sample_gpu(self):
        """
        Sample GPU usage and update histories. Returns dict per-GPU metrics:
        {gpu_id: {"util": ..., "mem_used": ..., "mem_total": ...}}
        """
        if not self.gpu_available:
            return {}

        gpu_info: Dict[int, Dict[str, float]] = {}
        for i in range(self.gpu_count):
            try:
                handle = nvmlDeviceGetHandleByIndex(i)
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                power_limit = nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                gpu_info[i] = {
                    "util": float(util.gpu),
                    "mem_used": float(mem.used),
                    "mem_total": float(mem.total),
                    "temperature": float(temp),
                    "power_usage": float(power),
                    "power_limit": float(power_limit),
                }

            except Exception as e:
                self.logger.error(f"[TraceML] GPU {i} sampling failed: {e}")
                gpu_info[i] = {
                    "util": 0.0,
                    "mem_used": 0.0,
                    "mem_total": 0.0,
                    "temperature": 0.0,
                    "power_usage": 0.0,
                    "power_limit": 0.0,
                }

        return gpu_info

    def sample(self):
        """
        Take one system snapshot and return backward-compatible dict.
        """
        try:
            cpu = self._sample_cpu()
            ram_used = self._sample_ram()
            gpu_raw = self._sample_gpu()

            record = {
                "timestamp": time.time(),
                "cpu_percent": cpu,
                "ram_used": ram_used,
                "ram_total": self.ram_total_memory,
                "gpu_available": self.gpu_available,
                "gpu_count": self.gpu_count,
                "gpu_raw": gpu_raw,
            }
            self.db.add_record("system", record)

        except Exception as e:
            self.logger.error(f"[TraceML] System sampling error: {e}")
