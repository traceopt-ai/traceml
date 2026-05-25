"""
Process telemetry schema.

This module defines the canonical data structures used to represent
process-level metrics collected from the running Python process.

Tracked metrics focus on *what this process consumes*, not the entire system:
- CPU usage attributed to the process
- Resident memory (RSS)
- GPU memory usage for the device used by this process

Design principles
-----------------
- Clear separation between:
    * internal representation (dataclasses)
    * wire representation (compact dicts)
- Cheap, explicit conversions
- Stable field names for storage and transport

"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Process GPU metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessGPUMetrics:
    """
    GPU memory usage attributed to a single process on a single device.

    This reflects memory allocated and reserved by the current process
    on the GPU it is executing on (typically one device per process).

    Units
    -----
    device_index : CUDA device index
    mem_used : bytes (allocated by the process)
    mem_reserved : bytes (reserved by the process)
    mem_total : bytes (total device memory)
    """

    device_index: int
    mem_used: float
    mem_reserved: float
    mem_total: float

    def to_wire(self) -> Dict[str, Any]:
        """
        Convert process GPU metrics to a wire-friendly representation.

        Returns
        -------
        Dict[str, Any]
            Dictionary suitable for transport or storage.
        """
        return {
            "device": self.device_index,
            "mem_used": self.mem_used,
            "mem_reserved": self.mem_reserved,
            "mem_total": self.mem_total,
        }

    @staticmethod
    def from_wire(data: Dict[str, Any]) -> "ProcessGPUMetrics":
        """
        Reconstruct ProcessGPUMetrics from its wire representation.

        Parameters
        ----------
        data : Dict[str, Any]
            Wire-format dictionary produced by `to_wire()`.

        Returns
        -------
        ProcessGPUMetrics
            Reconstructed GPU metrics instance.
        """
        return ProcessGPUMetrics(
            device_index=data["device"],
            mem_used=data["mem_used"],
            mem_reserved=data["mem_reserved"],
            mem_total=data["mem_total"],
        )


# ---------------------------------------------------------------------------
# Process snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessSample:
    """
    Process-level telemetry snapshot.

    Represents a single timestamped observation of resource usage
    attributed to the running Python process.

    Notes
    -----
    - CPU usage is process-relative (not system-wide)
    - RAM usage reflects resident set size (RSS)
    - GPU metrics, if present, refer to the single device used by the process
    - This class is immutable (`frozen=True`) to prevent accidental mutation
    """

    sample_idx: int
    timestamp: float
    pid: int

    cpu_percent: float
    cpu_logical_core_count: int

    ram_used: float
    ram_total: float

    gpu_available: bool
    gpu_count: int
    gpu: Optional[ProcessGPUMetrics]

    def to_wire(self) -> Dict[str, Any]:
        """
        Convert the process sample to a wire-friendly representation.

        Wire format rationale
        ---------------------
        - Flat dictionary for readability and ease of inspection
        - GPU metrics embedded as a nested object (single-device)

        Returns
        -------
        Dict[str, Any]
            Wire representation suitable for transport or storage.
        """
        return {
            "seq": self.sample_idx,
            "ts": self.timestamp,
            "pid": self.pid,
            "cpu": self.cpu_percent,
            "cpu_cores": self.cpu_logical_core_count,
            "ram_used": self.ram_used,
            "ram_total": self.ram_total,
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count,
            "gpu": self.gpu.to_wire() if self.gpu else None,
        }

    @staticmethod
    def from_wire(data: Dict[str, Any]) -> "ProcessSample":
        """
        Reconstruct a ProcessSample from its wire representation.

        Parameters
        ----------
        data : Dict[str, Any]
            Wire-format dictionary produced by `to_wire()`.

        Returns
        -------
        ProcessSample
            Reconstructed process telemetry snapshot.
        """
        gpu_data = data.get("gpu")

        return ProcessSample(
            sample_idx=data["seq"],
            timestamp=data["ts"],
            pid=data["pid"],
            cpu_percent=data["cpu"],
            cpu_logical_core_count=data["cpu_cores"],
            ram_used=data["ram_used"],
            ram_total=data["ram_total"],
            gpu_available=data["gpu_available"],
            gpu_count=data["gpu_count"],
            gpu=ProcessGPUMetrics.from_wire(gpu_data) if gpu_data else None,
        )
