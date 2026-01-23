"""
System telemetry schema for TraceML.

This module defines the canonical data structures used to represent
system-level metrics (CPU, RAM, GPU) inside TraceML.

Design principles
-----------------
- Explicit, self-documenting schema via dataclasses
- Clear separation between:
    * internal representation (dataclasses)
    * wire representation (compact lists / dicts)
- Cheap, explicit conversions (no reflection, no recursion)
- Stable ordering for network transport
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import IntEnum

# ---------------------------------------------------------------------------
# GPU metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUMetrics:
    """
    Per-GPU telemetry snapshot.

    All values are point-in-time measurements for a single GPU.

    Units
    -----
    util : percent (0–100)
    mem_used : bytes
    mem_total : bytes
    temperature : degrees Celsius
    power_usage : watts
    power_limit : watts
    """

    util: float
    mem_used: float
    mem_total: float
    temperature: float
    power_usage: float
    power_limit: float

    def to_wire(self) -> List[float]:
        """
        Convert this GPU metric to a compact wire representation.

        Wire format (fixed order):
            [
              util,
              mem_used,
              mem_total,
              temperature,
              power_usage,
              power_limit,
            ]

        Returns
        -------
        List[float]
            Compact, allocation-light representation suitable for
            JSON serialization or binary transport.
        """
        return [
            self.util,
            self.mem_used,
            self.mem_total,
            self.temperature,
            self.power_usage,
            self.power_limit,
        ]

    @staticmethod
    def from_wire(data: List[float]) -> "GPUMetrics":
        """
        Reconstruct GPUMetrics from its wire representation.

        Parameters
        ----------
        data : List[float]
            Wire-format list produced by `to_wire()`.

        Returns
        -------
        GPUMetrics
            Reconstructed GPU metrics instance.
        """
        return GPUMetrics(
            util=data[0],
            mem_used=data[1],
            mem_total=data[2],
            temperature=data[3],
            power_usage=data[4],
            power_limit=data[5],
        )

# ---------------------------------------------------------------------------
# System snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemSample:
    """
    System-level telemetry snapshot.

    Represents a single timestamped snapshot of system-level
    metrics, including CPU, memory, and GPU state.

    Notes
    -----
    - GPU metrics are stored as an ordered list.
      The list index corresponds to the GPU index on the host.
    - This class is immutable (`frozen=True`) to prevent accidental mutation
      after sampling.
    """
    sample_idx: int
    timestamp: float
    cpu_percent: float
    ram_used: float
    ram_total: float
    gpu_available: bool
    gpu_count: int
    gpus: List[GPUMetrics]

    def to_wire(self) -> Dict[str, Any]:
        """
        Convert the system sample to a network-friendly wire format.

        Wire format rationale
        ---------------------
        - Top-level remains a dict for readability and extensibility
        - GPU metrics are encoded as a list of fixed-order lists
          to minimize payload size and serialization cost

        Returns
        -------
        Dict[str, Any]
            Wire representation suitable for transport or storage.
        """
        return {
            "seq": self.sample_idx,
            "ts": self.timestamp,
            "cpu": self.cpu_percent,
            "ram_used": self.ram_used,
            "ram_total": self.ram_total,
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count,
            "gpus": [gpu.to_wire() for gpu in self.gpus],
        }

    @staticmethod
    def from_wire(data: Dict[str, Any]) -> "SystemSample":
        """
        Reconstruct a SystemSample from its wire representation.

        Parameters
        ----------
        data : Dict[str, Any]
            Wire-format dictionary produced by `to_wire()`.

        Returns
        -------
        SystemSample
            Reconstructed system telemetry snapshot.
        """
        return SystemSample(
            sample_idx=data["seq"],
            timestamp=data["ts"],
            cpu_percent=data["cpu"],
            ram_used=data["ram_used"],
            ram_total=data["ram_total"],
            gpu_available=data["gpu_available"],
            gpu_count=data["gpu_count"],
            gpus=[GPUMetrics.from_wire(g) for g in data.get("gpus", [])],
        )