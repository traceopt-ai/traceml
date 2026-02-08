"""
Step-level GPU memory schema.

This module defines the canonical data structures used to represent
step-scoped peak GPU memory metrics.

Design principles
-----------------
- Separation between:
    * internal representation (dataclasses)
    * wire representation (compact dict)
- Cheap conversions
- Stable keys for storage and transport
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class StepMemorySample:
    """
    Step-level peak memory snapshot.

    A single record corresponds to one training step for a given model/device.

    Units
    -----
    peak_allocated : bytes
    peak_reserved  : bytes
    """

    sample_idx: int
    timestamp: float
    model_id: Optional[int]
    device: Optional[str]
    step: Optional[int]
    peak_allocated: Optional[float]
    peak_reserved: Optional[float]

    def to_wire(self) -> Dict[str, Any]:
        """
        Convert to a network/storage-friendly wire format.

        Returns
        -------
        Dict[str, Any]
            Wire representation suitable for JSON storage/transport.
        """
        return {
            "seq": self.sample_idx,
            "ts": self.timestamp,
            "model_id": self.model_id,
            "device": self.device,
            "step": self.step,
            "peak_alloc": self.peak_allocated,
            "peak_resv": self.peak_reserved,
        }

    @staticmethod
    def from_wire(data: Dict[str, Any]) -> "StepMemorySample":
        """
        Reconstruct StepMemorySample from its wire representation.

        Parameters
        ----------
        data : Dict[str, Any]
            Wire dict produced by `to_wire()`.

        Returns
        -------
        StepMemorySample
            Reconstructed sample.
        """
        return StepMemorySample(
            sample_idx=data["seq"],
            timestamp=float(data.get("ts", 0.0)),
            model_id=data.get("model_id"),
            device=data.get("device"),
            step=data.get("step"),
            peak_allocated=data.get("peak_alloc"),
            peak_reserved=data.get("peak_resv"),
        )
