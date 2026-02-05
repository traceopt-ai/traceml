"""
Layer forward / backward activation memory schema for TraceML.

This module defines the canonical data structures used to represent
per-layer activation memory observed during the forward or backward
pass of a PyTorch model.

Tracked metrics focus on *runtime activation memory*:
- Per-layer activation / gradient memory (bytes)
- Model- and step-scoped observations

Semantics
---------
- One sample represents a single observation for a given:
    * model
    * training step
    * device
- Forward and backward passes share the same structure.
  Semantic differences are expressed by table identity, not schema.

Design principles
-----------------
- Step-level, event-driven telemetry
- Clear separation between:
    * internal representation (dataclasses)
    * wire representation (flat, list-based)
- Deterministic ordering for transport and compute
- Optimized for TCP serialization / deserialization
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass(frozen=True)
class LayerForwardBackwardMemoryPayload:
    """
    Per-layer activation memory payload.

    This payload captures memory attributed to each layer during
    either the forward or backward pass.

    Invariants
    ----------
    - layer_names and layer_memory_bytes have identical length
    - ordering is deterministic and stable
    - units are bytes
    """

    layer_names: List[str]
    layer_memory_bytes: List[float]

    def to_wire(self) -> Dict[str, Any]:
        """
        Convert the payload to a wire-friendly representation.

        Wire format rationale
        ---------------------
        - Parallel lists instead of dicts for:
            * faster serialization
            * lower overhead
            * deterministic ordering
        """
        return {
            "layers": self.layer_names,
            "bytes": self.layer_memory_bytes,
        }

    @staticmethod
    def from_wire(data: Dict[str, Any]) -> "LayerForwardBackwardMemoryPayload":
        """
        Reconstruct LayerForwardBackwardPayload from wire representation.

        Parameters
        ----------
        data : Dict[str, Any]
            Wire-format dictionary produced by `to_wire()`.

        Returns
        -------
        LayerForwardBackwardPayload
            Reconstructed payload instance.
        """
        return LayerForwardBackwardMemoryPayload(
            layer_names=data["layers"],
            layer_memory_bytes=data["bytes"],
        )


@dataclass(frozen=True)
class LayerForwardBackwardMemorySample:
    """
    Layer-level forward/backward activation memory snapshot.

    Represents a single, timestamped observation of per-layer
    activation memory for a given model, step, and device.

    Notes
    -----
    - This schema is *step-level*, not architecture-level
    - One row corresponds to one forward or backward event
    - Semantic meaning (forward vs backward) is defined by table identity
    - Immutability prevents accidental mutation across threads
    """

    sample_idx: int
    timestamp: float

    model_id: int
    step: int
    device: str

    payload: LayerForwardBackwardMemoryPayload

    def to_wire(self) -> Dict[str, Any]:
        """
        Convert the sample to a wire-friendly representation.

        Wire format rationale
        ---------------------
        - Flat top-level structure
        - Payload stored as compact parallel lists
        - No nested dict-of-dicts
        """
        return {
            "seq": self.sample_idx,
            "ts": self.timestamp,
            "model_id": self.model_id,
            "step": self.step,
            "device": self.device,
            "layers": self.payload.layer_names,
            "layer_bytes": self.payload.layer_memory_bytes,
        }

    @staticmethod
    def from_wire(data: Dict[str, Any]) -> "LayerForwardBackwardMemorySample":
        """
        Reconstruct LayerForwardBackwardSample from wire representation.

        Parameters
        ----------
        data : Dict[str, Any]
            Wire-format dictionary produced by `to_wire()`.

        Returns
        -------
        LayerForwardBackwardSample
            Reconstructed sample instance.
        """
        payload = LayerForwardBackwardMemoryPayload(
            layer_names=data["layers"],
            layer_memory_bytes=data["layer_bytes"],
        )

        return LayerForwardBackwardMemorySample(
            sample_idx=data["seq"],
            timestamp=data["ts"],
            model_id=data["model_id"],
            step=data["step"],
            device=data["device"],
            payload=payload,
        )