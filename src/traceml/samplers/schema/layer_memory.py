"""
Layer (parameter) memory schema for TraceML.

This module defines the canonical data structures used to represent
static, per-layer *parameter memory* of PyTorch models.

Tracked metrics focus on *model architecture*, not runtime state:
- Parameter memory per layer
- Total parameter memory
- Stable architecture identity via content signature

Design principles
-----------------
- Architecture-level
- One-time / low-frequency sampling
- Clear separation between:
    * internal representation (dataclasses)
    * wire representation (flat, list-based)
- Deterministic ordering for hashing and transport
- Optimized for TCP serialization / deserialization
"""

from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass(frozen=True)
class LayerMemoryPayload:
    """
    Per-layer parameter memory payload.

    This represents the *full architecture layout* of a model in terms
    of parameter memory only.

    Invariants
    ----------
    - layer_names and layer_param_bytes have identical length
    - ordering is deterministic and stable
    - units are bytes
    """

    layer_names: List[str]
    layer_param_bytes: List[float]

    def to_wire(self) -> Dict[str, Any]:
        """
        Convert layer payload to a wire-friendly representation.

        Wire format rationale
        ---------------------
        - Parallel lists instead of dicts for:
            * faster serialization
            * lower overhead
            * deterministic ordering
        """
        return {
            "layers": self.layer_names,
            "bytes": self.layer_param_bytes,
        }

    @staticmethod
    def from_wire(data: Dict[str, Any]) -> "LayerMemoryPayload":
        """
        Reconstruct LayerMemoryPayload from wire representation.
        """
        return LayerMemoryPayload(
            layer_names=data["layers"],
            layer_param_bytes=data["bytes"],
        )


@dataclass(frozen=True)
class LayerMemorySample:
    """
    Static layer-parameter memory snapshot for a model architecture.

    Represents a *single, deduplicated* model definition. This data
    is intended to be reused across steps, ranks, and runs.

    Notes
    -----
    - This schema is architecture-level
    - No process, rank, device, or step semantics
    - Immutability ensures safe sharing across threads and processes
    """

    sample_idx: int
    timestamp: float

    model_index: int
    model_signature: str

    total_param_bytes: float
    layer_count: int

    payload: LayerMemoryPayload

    def to_wire(self) -> Dict[str, Any]:
        """
        Convert the layer memory sample to a wire-friendly representation.

        Wire format rationale
        ---------------------
        - Flat top-level structure
        - Payload stored as compact parallel lists
        - No nested dict-of-dicts
        """
        return {
            "seq": self.sample_idx,
            "ts": self.timestamp,
            "model_index": self.model_index,
            "model_signature": self.model_signature,
            "total_param_bytes": self.total_param_bytes,
            "layer_count": self.layer_count,
            "layers": self.payload.layer_names,
            "layer_bytes": self.payload.layer_param_bytes,
        }

    @staticmethod
    def from_wire(data: Dict[str, Any]) -> "LayerMemorySample":
        """
        Reconstruct LayerMemorySample from wire representation.
        """
        payload = LayerMemoryPayload(
            layer_names=data["layers"],
            layer_param_bytes=data["layer_bytes"],
        )

        return LayerMemorySample(
            sample_idx=data["seq"],
            timestamp=data["ts"],
            model_index=data["model_index"],
            model_signature=data["model_signature"],
            total_param_bytes=data["total_param_bytes"],
            layer_count=data["layer_count"],
            payload=payload,
        )