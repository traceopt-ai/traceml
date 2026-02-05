"""
Layer forward / backward execution time schema.

This module defines data structures used to represent
per-layer *execution time* observed during the forward or backward
pass of a PyTorch model.

Tracked metrics focus on *runtime execution cost*:
- Per-layer CPU execution time (milliseconds)
- Per-layer GPU execution time (milliseconds), if available
- Number of execution calls per layer

Semantics
---------
- One sample represents a single observation for a given:
    * model
    * training step
    * device
- Forward and backward passes share the same structure.
- GPU timings may resolve asynchronously and may be absent.

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
from typing import Dict, Any, List, Optional



@dataclass(frozen=True)
class LayerForwardBackwardTimePayload:
    """
    Per-layer execution time payload.

    This payload captures *aggregated* execution time per layer
    for a fully resolved forward or backward step.

    Invariants
    ----------
    - All lists have identical length
    - Ordering is deterministic and stable
    - Units:
        * cpu_time_ms : milliseconds
        * gpu_time_ms : milliseconds (None if unavailable)
        * n_calls     : integer count
    """

    layer_names: List[str]
    cpu_time_ms: List[float]
    gpu_time_ms: List[Optional[float]]
    n_calls: List[int]

    def to_wire(self) -> Dict[str, Any]:
        """
        Convert the payload to a wire-friendly representation.

        Wire format rationale
        ---------------------
        - Parallel lists instead of dicts for:
            * faster serialization
            * lower overhead
            * deterministic ordering
        - Optional GPU timings encoded as nulls
        """
        return {
            "layers": self.layer_names,
            "cpu_ms": self.cpu_time_ms,
            "gpu_ms": self.gpu_time_ms,
            "n_calls": self.n_calls,
        }

    @staticmethod
    def from_wire(data: Dict[str, Any]) -> "LayerForwardBackwardTimePayload":
        """
        Reconstruct LayerForwardBackwardTimePayload from wire representation.

        Parameters
        ----------
        data : Dict[str, Any]
            Wire-format dictionary produced by `to_wire()`.

        Returns
        -------
        LayerForwardBackwardTimePayload
            Reconstructed payload instance.
        """
        return LayerForwardBackwardTimePayload(
            layer_names=data["layers"],
            cpu_time_ms=data["cpu_ms"],
            gpu_time_ms=data["gpu_ms"],
            n_calls=data["n_calls"],
        )


@dataclass(frozen=True)
class LayerForwardBackwardTimeSample:
    """
    Layer-level forward/backward execution time snapshot.

    Represents a single, timestamped observation of per-layer
    execution time for a given model, step, and device.

    Notes
    -----
    - This schema is *step-level*, not architecture-level
    - One row corresponds to one fully resolved forward or backward step
    - Multiple calls per layer are aggregated
    - Semantic meaning (forward vs backward) is defined by table identity
    - Immutability prevents accidental mutation across threads
    """

    sample_idx: int
    timestamp: float

    model_id: int
    step: int
    device: str

    payload: LayerForwardBackwardTimePayload

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
            "cpu_ms": self.payload.cpu_time_ms,
            "gpu_ms": self.payload.gpu_time_ms,
            "n_calls": self.payload.n_calls,
        }

    @staticmethod
    def from_wire(data: Dict[str, Any]) -> "LayerForwardBackwardTimeSample":
        """
        Reconstruct LayerForwardBackwardTimeSample from wire representation.

        Parameters
        ----------
        data : Dict[str, Any]
            Wire-format dictionary produced by `to_wire()`.

        Returns
        -------
        LayerForwardBackwardTimeSample
            Reconstructed sample instance.
        """
        payload = LayerForwardBackwardTimePayload(
            layer_names=data["layers"],
            cpu_time_ms=data["cpu_ms"],
            gpu_time_ms=data["gpu_ms"],
            n_calls=data["n_calls"],
        )

        return LayerForwardBackwardTimeSample(
            sample_idx=data["seq"],
            timestamp=data["ts"],
            model_id=data["model_id"],
            step=data["step"],
            device=data["device"],
            payload=payload,
        )
