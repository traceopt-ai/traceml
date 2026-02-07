"""
Time Event Schema (Shared Contract)

This module defines the **data contract** between TraceML's TimeSampler
and downstream compute layers (e.g., step-combined metrics).

Purpose
-------
- Provide a stable, explicit schema for step-level timing events
- Decouple sampler implementation from compute logic
- Avoid implicit assumptions about clocks or fields

Concept
-------
TimeSampler emits a *totally ordered* stream of resolved timing events
per rank. Each event is written as a single row into a DB table whose
name is the event name (`evt.name`).

Clocks
------
- CPU events use CPU wall-clock time.
- GPU events use CUDA event (stream) time.
- Both are stored in `duration_ms` and disambiguated by `is_gpu`.

Schema (per DB row)
------------------
{
    "timestamp": float,    # CPU end timestamp (seconds)
    "step": int,           # training step id
    "scope": str,          # logical scope label (e.g. "train_step")
    "device": str,         # "cpu" or "cuda:0"
    "is_gpu": bool,        # True if CUDA timing
    "duration_ms": float,  # milliseconds (CPU wall or CUDA stream)
}

Design guarantees
-----------------
- Flat, wire-friendly structure
- Deterministic ordering per rank
- No mixed-clock arithmetic
- Suitable for TCP transport and DB storage
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class TimeEventSample:
    """
    Canonical timing event sample.

    Represents a single resolved timing observation at step-level
    granularity. This schema is shared between sampler and compute layers.
    """
    sample_idx: int
    timestamp: float
    step: int
    scope: str
    device: str
    is_gpu: bool
    duration_ms: float

    def to_wire(self) -> Dict[str, Any]:
        """
        Convert the sample to a wire-friendly dictionary.
        """
        return {
            "seq": self.sample_idx,
            "timestamp": float(self.timestamp),
            "step": int(self.step),
            "scope": str(self.scope),
            "device": str(self.device),
            "is_gpu": bool(self.is_gpu),
            "duration_ms": float(self.duration_ms),
        }

    @staticmethod
    def from_wire(data: Dict[str, Any]) -> "TimeEventSample":
        """
        Reconstruct TimeEventSample from wire format.
        """
        return TimeEventSample(
            sample_idx=data["seq"],
            timestamp=float(data["timestamp"]),
            step=int(data["step"]),
            scope=str(data.get("scope", "")),
            device=str(data.get("device", "")),
            is_gpu=bool(data.get("is_gpu", False)),
            duration_ms=float(data.get("duration_ms", 0.0)),
        )
