"""
Step Time Event Schema (Shared Contract)

This module defines the **data contract** between TraceML's StepTimeSampler
and downstream compute/UI layers.

Purpose
-------
- Provide a stable, explicit schema for step-level timing events
- Decouple sampler implementation from compute logic
- Avoid implicit assumptions about clocks or fields

Concept
-------
StepTimeSampler emits a *totally ordered* stream of **one record per step**
(per rank). A record is written into a single DB table (e.g. "step_time").

Each record contains:
- step metadata (seq, timestamp, step)
- a nested event map holding aggregated timings for that step

Aggregation rules (produced by sampler)
---------------------------------------
- Events are grouped by (event_name, device).
- duration_ms is SUM'ed across repeated occurrences within a step
  (e.g., gradient accumulation regions).
- n_calls counts how many occurrences were aggregated.

Clocks
------
- CPU events use CPU wall-clock time.
- GPU events use CUDA event (stream) time.
- Both are stored in `duration_ms` and disambiguated by `is_gpu`.

Schema (per DB row)
------------------
{
    "seq": int,              # monotonically increasing sequence per rank
    "timestamp": float,      # step timestamp (seconds); typically max cpu_end in step
    "step": int,             # training step id
    "events": {              # nested map: events[event_name][device] = stats
        "<event_name>": {
            "<device>": {
                "is_gpu": bool,
                "duration_ms": float,
                "n_calls": int
            },
            ...
        },
        ...
    }
}
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping

# Type alias for readability and to keep the contract explicit.
# events[event_name][device] = {"is_gpu": bool, "duration_ms": float, "n_calls": int}
StepEventMap = Dict[str, Dict[str, Dict[str, Any]]]


@dataclass(frozen=True)
class StepTimeEventSample:
    """
    Canonical *per-step* timing sample.

    This is the shared contract between StepTimeSampler and all downstream
    consumers. It intentionally stores the aggregated events as a single
    nested mapping to guarantee step alignment and reduce write overhead
    (one DB insert per step).

    Notes
    -----
    - `events` must be JSON/msgpack-serializable (str keys; bool/int/float values).
    - `timestamp` is in seconds (CPU wall clock), typically max cpu_end in step.
    - Consumers should treat unknown keys as forward-compatible extensions.
    """

    seq: int
    timestamp: float
    step: int
    events: StepEventMap

    def to_wire(self) -> Dict[str, Any]:
        """
        Convert this sample into a wire-friendly dictionary.

        This format is suitable for TCP transport (JSON/msgpack)
        """
        return {
            "seq": int(self.seq),
            "timestamp": float(self.timestamp),
            "step": int(self.step),
            "events": self.events,
        }

    @staticmethod
    def from_wire(data: Mapping[str, Any]) -> "StepTimeEventSample":
        """
        Reconstruct StepTimeEventSample from wire format.

        Forward-compatibility:
        - Unknown fields are ignored.
        - Missing optional fields fall back to defaults.
        """
        return StepTimeEventSample(
            seq=int(data["seq"]),
            timestamp=float(data["timestamp"]),
            step=int(data["step"]),
            events=dict(data.get("events", {})),
        )
