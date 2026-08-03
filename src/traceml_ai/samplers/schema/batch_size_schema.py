"""
Batch Size (Bytes) Schema (Shared Contract)

Defines the per-step batch-size sample produced by BatchSizeSampler and
persisted by the aggregator into SQLite.

Concept
-------
BatchSizeSampler emits **one record per step, per rank** containing the
total tensor bytes fetched from the dataloader during that step.
Gradient accumulation is handled naturally because all fetches inside
the same step are summed before emission. Sizing at the dataloader keeps
the metric device-agnostic: CPU-only training records the same batch
bytes as GPU training.

Units
-----
- bytes_total : int  (sum of element_size * numel across the batches
                      fetched in the step)
- n_fetches   : int  (number of dataloader fetches summed; diagnostic)

Schema (per DB row)
-------------------
{
    "seq": int,            # monotonically increasing sequence per rank
    "timestamp": float,    # step timestamp (seconds, CPU wall clock)
    "step": int,           # training step id
    "bytes_total": int,    # summed batch bytes for the step
    "n_fetches": int       # how many fetches were summed
}
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class BatchSizeSample:
    """Canonical per-step batch-size-in-bytes sample."""

    seq: int
    timestamp: float
    step: int
    bytes_total: int
    n_fetches: int

    def to_wire(self) -> Dict[str, Any]:
        """Convert to a wire-friendly dictionary (msgpack/JSON safe)."""
        return {
            "seq": int(self.seq),
            "timestamp": float(self.timestamp),
            "step": int(self.step),
            "bytes_total": int(self.bytes_total),
            "n_fetches": int(self.n_fetches),
        }

    @staticmethod
    def from_wire(data: Mapping[str, Any]) -> "BatchSizeSample":
        """Reconstruct from wire format. Unknown fields are ignored."""
        return BatchSizeSample(
            seq=int(data["seq"]),
            timestamp=float(data["timestamp"]),
            step=int(data["step"]),
            bytes_total=int(data.get("bytes_total", 0)),
            n_fetches=int(data.get("n_fetches", 0)),
        )
