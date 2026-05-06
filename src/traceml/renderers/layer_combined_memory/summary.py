"""Best-effort summaries for deep layer-memory profiling."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from traceml.database.database import Database
from traceml.database.remote_database_store import RemoteDBStore
from traceml.loggers.error_log import get_error_logger
from traceml.renderers.layer_combined_memory.compute import (
    LayerCombinedMemoryData,
)
from traceml.samplers.schema.layer_forward_backward_memory import (
    LayerForwardBackwardMemorySample,
)
from traceml.samplers.schema.layer_memory import LayerMemorySample


class LayerCombinedMemorySummary:
    """
    Coarse historical stats for the optional deep layer-memory profiler.

    This is separate from the live renderer compute path. It scans available
    rank data without step-alignment guarantees, so callers should treat the
    result as a diagnostic hint rather than a final report contract.
    """

    def __init__(
        self,
        remote_store: RemoteDBStore,
        *,
        layer_memory_name: str = "LayerMemory",
        layer_forward_name: str = "LayerForwardMemory",
        layer_backward_name: str = "LayerBackwardMemory",
    ):
        self._remote_store = remote_store
        self.layer_memory_name = layer_memory_name
        self.layer_forward_name = layer_forward_name
        self.layer_backward_name = layer_backward_name
        self.logger = get_error_logger("LayerCombinedMemorySummary")

    def compute_layer_memory_summary(self) -> Dict[str, Any]:
        """Return average static parameter memory across reported ranks."""
        signatures: Set[str] = set()
        totals: List[float] = []

        for rank in self._remote_store.ranks():
            db = self._safe_get_db(rank, self.layer_memory_name + "Sampler")
            if not db:
                continue

            rows = db.get_table(self.layer_memory_name + "Table")
            if not rows:
                continue

            try:
                last = LayerMemorySample.from_wire(rows[-1])
            except Exception:
                continue

            if last.model_signature:
                signatures.add(last.model_signature)
            totals.append(float(last.total_param_bytes))

        return {
            "total_models_seen": len(signatures),
            "model_memory": (sum(totals) / len(totals)) if totals else 0.0,
        }

    def compute_global_peaks(self, is_forward: bool) -> Dict[str, float]:
        """Return per-layer activation peaks across all ranks and steps."""
        sampler = (
            self.layer_forward_name if is_forward else self.layer_backward_name
        )
        sampler_db_name = sampler + "Sampler"

        peaks: Dict[str, float] = {}

        for rank in self._remote_store.ranks():
            db = self._safe_get_db(rank, sampler_db_name)
            if not db:
                continue

            rows = db.get_table(sampler + "Table")
            if not rows:
                continue

            samples = LayerCombinedMemoryData._load_samples_backwards(
                rows, LayerForwardBackwardMemorySample, 0
            )
            for sample in samples:
                for layer, mem in zip(
                    sample.payload.layer_names,
                    sample.payload.layer_memory_bytes,
                ):
                    peaks[layer] = max(peaks.get(layer, 0.0), float(mem))

        return peaks

    @staticmethod
    def top_n_from_dict(d: Dict[str, float], n: int = 3):
        """Return the top-N dict items by descending value."""
        if not d:
            return []
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]

    def _safe_get_db(self, rank: int, sampler_name: str) -> Optional[Database]:
        try:
            return self._remote_store.get_db(rank, sampler_name)
        except Exception:
            self.logger.exception(
                "Failed to fetch rank DB from remote store",
                extra={"rank": rank, "sampler_name": sampler_name},
            )
            return None


__all__ = ["LayerCombinedMemorySummary"]
