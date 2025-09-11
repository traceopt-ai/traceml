from dataclasses import dataclass, field
from collections import deque
import torch
from typing import Dict, Any, Tuple, Optional, Deque, Set

from .base_sampler import BaseSampler
from traceml.utils.patch import get_model_queue
from traceml.loggers.error_log import get_error_logger, setup_error_logger


@dataclass
class ModelMemorySnapshot:
    model_index: int = -1
    total_memory: float = 0.0
    layer_memory: Dict[str, float] = field(default_factory=dict)
    model_signature: Optional[str] = None
    error: Optional[str] = None


class LayerMemorySampler(BaseSampler):
    """
    Sampler that tracks parameter memory usage of PyTorch models at a per-layer level.
    """

    def __init__(self):
        super().__init__()
        setup_error_logger()
        self.logger = get_error_logger("LayerMemorySampler")

        # Deduplication store for seen models
        self.seen_signatures: Set[Tuple] = set()

        # Capped sampling history
        self.memory_history: Deque[ModelMemorySnapshot] = deque(maxlen=10_000)

        # Latest snapshot
        self._latest_snapshot: ModelMemorySnapshot = None

        # Counters
        self.total_samples: int = 0

    def _get_model_signature(self, model: torch.nn.Module) -> Tuple:
        """
        Generate a unique signature for the model.
        """
        return tuple((name, tuple(p.shape)) for name, p in model.named_parameters())

    def _build_snapshot_from_model(
        self, model: torch.nn.Module, signature: Tuple
    ) -> ModelMemorySnapshot:
        """
        Compute per-layer parameter memory and aggregate total.
        """
        layer_memory: Dict[str, float] = {}
        total_memory = 0.0

        for name, param in model.named_parameters():
            memory = (param.element_size() * param.nelement()) / (1024**2)
            memory = round(float(memory), 4)
            layer_memory[name] = memory
            total_memory += memory

        snapshot = ModelMemorySnapshot(
            model_index=len(self.seen_signatures),
            total_memory=round(float(total_memory), 4),
            layer_memory=layer_memory,
            model_signature=str(signature),
            error=None,
        )
        return snapshot

    def _get_model_memory(self, model: torch.nn.Module) -> ModelMemorySnapshot:
        """Compute memory usage of a single model if it hasn't been sampled before."""
        try:
            signature = self._get_model_signature(model)
            if signature in self.seen_signatures:
                return None

            self.seen_signatures.add(signature)
            snapshot = self._build_snapshot_from_model(model, signature)
            self._latest_snapshot = snapshot
            self.memory_history.append(snapshot)
            self.total_samples += 1
            return snapshot
        except Exception as e:
            self.logger.error(f"[TraceML] Error sampling model: {e}")
            return None

    def _sample_from_queue(self) -> Optional[ModelMemorySnapshot]:
        """Iterate the traced model queue and sample the first unseen model."""
        try:
            queue = get_model_queue()
            if queue.empty():
                return None

            while not queue.empty():
                model = queue.get_nowait()
                snap = self._get_model_memory(model)
                if snap is not None:
                    return snap
        except Exception as e:
            self.logger.error(f"[TraceML] Queue sampling failed: {e}")
        return None

    def sample(self) -> Dict[str, Any]:
        """
        Sample memory usage from models in the queue.
        Returns the latest new snapshot or the last seen one.
        """
        snap = self._sample_from_queue()
        if isinstance(snap, ModelMemorySnapshot):
            self._latest_snapshot = snap

        if self._latest_snapshot is not None:
            ok = self._latest_snapshot.error is None
            message = (
                "sampled successfully"
                if ok
                else f"sampling completed with error: {self._latest_snapshot.error}"
            )
            envelope = self.make_snapshot(
                ok=ok,
                message=message,
                source="layer_memory",
                data=self._latest_snapshot.__dict__,
            )
        else:
            envelope = self.make_snapshot(
                ok=False,
                message="no model found",
                source="process",
                data=None,
            )
        return self.snapshot_dict(envelope)

    def get_summary(self) -> Dict[str, Any]:
        """
        Return summary statistics over all models seen.
        """
        print()
        total_models = len(self.seen_signatures)
        avg_total_memory = 0.0
        max_total_memory = 0.0
        print(total_models)

        if self.total_samples:
            totals = [s.total_memory for s in self.memory_history if s.error is None]
            avg_total_memory = (
                round(float(sum(totals) / len(totals)), 4) if totals else 0.0
            )
            max_total_memory = round(max(totals), 4) if totals else 0.0

        return {
            "total_models_seen": total_models,
            "total_samples_taken": self.total_samples,
            "average_model_memory": avg_total_memory,
            "peak_model_memory": max_total_memory,
            "last_model_snapshot": self._latest_snapshot,
        }
