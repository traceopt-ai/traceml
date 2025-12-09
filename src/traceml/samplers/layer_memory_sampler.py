import torch
import time
from typing import Dict, Any, Optional, Set
import hashlib
from .base_sampler import BaseSampler
from traceml.utils.patch import get_model_queue
from traceml.loggers.error_log import get_error_logger


class LayerMemorySampler(BaseSampler):
    """
    Sampler that tracks parameter memory usage of PyTorch models at a per-layer level.
    """

    def __init__(self) -> None:
        self.sampler_name = "LayerMemorySampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)
        self._table = self.db.create_or_get_table("layer_memory")
        # Deduplication store for seen models
        self.seen_signatures: Set[str] = set()

    def _get_model_signature(self, model: torch.nn.Module) -> str:
        """
        Generate a unique signature for the model.
        """
        items = [model.__class__.__name__]
        # record each module class name in order
        for name, module in model.named_modules():
            items.append(module.__class__.__name__)
        raw = "|".join(items)
        return hashlib.md5(raw.encode()).hexdigest()

    def _compute_layer_memory(self, model: torch.nn.Module) -> Dict[str, float]:
        """Compute per-layer parameter memory."""
        layer_mem = {}
        for name, module in model.named_modules():
            if any(module.children()):  # skip containers
                continue

            total = 0.0
            for p in module.parameters(recurse=False):
                total += p.element_size() * p.nelement()

            if total > 0:
                layer_mem[name] = total
        return layer_mem

    def _sample_model(self, model: torch.nn.Module) -> Optional[Dict[str, Any]]:
        """Process one model if new, otherwise ignore."""
        try:
            sig = self._get_model_signature(model)
            if sig in self.seen_signatures:
                return None

            self.seen_signatures.add(sig)
            layer_mem = self._compute_layer_memory(model)

            sample = {
                "timestamp": time.time(),
                "model_index": len(self.seen_signatures) - 1,
                "total_memory": float(sum(layer_mem.values())),
                "layer_memory": layer_mem,
                "model_signature": str(sig),
            }
            return sample

        except Exception as e:
            self.logger.error(f"[TraceML] Error sampling model: {e}")
            return {
                "timestamp": time.time(),
                "error": str(e),
                "model_index": -1,
                "total_memory": 0.0,
                "layer_memory": {},
                "model_signature": None,
            }

    def _sample_from_queue(self) -> Optional[Dict[str, Any]]:
        """Iterate the traced model queue and sample the first unseen model."""
        try:
            queue = get_model_queue()
            if queue.empty():
                return None

            while not queue.empty():
                model = queue.get_nowait()
                snap = self._sample_model(model)
                if snap is not None:
                    return snap
        except Exception as e:
            self.logger.error(f"[TraceML] Queue sampling failed: {e}")
        return None

    def sample(self):
        """
        Sample memory usage from models in the queue.
        """
        try:
            sample = self._sample_from_queue()
            if sample is not None:
                self._table.append(sample)
        except Exception as e:
            self.logger.error(f"[TraceML] Layer memory sampling error: {e}")
