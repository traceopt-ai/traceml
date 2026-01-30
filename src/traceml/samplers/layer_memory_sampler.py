import hashlib
import time
from typing import Any, Dict, Optional, Set

from traceml.loggers.error_log import get_error_logger
from traceml.samplers.base_sampler import BaseSampler
from traceml.utils.layer_parameter_memory import get_model_queue


class LayerMemorySampler(BaseSampler):
    """
    Sampler for static, per-layer *parameter* memory of PyTorch models.

    This sampler ingests precomputed layer-memory snapshots produced
    by the training code (not live model objects). Each unique model
    architecture is recorded at most once.

    Scope
    -----
    - One-time / low-frequency sampling
    - Parameter memory only (no activations, no gradients)
    - Architecture-level, not step-level

    Design principles
    -----------------
    - Sampler never touches live `nn.Module` objects
    - Deterministic, race-free ingestion
    - Deduplication based on stable content signature
    - Safe to run asynchronously
    """

    TABLE_NAME = "layer_memory"

    def __init__(self) -> None:
        self.sampler_name = "LayerMemorySampler"
        super().__init__(sampler_name=self.sampler_name)
        self.sample_idx = 0

        self.logger = get_error_logger(self.sampler_name)

        # Deduplication store for seen models
        self.seen_signatures: Set[str] = set()

    def _compute_signature(self, layer_memory: Dict[str, float]) -> str:
        """
        Compute a stable signature for a model based on its layer memory.

        The signature is derived from the ordered (layer_name, bytes)
        pairs, making it robust to object identity and process lifetime.

        Parameters
        ----------
        layer_memory : Dict[str, float]
            Mapping from layer name to parameter memory (bytes).

        Returns
        -------
        str
            Stable hash identifying the model architecture.
        """
        items = [f"{k}:{int(v)}" for k, v in sorted(layer_memory.items())]
        raw = "|".join(items)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _build_sample(
        self,
        layer_memory: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Build a normalized sample record for storage.

        Parameters
        ----------
        layer_memory : Dict[str, float]
            Per-layer parameter memory (bytes).

        Returns
        -------
        Dict[str, Any]
            Record ready to be stored in the database.
        """
        signature = self._compute_signature(layer_memory)

        if signature in self.seen_signatures:
            return {}

        self.seen_signatures.add(signature)

        return {
            "timestamp": time.time(),
            "model_index": len(self.seen_signatures) - 1,
            "model_signature": signature,
            "total_memory": float(sum(layer_memory.values())),
            "layer_memory": layer_memory,
        }

    def _sample_from_queue(self) -> Optional[Dict[str, Any]]:
        """
        Consume the model queue and return the first unseen sample.

        The queue is expected to contain **dict payloads** produced by
        training code (e.g. via `collect_layer_parameter_memory`).

        Returns
        -------
        Optional[Dict[str, Any]]
            A new sample if available, otherwise None.
        """
        try:
            queue = get_model_queue()
            if queue.empty():
                return None

            while not queue.empty():
                payload = queue.get_nowait()
                if not payload:
                    continue

                sample = self._build_sample(payload)
                if sample:
                    return sample

        except Exception as e:
            self.logger.error(
                f"[TraceML] Layer memory queue ingestion failed: {e}",
            )
        return None

    def sample(self) -> None:
        """
        Ingest one layer-memory snapshot from the queue (if available).

        This method is safe to call frequently; actual writes occur
        only when a new, unseen model snapshot is encountered.
        """
        self.sample_idx += 1
        try:
            sample = self._sample_from_queue()
            if sample:
                sample["seq"] = self.sample_idx
                self.db.add_record(self.TABLE_NAME, sample)

        except Exception as e:
            # Absolute safety net — sampling must never break training
            self.logger.error(f"[TraceML] LayerMemorySampler error: {e}")
