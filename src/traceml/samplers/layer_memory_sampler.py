import time
import hashlib
from typing import Dict, Optional, Set, List

from traceml.samplers.base_sampler import BaseSampler
from traceml.loggers.error_log import get_error_logger
from traceml.utils.layer_parameter_memory import get_model_queue


from traceml.samplers.schema.layer_memory import (
    LayerMemoryPayload,
    LayerMemorySample,
)


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


    def __init__(self) -> None:
        self.name = "LayerMemory"
        self.sampler_name = self.name+"Sampler"
        self.table_name = self.name+"Table"
        super().__init__(sampler_name=self.sampler_name)
        self.sample_idx = 0

        self.logger = get_error_logger(self.sampler_name)

        # Deduplication store for seen models
        self.seen_signatures: Set[str] = set()

    def _compute_signature(self, layer_names: List[str], layer_bytes: List[float]) -> str:
        """
        Compute a stable architecture signature from ordered layer memory.

        The signature is derived from ordered (layer_name, bytes) pairs,
        making it robust to object identity, process lifetime, and dict order.
        """
        items = [f"{name}:{int(b)}" for name, b in zip(layer_names, layer_bytes)]
        raw = "|".join(items)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _normalize_layer_memory(
        self, layer_memory: Dict[str, float]
    ) -> LayerMemoryPayload:
        """
        Normalize raw layer-memory dict into a deterministic payload.

        Sorting happens *once* here to guarantee:
        - stable hashing
        - stable wire representation
        """
        names = sorted(layer_memory.keys())
        bytes_ = [float(layer_memory[name]) for name in names]
        return LayerMemoryPayload(layer_names=names, layer_param_bytes=bytes_)

    def _build_sample(
            self,
            layer_memory: Dict[str, float],
    ) -> Optional[LayerMemorySample]:
        """
        Build a LayerMemorySample from raw layer-memory input.

        Returns None if the architecture has already been seen.
        """
        payload = self._normalize_layer_memory(layer_memory)
        signature = self._compute_signature(
            payload.layer_names, payload.layer_param_bytes
        )

        if signature in self.seen_signatures:
            return None

        self.seen_signatures.add(signature)

        return LayerMemorySample(
            sample_idx=self.sample_idx,
            timestamp=time.time(),
            model_index=len(self.seen_signatures) - 1,
            model_signature=signature,
            total_param_bytes=float(sum(payload.layer_param_bytes)),
            layer_count=len(payload.layer_names),
            payload=payload,
        )

    def _sample_from_queue(self) -> Optional[LayerMemorySample]:
        """
        Consume the model queue and return the first unseen architecture.

        The queue is expected to contain dict payloads mapping
        layer_name -> parameter_bytes.
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
                if sample is not None:
                    return sample

        except Exception as e:
            self.logger.error(
                f"[TraceML] Layer memory queue ingestion failed: {e}"
            )
        return None

    def sample(self) -> None:
        """
        Ingest one layer-memory snapshot from the queue (if available).

        This method is safe to call frequently; actual writes occur
        only when a new, unseen model architecture is encountered.
        """
        self.sample_idx += 1
        try:
            sample = self._sample_from_queue()
            if sample:
                self.db.add_record(self.table_name, sample.to_wire())

        except Exception as e:
            # Absolute safety net — sampling must never break training
            self.logger.error(f"[TraceML] LayerMemorySampler error: {e}")
