from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Optional, Set

from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.schema.layer_memory import (
    LayerMemoryPayload,
    LayerMemorySample,
)
from traceml.samplers.utils import drain_queue_nowait
from traceml.utils.layer_parameter_memory import get_model_queue


class LayerMemorySampler(BaseSampler):
    """
    Sampler for static, per-layer parameter memory of PyTorch models.
    """

    def __init__(self) -> None:
        super().__init__(
            sampler_name="LayerMemorySampler",
            table_name="LayerMemoryTable",
        )
        self.sample_idx = 0
        self.seen_signatures: Set[str] = set()

    def _compute_signature(
        self, layer_names: List[str], layer_bytes: List[float]
    ) -> str:
        """
        Compute a stable architecture signature from ordered layer memory.
        """
        items = [
            f"{name}:{int(b)}" for name, b in zip(layer_names, layer_bytes)
        ]
        raw = "|".join(items)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _normalize_layer_memory(
        self, layer_memory: Dict[str, float]
    ) -> LayerMemoryPayload:
        """
        Normalize raw layer-memory dict into a deterministic payload.
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
        """
        try:
            for payload in drain_queue_nowait(get_model_queue()):
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
        Ingest one layer-memory snapshot from the queue if available.
        """
        self.sample_idx += 1
        try:
            sample = self._sample_from_queue()
            if sample:
                self._add_record(sample.to_wire())

        except Exception as e:
            self.logger.error(f"[TraceML] LayerMemorySampler error: {e}")
