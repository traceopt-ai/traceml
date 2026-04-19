from __future__ import annotations

import time
from typing import Any

from traceml.hooks.layer_backward_memory_hooks import get_layer_backward_queue
from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.layer_memory_common import (
    aggregate_layer_memory_payload_max,
)
from traceml.samplers.schema.layer_forward_backward_memory import (
    LayerForwardBackwardMemorySample,
)
from traceml.samplers.utils import drain_queue_nowait


class LayerBackwardMemorySampler(BaseSampler):
    """
    Sampler for backward-pass activation memory at the layer level.
    """

    def __init__(self) -> None:
        super().__init__(
            sampler_name="LayerBackwardMemorySampler",
            table_name="LayerBackwardMemoryTable",
            max_rows_per_flush=5,
        )
        self.sample_idx = 0

    def _drain_queue(self) -> None:
        """
        Drain the backward-memory queue and persist all available events.
        """
        for event in drain_queue_nowait(get_layer_backward_queue()):
            self._save_event(event)

    def _save_event(self, event: Any) -> None:
        """
        Persist a single backward-memory event to the database.
        """
        layers = getattr(event, "layers", None)
        if not layers:
            return

        try:
            payload = aggregate_layer_memory_payload_max(layers)

            sample = LayerForwardBackwardMemorySample(
                sample_idx=self.sample_idx,
                timestamp=time.time(),
                model_id=getattr(event, "model_id", None),
                step=getattr(event, "step", None),
                device=getattr(event, "device", None),
                payload=payload,
            )

            self._add_record(sample.to_wire())

        except Exception as e:
            self.logger.error(
                f"[TraceML] Failed to persist backward layer memory event: {e}"
            )

    def sample(self) -> None:
        """
        Ingest all available backward-memory events from the queue.
        """
        self.sample_idx += 1
        try:
            self._drain_queue()
        except Exception as e:
            self.logger.error(
                f"[TraceML] LayerBackwardMemorySampler error: {e}"
            )
