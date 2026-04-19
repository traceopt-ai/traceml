from __future__ import annotations

import time

from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.utils import drain_queue_nowait
from traceml.utils.hooks.model_forward_memory_hook import (
    get_model_forward_memory_queue,
)


class ModelForwardMemorySampler(BaseSampler):
    """
    Drain-all model forward peak-memory sampler.

    Each call to `sample()`:
      - Drains the model-forward memory queue.
      - Stores one record per forward pass.
    """

    def __init__(self) -> None:
        super().__init__(
            sampler_name="ModelForwardMemorySampler",
            table_name="model_forward_memory",
        )

    def _drain_queue(self) -> None:
        """
        Drain entire model forward memory queue.
        """
        for event in drain_queue_nowait(get_model_forward_memory_queue()):
            self._save_event(event)

    def _save_event(self, event) -> None:
        """
        Save a single model-level forward memory event.
        """
        timestamp = time.time()

        model_id = getattr(event, "model_id", None)
        device = getattr(event, "device", None)

        peak_alloc = getattr(event, "peak_allocated_mb", None)
        peak_resv = getattr(event, "peak_reserved_mb", None)

        if peak_alloc is None and peak_resv is None:
            return

        record = {
            "timestamp": timestamp,
            "model_id": model_id,
            "device": device,
            "peak_allocated_mb": peak_alloc,
            "peak_reserved_mb": peak_resv,
        }
        self._add_record(record)

    def sample(self) -> None:
        """
        Drain queue -> save raw events -> no aggregation.
        """
        try:
            self._drain_queue()
        except Exception as e:
            self.logger.error(
                f"[TraceML] ModelForwardMemorySampler error: {e}"
            )
