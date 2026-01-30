from typing import Any

from traceml.loggers.error_log import get_error_logger
from traceml.utils.layer_forward_memory_hook import (
    get_layer_forward_memory_queue,
)

from .base_sampler import BaseSampler


class LayerForwardMemorySampler(BaseSampler):
    """
    Sampler for forward-pass activation memory at the layer level.

    This sampler drains the shared forward-memory event queue and
    stores each layer's activation memory as a flat record in a
    single database table.

    Design
    ------
    - One table: `layer_forward_memory`
    - One row per (model_id, layer_name, step)

    Expected event payload
    ----------------------
    LayerForwardMemoryEvents with fields:
      - model_id : int
      - layers   : List[(layer_name: str, memory_bytes: float)]
      - device   : str
      - step     : int
    """

    TABLE_NAME = "layer_forward_memory"

    def __init__(self) -> None:
        self.sampler_name = "LayerForwardMemorySampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)
        self.sample_idx = 0

    def _drain_queue(self) -> None:
        """
        Drain the forward-memory queue and persist all events.

        This method is intentionally non-blocking and tolerant to
        malformed or partial events.
        """
        queue = get_layer_forward_memory_queue()

        while not queue.empty():
            try:
                event = queue.get_nowait()
            except Exception:
                # Queue state changed unexpectedly
                break

            if event is None:
                continue
            self._save_event(event)

    def _save_event(self, event: Any) -> None:
        """
        Saves a single forward-memory event into the database.

        Parameters
        ----------
        event : LayerForwardMemoryEvents
            Event produced by forward hooks.
        """

        layers = getattr(event, "layers", None)
        if not layers:
            return

        record = {
            "seq": self.sample_idx,
            "model_id": getattr(event, "model_id", None),
            "step": getattr(event, "step", None),
            "device": getattr(event, "device", None),
            "layers": layers,
        }
        self.db.add_record(self.TABLE_NAME, record)

    def sample(self):
        """
        Ingest all available forward-memory events from the queue.

        Safe to call frequently; does nothing if the queue is empty.
        """
        self.sample_idx += 1
        try:
            self._drain_queue()
        except Exception as e:
            self.logger.error(
                f"[TraceML] LayerForwardMemorySampler error: {e}",
            )
