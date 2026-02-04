from typing import Any
from .base_sampler import BaseSampler
from traceml.utils.layer_backward_memory_hook import get_layer_backward_queue
from traceml.loggers.error_log import get_error_logger


class LayerBackwardMemorySampler(BaseSampler):
    """
    Sampler for backward-pass (gradient) activation memory at the layer level.

    This sampler drains the shared backward-memory event queue and
    stores each backward memory event as a single record in the database.

    Design
    ------
    - One table: `layer_backward_memory`
    - One row per event (per step flush)

    Expected event payload
    ----------------------
    LayerBackwardMemoryEvents with fields:
      - model_id : int
      - layers   : List[(layer_name: str, gradient_memory_bytes: float)]
      - device   : str
      - step     : int
    """


    def __init__(self) -> None:
        self.name = "LayerBackwardMemory"
        self.sampler_name = self.name+"Sampler"
        self.table_name = self.name+"Table"
        super().__init__(sampler_name=self.sampler_name)

        # Sample transport: send only the most recent row per flush (drops backlog)
        self.sender.max_rows_per_flush = 1

        self.logger = get_error_logger(self.sampler_name)
        self.sample_idx = 0

    def _drain_queue(self) -> None:
        """
        Drain the backward-memory queue and persist all available events.

        This method is intentionally non-blocking and tolerant to
        malformed or partial events.
        """
        queue = get_layer_backward_queue()
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
        Save a single backward-memory event into the database.

        Parameters
        ----------
        event : LayerBackwardMemoryEvents
            Event produced by backward hooks.
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
        self.db.add_record(self.table_name, record)

    def sample(self) -> None:
        """
        Ingest all available backward-memory events from the queue.

        Safe to call frequently; does nothing if the queue is empty.
        """
        self.sample_idx += 1
        try:
            self._drain_queue()
        except Exception as e:
            # Sampler must never disrupt training
            self.logger.error(f"[TraceML] LayerBackwardMemorySampler error: {e}")
