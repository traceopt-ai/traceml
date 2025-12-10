from typing import Dict, Any
import time
from .base_sampler import BaseSampler
from traceml.utils.activation_memory_hook import get_activation_memory_queue
from traceml.loggers.error_log import get_error_logger


class ActivationMemorySampler(BaseSampler):
    """
    Drain-all activation-event sampler.

    Each call to `sample()`:
      - Drains the activation queue.
      - Save it internally in dict.
    """

    def __init__(self) -> None:
        self.sampler_name = "ActivationMemorySampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)

    def _drain_queue(self) -> None:
        """
        Drain entire activation queue and save every event.
        """
        queue = get_activation_memory_queue()
        if queue.empty():
            return

        while not queue.empty():
            try:
                event = queue.get_nowait()
            except Exception:
                break

            if event is None:
                continue
            self._save_event(event)

    def _save_event(self, event: Dict[str, Any]) -> None:
        """
        Save one ActivationEvent into per-layer tables.
        """
        layer_name = getattr(event, "layer_name", None)
        record = {
            "timestamp": time.time(),
            "model_id": getattr(event, "model_id", None),
            "memory": getattr(event, "memory_per_device", None),
        }

        table = self.db.create_or_get_table(layer_name)
        table.append(record)

    def sample(self):
        """
        Drain queue → save raw events → no computation.
        """
        try:
            self._drain_queue()
        except Exception as e:
            self.logger.error(f"[TraceML] ActivationMemorySampler error: {e}")
