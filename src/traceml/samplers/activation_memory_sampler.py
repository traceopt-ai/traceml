from typing import Dict, Any
from .base_sampler import BaseSampler
from traceml.utils.activation_hook import get_activation_queue
from traceml.loggers.error_log import get_error_logger, setup_error_logger
from traceml.database.database import Database


class ActivationMemorySampler(BaseSampler):
    """
    Drain-all activation-event sampler.

    Each call to `sample()`:
      - Drains the activation queue.
      - Save it internally in dict.
    """

    def __init__(
        self,
    ):
        super().__init__()
        setup_error_logger()
        self.logger = get_error_logger("ActivationMemorySampler")
        self.db = Database()

    def _drain_queue(self) -> None:
        """
        Drain entire activation queue and save every event.
        """
        queue = get_activation_queue()
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
            "timestamp": getattr(event, "timestamp", None),
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
