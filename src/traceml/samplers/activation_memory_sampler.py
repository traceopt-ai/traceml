from typing import Dict


from .base_sampler import BaseSampler
from traceml.utils.activation_hook import get_activation_queue
from traceml.loggers.error_log import get_error_logger, setup_error_logger
from traceml.database.global_database import GlobalDatabase


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
        self.db = GlobalDatabase()

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

    def _save_event(self, event) -> None:
        """
        Save one ActivationEvent into per-layer tables.
        """
        per_layer_dict: Dict[str, Dict[str, float]] = event.per_layer

        for layer_name, mem_dict in per_layer_dict.items():
            table_name = f"{layer_name}"
            # create or reuse existing table
            table = self.db.create_or_get_table(table_name)
            # append raw row
            table.append(
                {
                    "timestamp": event.timestamp,
                    "model_id": event.model_id,
                    "memory": mem_dict,  # raw per-device memory dict
                }
            )

    def sample(self):
        """
        Drain queue → save raw events → no computation.
        """
        try:
            self._drain_queue()
        except Exception as e:
            self.logger.error(f"[TraceML] ActivationMemorySampler error: {e}")
