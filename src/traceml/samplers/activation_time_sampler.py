from typing import Dict, Any
from .base_sampler import BaseSampler
from traceml.utils.activation_time_hooks import get_activation_time_queue
from traceml.loggers.error_log import get_error_logger


class ActivationTimeSampler(BaseSampler):
    """
    Drain-all activation-time event sampler.

    Each call to `sample()`:
      - Drains the activation time queue.
      - Stores each event in a per-layer table inside the local DB.
    """

    def __init__(self) -> None:
        self.sampler_name = "ActivationTimeSampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)

    def _drain_queue(self) -> None:
        """
        Drain entire activation-time queue and save every event.
        """
        queue = get_activation_time_queue()
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

    def _save_event(self, event: Any) -> None:
        """
        Save one ActivationTimeEvent into per-layer tables.
        Expected event fields:
          - model_id
          - layer_name
          - start_time
          - end_time
          - duration_ms
        """
        layer_name = getattr(event, "layer_name", None)
        record = {
            "model_id": getattr(event, "model_id", None),
            "start_time": getattr(event, "start_time", None),
            "end_time": getattr(event, "end_time", None),
            "duration_ms": getattr(event, "duration_ms", None),
        }

        table = self.db.create_or_get_table(layer_name)
        table.append(record)

    def sample(self):
        """
        Drain queue → save events → no computation.
        """
        try:
            self._drain_queue()
        except Exception as e:
            self.logger.error(f"[TraceML] ActivationTimeSampler error: {e}")
