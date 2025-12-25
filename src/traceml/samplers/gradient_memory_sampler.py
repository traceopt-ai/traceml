from typing import Any, Dict
import time
from .base_sampler import BaseSampler
from traceml.utils.gradient_memory_hook import get_gradient_queue
from traceml.loggers.error_log import get_error_logger


class GradientMemorySampler(BaseSampler):
    """
    Drain-all gradient-event sampler.

    Each call to `sample()`:
      - Drains the gradient queue.
      - Save it internally in dict.
    """

    def __init__(self) -> None:
        self.sampler_name = "GradientMemorySampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)

    def _save_event(self, event: Dict[str, Any]) -> None:
        timestamp = time.time()
        model_id = getattr(event, "model_id", None)

        layers = getattr(event, "layers", None)
        if not layers:
            return

        for layer_name, memory_per_device in layers:
            record = {
                "timestamp": timestamp,
                "model_id": model_id,
                "memory": memory_per_device,
            }

            table = self.db.create_or_get_table(layer_name)
            table.append(record)

    def _drain_queue(self) -> None:
        """
        Drain entire activation queue and save every event.
        """
        queue = get_gradient_queue()
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

    def sample(self):
        """
        Drain queue → save raw events → no computation.
        """
        try:
            self._drain_queue()
        except Exception as e:
            self.logger.error(f"[TraceML] GradientMemorySampler error: {e}")
