from typing import Dict, Any
from .base_sampler import BaseSampler
from traceml.utils.layer_forward_memory_hook import get_layer_forward_memory_queue
from traceml.loggers.error_log import get_error_logger


class LayerForwardMemorySampler(BaseSampler):
    """
    Drain-all layer forward-event sampler.

    Each call to `sample()`:
      - Drains the forward queue.
      - Save it internally in dict.
    """

    def __init__(self) -> None:
        self.sampler_name = "LayerForwardMemorySampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)

    def _drain_queue(self) -> None:
        """
        Drain entire forward queue and save every event.
        """
        queue = get_layer_forward_memory_queue()
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
        model_id = getattr(event, "model_id", None)
        layers = getattr(event, "layers", None)
        step = getattr(event, "step", None)

        if not layers:
            return
        for layer_name, memory_per_device in layers:
            record = {
                "model_id": model_id,
                "memory": memory_per_device,
                "step": step,
            }
            self.db.add_record(layer_name, record)

    def sample(self):
        """
        Drain queue → save raw events → no computation.
        """
        try:
            self._drain_queue()
        except Exception as e:
            self.logger.error(f"[TraceML] LayerForwardMemorySampler error: {e}")
