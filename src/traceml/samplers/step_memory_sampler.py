import time

from .base_sampler import BaseSampler
from traceml.utils.step_memory import step_memory_queue
from traceml.loggers.error_log import get_error_logger


class StepMemorySampler(BaseSampler):
    """
    Drain-all step-level peak-memory sampler.

    Each call to `sample()`:
      - Drains the step memory queue.
      - Stores one record per TraceML step.
    """

    def __init__(self) -> None:
        self.sampler_name = "StepMemorySampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)
        self.db.create_table("step_memory")

    def _drain_queue(self) -> None:
        """
        Drain entire step memory queue.
        """
        queue = step_memory_queue
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
        Save a single step-level memory event.
        """
        timestamp = time.time()

        model_id = getattr(event, "model_id", None)
        device = getattr(event, "device", None)
        step = getattr(event, "step", None)
        peak_alloc = getattr(event, "peak_allocated_mb", None)
        peak_resv = getattr(event, "peak_reserved_mb", None)

        if peak_alloc is None and peak_resv is None:
            return

        record = {
            "timestamp": timestamp,
            "model_id": model_id,
            "device": device,
            "step": step,
            "peak_allocated_mb": peak_alloc,
            "peak_reserved_mb": peak_resv,
        }

        self.db.add_record("step_memory", record)

    def sample(self):
        """
        Drain queue → save raw events → no aggregation.
        """
        try:
            self._drain_queue()
        except Exception as e:
            self.logger.error(f"[TraceML] StepMemorySampler error: {e}")
