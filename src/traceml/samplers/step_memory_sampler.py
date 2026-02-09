import time
from typing import Any, Optional

from traceml.loggers.error_log import get_error_logger
from traceml.utils.step_memory import step_memory_queue

from .base_sampler import BaseSampler
from traceml.samplers.schema.step_memory import StepMemorySample


class StepMemorySampler(BaseSampler):
    """
    Drain-all step-level peak-memory sampler.

    Each call to `sample()`:
      - Drains the step memory queue.
      - Stores one record per TraceML step (per event).

    Guarantees
    ----------
    - Never raises; failures are logged and swallowed
    - Drain-all behavior: does not aggregate or drop records intentionally
    - Records stored are wire-format dicts (schema-owned)
    """

    def __init__(self) -> None:
        self.name = "StepMemory"
        self.sampler_name = self.name + "Sampler"
        self.table_name = "step_memory"
        self.sample_idx = 0

        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)

    def _drain_queue(self) -> None:
        """
        Drain entire step memory queue.
        """
        q = step_memory_queue
        while not q.empty():
            try:
                event = q.get_nowait()
            except Exception:
                break

            if event is None:
                continue

            sample = self._event_to_sample(event)
            if sample is None:
                continue

            # Store wire representation to keep DB/transport independent
            # of Python objects.
            self.db.add_record(self.table_name, sample.to_wire())

    def _event_to_sample(self, event: Any) -> Optional[StepMemorySample]:
        """
        Convert a raw queue event to a StepMemorySample.

        Returns None if the event doesn't contain any usable memory signals.
        """
        ts = time.time()

        model_id = getattr(event, "model_id", None)
        device = getattr(event, "device", None)
        step = getattr(event, "step", None)

        peak_alloc = getattr(event, "peak_allocated_mb", None)
        peak_resv = getattr(event, "peak_reserved_mb", None)

        # Drop records that contain no memory signal
        if peak_alloc is None and peak_resv is None:
            return None

        return StepMemorySample(
            sample_idx=self.sample_idx,
            timestamp=ts,
            model_id=model_id,
            device=device,
            step=step,
            peak_allocated=peak_alloc,
            peak_reserved=peak_resv,
        )

    def sample(self) -> None:
        """
        Drain queue → save raw events → no aggregation.

        Safe to call frequently; never interferes with training.
        """
        self.sample_idx += 1
        try:
            self._drain_queue()
        except Exception as e:
            self.logger.error(f"[TraceML] StepMemorySampler error: {e}")
