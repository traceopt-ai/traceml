from __future__ import annotations

import time
from typing import Any, Optional

from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.schema.step_memory import StepMemorySample
from traceml.samplers.utils import drain_queue_nowait
from traceml.utils.step_memory import step_memory_queue


class StepMemorySampler(BaseSampler):
    """
    Drain-all step-level peak-memory sampler.
    """

    def __init__(self) -> None:
        super().__init__(
            sampler_name="StepMemorySampler",
            table_name="step_memory",
        )
        self.sample_idx = 0

    def _drain_queue(self) -> None:
        """
        Drain entire step memory queue.
        """
        for event in drain_queue_nowait(step_memory_queue):
            sample = self._event_to_sample(event)
            if sample is None:
                continue
            self._add_record(sample.to_wire())

    def _event_to_sample(self, event: Any) -> Optional[StepMemorySample]:
        """
        Convert a raw queue event to a StepMemorySample.
        """
        ts = time.time()

        model_id = getattr(event, "model_id", None)
        device = getattr(event, "device", None)
        step = getattr(event, "step", None)

        peak_alloc = getattr(event, "peak_allocated", None)
        peak_resv = getattr(event, "peak_reserved", None)

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
        Drain queue -> save raw events -> no aggregation.
        """
        self.sample_idx += 1
        try:
            self._drain_queue()
        except Exception as e:
            self.logger.error(f"[TraceML] StepMemorySampler error: {e}")
