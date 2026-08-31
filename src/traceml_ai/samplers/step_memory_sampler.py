from __future__ import annotations

from traceml_ai.samplers.base_sampler import BaseSampler
from traceml_ai.samplers.schema.step_memory import StepMemorySample
from traceml_ai.samplers.utils import drain_queue_nowait
from traceml_ai.utils.step_memory import StepMemoryEvent, step_memory_queue


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
            self._add_record(sample.to_wire())

    def _event_to_sample(self, event: StepMemoryEvent) -> StepMemorySample:
        """
        Convert a raw queue event to a StepMemorySample.
        """
        return StepMemorySample(
            sample_idx=self.sample_idx,
            timestamp=float(event.timestamp),
            model_id=event.model_id,
            device=event.device,
            step=event.step,
            peak_allocated=event.peak_allocated,
            peak_reserved=event.peak_reserved,
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
