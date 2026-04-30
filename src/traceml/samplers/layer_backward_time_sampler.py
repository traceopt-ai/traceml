from __future__ import annotations

import time
from collections import deque
from typing import Deque

from traceml.instrumentation.hooks.layer_backward_time_hooks import (
    LayerBackwardTimeStepEvent,
    get_layer_backward_time_queue,
)
from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.layer_time_common import (
    aggregate_layer_time_payload,
    all_layer_events_resolved,
)
from traceml.samplers.schema.layer_forward_backward_time import (
    LayerForwardBackwardTimeSample,
)
from traceml.samplers.utils import append_queue_nowait_to_deque


class LayerBackwardTimeSampler(BaseSampler):
    """
    Sampler for backward-pass execution time at the layer level.
    """

    def __init__(self) -> None:
        super().__init__(
            sampler_name="LayerBackwardTimeSampler",
            table_name="LayerBackwardTimeTable",
            max_rows_per_flush=5,
        )
        self._local_buffer: Deque[LayerBackwardTimeStepEvent] = deque()
        self.sample_idx = 0

    def _ingest_queue(self) -> None:
        """
        Drain the shared backward-time queue into the local FIFO buffer.
        """
        append_queue_nowait_to_deque(
            get_layer_backward_time_queue(),
            self._local_buffer,
        )

    def _step_is_resolved(self, event: LayerBackwardTimeStepEvent) -> bool:
        return all_layer_events_resolved(event.layers)

    def sample(self) -> None:
        """
        Ingest -> resolve earliest step -> aggregate -> persist.
        """
        try:
            self._ingest_queue()

            while self._local_buffer:
                event = self._local_buffer[0]

                if not self._step_is_resolved(event):
                    break

                self._local_buffer.popleft()
                self.sample_idx += 1

                payload = aggregate_layer_time_payload(event.layers)

                sample = LayerForwardBackwardTimeSample(
                    sample_idx=self.sample_idx,
                    timestamp=time.time(),
                    model_id=event.model_id,
                    step=event.step,
                    device=event.device,
                    payload=payload,
                )

                self._add_record(sample.to_wire())

        except Exception as e:
            self.logger.error(f"[TraceML] LayerBackwardTimeSampler error: {e}")
