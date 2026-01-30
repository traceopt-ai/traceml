from collections import deque
from queue import Empty
from typing import Deque, List

from traceml.loggers.error_log import get_error_logger
from traceml.utils.layer_backward_time_hooks import (
    LayerBackwardTimeEvent,
    get_layer_backward_time_queue,
)

from .base_sampler import BaseSampler


class LayerBackwardTimeSampler(BaseSampler):
    """
    Drain-all gradient-time event sampler.

    Each call to `sample()`:
      - Drains the gradient time queue.
      - Resolves GPU events non-blocking via try_resolve()
      - Stores each event in a per-layer table inside the local DB.
    """

    def __init__(self) -> None:
        self.sampler_name = "LayerBackwardTimeSampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)
        # Local FIFO buffer owned by the sampler
        self._local_buffer: Deque[LayerBackwardTimeEvent] = deque()

    def _ingest_queue(self) -> None:
        """
        Drain shared gradient-time queue and append all batches
        into the local FIFO buffer (order preserved).
        """
        q = get_layer_backward_time_queue()
        while True:
            try:
                batch = q.get_nowait()
            except Empty:
                break
            # Preserve order across batches
            self._local_buffer.extend(batch)

    def _resolve_ready_events(self) -> List[LayerBackwardTimeEvent]:
        """
        Resolve events from the head of the local buffer.

        FIFO invariant:
          If the first event is not resolved, no later event can be resolved.
        """
        resolved: List = []

        while self._local_buffer:
            evt = self._local_buffer[0]  # peek head
            if not evt.try_resolve():
                break
            resolved.append(evt)
            self._local_buffer.popleft()

        return resolved

    def _save_events(self, events: List) -> None:
        """
        Save resolved gradient timing events into per-layer tables.
        """
        for evt in events:
            table_name = f"{evt.layer_name}"
            record = {
                "timestamp": evt.cpu_end,
                "model_id": evt.model_id,
                "layer_name": evt.layer_name,
                "on_gpu": evt.on_gpu,
                "cpu_duration_ms": evt.cpu_duration_ms,
                "gpu_duration_ms": evt.gpu_duration_ms,
                "step": evt.step,
            }
            self.db.add_record(table_name, record)

    def sample(self) -> None:
        """
        Ingest → resolve (FIFO) → persist
        """
        try:
            self._ingest_queue()
            ready_events = self._resolve_ready_events()
            self._save_events(ready_events)
        except Exception as e:
            self.logger.error(f"[TraceML] LayerBackwardTimeSampler error: {e}")
