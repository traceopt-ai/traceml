from typing import List, Deque
from queue import Empty, Full
from collections import deque

from .base_sampler import BaseSampler
from traceml.loggers.error_log import get_error_logger
from traceml.utils.activation_time_hooks import (
    ActivationTimeEvent,
    get_activation_time_queue
)

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

        # Local FIFO buffer owned by the sampler
        self._local_buffer: Deque[ActivationTimeEvent] = deque()

    def _ingest_queue(self) -> None:
        """
        Drain shared activation-time queue and append all events
        into the local FIFO buffer (order preserved).
        """
        q = get_activation_time_queue()

        while True:
            try:
                batch = q.get_nowait()  # batch is Deque[ActivationTimeEvent]
            except Empty:
                break

            # Extend local FIFO with the batch (order preserved)
            self._local_buffer.extend(batch)


    def _resolve_ready_events(self) -> List[ActivationTimeEvent]:
        """
        Resolve events from the head of the local buffer.

        Stops at first unresolved event to preserve FIFO semantics.
        """
        resolved: List[ActivationTimeEvent] = []
        while self._local_buffer:
            evt = self._local_buffer[0]
            if not evt.try_resolve():
                break
            resolved.append(evt)
            self._local_buffer.popleft()
        return resolved



    def _save_events(self, events: List[ActivationTimeEvent]) -> None:
        """
        Save resolved activation timing events into per-layer tables.
        """
        for evt in events:
            table_name = f"{evt.layer_name}"
            table = self.db.create_or_get_table(table_name)

            record = {
                "timestamp": evt.cpu_end,
                "model_id": evt.model_id,
                "layer_name": evt.layer_name,
                "on_gpu": evt.on_gpu,
                "cpu_duration_ms": evt.cpu_duration_ms,
                "gpu_duration_ms": evt.gpu_duration_ms,
            }
            table.append(record)

    def sample(self) -> None:
        """
        Ingest → resolve (FIFO) → persist
        """
        try:
            self._ingest_queue()
            ready_events = self._resolve_ready_events()
            self._save_events(ready_events)

        except Exception as e:
            self.logger.error(
                f"[TraceML] ActivationTimeSampler error: {e}"
            )