from typing import List
from queue import Empty, Full

from .base_sampler import BaseSampler
from traceml.loggers.error_log import get_error_logger
from traceml.utils.gradient_time_hooks import (
    GradientTimeEvent,
    get_gradient_time_queue,
)


class GradientTimeSampler(BaseSampler):
    """
    Drain-all gradient-time event sampler.

    Each call to `sample()`:
      - Drains the gradient time queue.
      - Resolves GPU events non-blocking via try_resolve()
      - Stores each event in a per-layer table inside the local DB.
    """

    def __init__(self) -> None:
        self.sampler_name = "GradientTimeSampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)

    def _drain_queue(self) -> List[GradientTimeEvent]:
        q = get_gradient_time_queue()
        events: List[GradientTimeEvent] = []

        while True:
            try:
                evt = q.get_nowait()
            except Empty:
                break

            if evt.try_resolve():
                events.append(evt)
            else:
                # FIFO guarantee: later events cannot resolve earlier
                try:
                    q.put_nowait(evt)
                except Full:
                    self.logger.warning(
                        "[TraceML] GradientTime queue full on requeue"
                    )
                break

        return events

    def _save_events(self, events: List[GradientTimeEvent]) -> None:
        """
        Save raw gradient timing events into per-layer tables.
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

    def sample(self):
        """
        Drain → resolve → save raw events
        """
        try:
            events = self._drain_queue()
            self._save_events(events)
        except Exception as e:
            self.logger.error(
                f"[TraceML] GradientTimeSampler error: {e}"
            )
