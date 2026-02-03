from typing import List
from queue import Empty
from collections import deque

from traceml.utils.timing import TimeEvent, get_time_queue
from .base_sampler import BaseSampler
from traceml.loggers.error_log import get_error_logger


class TimeSampler(BaseSampler):
    """
    Design:
    - Single totally ordered timing stream per rank.
    - One DB table per event_name: event_name
    - CPU events are resolved immediately
    - GPU events are staged in a local FIFO queue
    - GPU events are resolved strictly in order (front-only)

    Table name: event_name
    Each row contains:
      - step
      - timestamp
      - device
      - is_gpu
      - duration_ms
    """

    def __init__(self) -> None:
        self.sampler_name = "TimeSampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)

        # Local FIFO queue for unresolved GPU events.
        # Keeps ordering and avoids GPU events blocking CPU events in the shared queue.
        self._local_gpu_q = deque()

    def _drain_global_queue(self) -> List[TimeEvent]:
        """
        Drain the shared queue completely.
        - CPU-only events are appended to `ready` (they resolve immediately).
        - GPU events are moved to the local GPU deque (no resolving here).
        """
        q = get_time_queue()
        ready: List[TimeEvent] = []

        while True:
            try:
                evt = q.get_nowait()
            except Empty:
                break

            # CPU-only events, resolve immediately.
            if evt.gpu_start is None or evt.gpu_end is None:
                evt.try_resolve()  # marks resolved
                ready.append(evt)
            else:
                # GPU events: keep FIFO in local queue; resolve later in order.
                self._local_gpu_q.append(evt)

        return ready

    def _drain_local_gpu_queue(self) -> List[TimeEvent]:
        """
        Resolve GPU events in strict FIFO order.
        We only attempt to resolve the *front*; if it isn't ready, stop.
        """
        ready: List[TimeEvent] = []

        while self._local_gpu_q:
            evt = self._local_gpu_q[0]  # peek front (preserves FIFO)
            if evt.try_resolve():
                self._local_gpu_q.popleft()
                ready.append(evt)
            else:
                break

        return ready

    def _save_events(self, events: List[TimeEvent]) -> None:
        """
        Save resolved events into per-event tables.
        """
        for evt in events:
            cpu_ms = (evt.cpu_end - evt.cpu_start) * 1000.0
            is_gpu = evt.gpu_time_ms is not None

            record = {
                "timestamp": float(evt.cpu_end),
                "step": int(evt.step),
                "scope": evt.scope,
                "event_name": evt.name,
                "device": evt.device,  # 'cpu' or 'cuda:0'
                "is_gpu": is_gpu,
                "duration_ms": float(evt.gpu_time_ms if is_gpu else cpu_ms),
            }
            self.db.add_record(evt.name, record)

    def sample(self):
        """
        1) Drain global queue fully (CPU events saved immediately; GPU events staged locally)
        2) Resolve local GPU queue from front until first unresolved
        3) Save all ready events
        """
        try:
            ready_cpu = self._drain_global_queue()
            ready_gpu = self._drain_local_gpu_queue()
            self._save_events(ready_cpu + ready_gpu)
        except Exception as e:
            self.logger.error(f"[TraceML] TimeSampler error: {e}")
