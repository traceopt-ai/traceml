"""
TimeSampler

Step-level timing sampler for TraceML.

Responsibilities
----------------
- Consume TimeEvent objects from the global timing queue
- Resolve CPU events immediately
- Resolve GPU events asynchronously in strict FIFO order
- Persist resolved events into DB tables (one table per event name)

This sampler is *clock-agnostic*:
- CPU vs GPU timing is encoded via `is_gpu`
- Duration is always stored in `duration_ms`
"""

from collections import deque
from queue import Empty
from typing import List, Optional

from traceml.loggers.error_log import get_error_logger
from traceml.samplers.schema.time_schema import TimeEventSample
from traceml.utils.timing import TimeEvent, get_time_queue

from .base_sampler import BaseSampler


class TimeSampler(BaseSampler):
    """
    Step-level timing sampler.

    Design
    ------
    - One totally ordered timing stream per rank
    - One DB table per event name (`evt.name`)
    - CPU events never block GPU events
    - GPU events resolve strictly FIFO

    DB row schema
    -------------
    Uses TimeEventSample.to_wire()
    """

    def __init__(self) -> None:
        self.sampler_name = "TimeSampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)
        self.sample_idx = 0

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

    @staticmethod
    def _cpu_duration_ms(evt: TimeEvent) -> float:
        """
        Compute CPU wall duration in milliseconds.
        """
        return (evt.cpu_end - evt.cpu_start) * 1000.0

    def _event_to_sample(self, evt: TimeEvent) -> Optional[TimeEventSample]:
        """
        Convert a resolved TimeEvent into a TimeEventSample.
        """
        try:
            is_gpu = evt.gpu_time_ms is not None
            duration_ms = (
                float(evt.gpu_time_ms)
                if is_gpu
                else float(self._cpu_duration_ms(evt))
            )

            return TimeEventSample(
                sample_idx=self.sample_idx,
                timestamp=float(evt.cpu_end),
                step=int(evt.step),
                scope=str(evt.scope),
                device=str(evt.device),
                is_gpu=bool(is_gpu),
                duration_ms=duration_ms,
            )
        except Exception:
            return None

    def _save_events(self, events: List[TimeEvent]) -> None:
        """
        Persist resolved events into DB tables.

        Table name = evt.name
        Row schema = TimeEventSample.to_wire()
        """
        for evt in events:
            sample = self._event_to_sample(evt)
            if sample is None:
                continue
            self.db.add_record(evt.name, sample.to_wire())

    def sample(self) -> None:
        """
        Sampling loop:

        1) Drain global timing queue
           - CPU events resolved immediately
           - GPU events staged locally
        2) Resolve local GPU queue (FIFO)
        3) Persist all newly resolved events
        """
        self.sample_idx += 1
        try:
            ready_cpu = self._drain_global_queue()
            ready_gpu = self._drain_local_gpu_queue()
            self._save_events(ready_cpu + ready_gpu)
        except Exception as e:
            self.logger.error(f"[TraceML] TimeSampler error: {e}")
