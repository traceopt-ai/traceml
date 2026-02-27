"""
GlobalTimeSampler

GLOBAL timing sampler for TraceML.

Consumes TimeEvent objects from the GLOBAL timing queue, resolves CPU events
immediately, resolves GPU events asynchronously in strict FIFO order, and
persists resolved events into DB tables (one table per event name).
"""

from collections import deque
from queue import Empty
from typing import List, Optional

from traceml.loggers.error_log import get_error_logger
from traceml.samplers.schema.time_schema import TimeEventSample
from traceml.utils.timing import TimeEvent, get_global_time_queue

from .base_sampler import BaseSampler


class GlobalTimeSampler(BaseSampler):
    def __init__(self) -> None:
        self.sampler_name = "GlobalTimeSampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)
        self.sample_idx = 0
        self._local_gpu_q = deque()

    def _drain_global_queue(self) -> List[TimeEvent]:
        q = get_global_time_queue()
        ready: List[TimeEvent] = []

        while True:
            try:
                evt = q.get_nowait()
            except Empty:
                break

            if evt.gpu_start is None or evt.gpu_end is None:
                evt.try_resolve()
                ready.append(evt)
            else:
                self._local_gpu_q.append(evt)

        return ready

    def _drain_local_gpu_queue(self) -> List[TimeEvent]:
        ready: List[TimeEvent] = []
        while self._local_gpu_q:
            evt = self._local_gpu_q[0]
            if evt.try_resolve():
                self._local_gpu_q.popleft()
                ready.append(evt)
            else:
                break
        return ready

    @staticmethod
    def _cpu_duration_ms(evt: TimeEvent) -> float:
        return (evt.cpu_end - evt.cpu_start) * 1000.0

    def _event_to_sample(self, evt: TimeEvent) -> Optional[TimeEventSample]:
        try:
            is_gpu = evt.gpu_time_ms is not None
            duration_ms = float(evt.gpu_time_ms) if is_gpu else float(self._cpu_duration_ms(evt))
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
        for evt in events:
            sample = self._event_to_sample(evt)
            if sample is None:
                continue
            self.db.add_record(evt.name, sample.to_wire())

    def sample(self) -> None:
        self.sample_idx += 1
        try:
            ready_cpu = self._drain_global_queue()
            ready_gpu = self._drain_local_gpu_queue()
            self._save_events(ready_cpu + ready_gpu)
        except Exception as e:
            self.logger.error(f"[TraceML] GlobalTimeSampler error: {e}")