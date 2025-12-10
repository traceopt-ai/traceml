from typing import Dict, List
from queue import Empty, Full

from traceml.utils.steptimer import StepTimeEvent, get_steptimer_queue
from .base_sampler import BaseSampler
from traceml.loggers.error_log import get_error_logger


class StepTimerSampler(BaseSampler):
    """
    Drain-all step-timer sampler.

    Each sample():
        - drains the step_time_queue
        - resolves CPU + GPU timings via try_resolve()
        - saves raw timing to CPU or per-GPU tables

    Tables created:
        step_timer_cpu
        step_timer_cuda:0
        step_timer_cuda:1
        ...
    """

    def __init__(self) -> None:
        self.sampler_name = "StepTimerSampler"
        super().__init__(sampler_name=self.sampler_name)
        self.logger = get_error_logger(self.sampler_name)

        self.cpu_table = self.db.create_or_get_table("step_timer_cpu")
        self.gpu_tables: Dict[str, list] = {}

    def _get_gpu_table(self, device: str):
        """Create GPU table like 'step_timer_cuda:0' on demand."""
        if device not in self.gpu_tables:
            table_name = f"step_timer_{device.replace(':', '_')}"
            self.gpu_tables[device] = self.db.create_or_get_table(table_name)
        return self.gpu_tables[device]

    def _drain_queue(self) -> List[StepTimeEvent]:
        q = get_steptimer_queue()
        events = []

        while True:
            try:
                evt = q.get_nowait()
            except Empty:
                break

            # Resolve GPU event non-blocking
            if evt.try_resolve():
                events.append(evt)
            else:
                # Put back unresolved event
                try:
                    q.put_nowait(evt)
                except Full:
                    self.logger.warning("[TraceML] StepTimer queue full on requeue")
                break

        return events

    def _save_events(self, events: List[StepTimeEvent]) -> None:
        """
        Saves raw per-device timing into DB tables.
        CPU → step_timer_cpu
        GPU → step_timer_cuda_X
        """

        for evt in events:

            # Always save CPU time
            cpu_ms = (evt.cpu_end - evt.cpu_start) * 1000.0
            self.cpu_table.append(
                {
                    "timestamp": evt.cpu_end,
                    "event_name": evt.name,
                    "device": evt.device,
                    "duration_ms": float(cpu_ms),
                }
            )

            # Save GPU time only if available
            if evt.gpu_time_ms is not None:
                gpu_table = self._get_gpu_table(evt.device)
                gpu_table.append(
                    {
                        "timestamp": evt.cpu_end,
                        "event_name": evt.name,
                        "device": evt.device,
                        "duration_ms": float(evt.gpu_time_ms),
                    }
                )

    def sample(self):
        """Drain → save raw events"""
        try:
            events = self._drain_queue()
            self._save_events(events)
        except Exception as e:
            self.logger.error(f"[TraceML] StepTimerSampler error: {e}")
