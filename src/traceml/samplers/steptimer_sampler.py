from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, Any, Optional, Deque, List
from queue import Empty, Full

import numpy as np

from traceml.utils.steptimer import StepTimeEvent, get_steptimer_queue
from .base_sampler import BaseSampler
from traceml.loggers.error_log import setup_error_logger, get_error_logger


@dataclass
class StepTimeSnapshot:
    """
    Snapshot of timings in the last interval.
    Contains average and max duration for each event name.
    """
    event_stats: Dict[str, Dict[str, float]]


class StepTimerSampler(BaseSampler):
    """
    Sampler that collects timing events from the step_time_queue,
    resolves them (CPU + GPU), and aggregates statistics.
    """

    def __init__(self, maxlen: int = 10_000):
        super().__init__()
        setup_error_logger()
        self.logger = get_error_logger("StepTimeSampler")

        # Internal history log: event_name -> deque of durations
        self.cpu_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=maxlen))
        self.gpu_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=maxlen))

        self.latest: Optional[StepTimeSnapshot] = None

    def _drain_queue(self) -> List[StepTimeEvent]:
        q = get_steptimer_queue()
        events = []
        while True:
            try:
                evt = q.get_nowait()
            except Empty:
                break

            if evt.try_resolve():
                events.append(evt)
            else:
                try:
                    q.put_nowait(evt)  # push back unresolved event
                except Full:
                    # extremely rare unless queue is tiny
                    self.logger.warning("[TraceML] StepTime queue full on requeue")
                break  # avoid spinning on unresolved GPU work
        return events

    def _aggregate_events(self, events: List[StepTimeEvent]) -> StepTimeSnapshot:
        """
        Aggregate CPU and GPU timings by event name (average + max).
        """
        stats: Dict[str, Dict[str, float]] = {}

        grouped_cpu: Dict[str, List[float]] = defaultdict(list)
        grouped_gpu: Dict[str, List[float]] = defaultdict(list)

        for evt in events:
            cpu_dur = evt.cpu_end - evt.cpu_start
            grouped_cpu[evt.name].append(cpu_dur)
            self.cpu_history[evt.name].append(cpu_dur)

            if evt.gpu_time_ms is not None:
                gpu_dur = evt.gpu_time_ms / 1000.0  # ms → s
                grouped_gpu[evt.name].append(gpu_dur)
                self.gpu_history[evt.name].append(gpu_dur)

        for name in set(list(grouped_cpu.keys()) + list(grouped_gpu.keys())):
            cpu_vals = grouped_cpu.get(name, [])
            gpu_vals = grouped_gpu.get(name, [])

            stats[name] = {
                "cpu_avg_s": float(np.mean(cpu_vals)) if cpu_vals else 0.0,
                "cpu_max_s": float(np.max(cpu_vals)) if cpu_vals else 0.0,
                "gpu_avg_s": float(np.mean(gpu_vals)) if gpu_vals else 0.0,
                "gpu_max_s": float(np.max(gpu_vals)) if gpu_vals else 0.0,
            }

        return StepTimeSnapshot(event_stats=stats)

    def sample(self) -> Dict[str, Any]:
        """
        Collect all resolved events since last call and return a snapshot.
        """
        try:
            events = self._drain_queue()
            snap = self._aggregate_events(events)
            self.latest = snap

            envelope =  self.make_snapshot(
                ok=True,
                message="sampled successfully",
                source="step_timer",
                data=snap.event_stats,
            )
        except Exception as e:
            self.logger.error(f"[TraceML] StepTime sampling error: {e}")
            self.latest = None
            envelope =  self.make_snapshot(
                ok=False,
                message=f"sampling failed: {e}",
                source="step_timer",
                data=None,
            )
        return self.snapshot_dict(envelope)


    def get_summary(self) -> Dict[str, Any]:
        """
        Compute max and average timing across all iterations.
        """
        summary: Dict[str, Any] = {}
        try:
            for name, vals in self.cpu_history.items():
                cpu_arr = np.array(vals)
                summary[f"{name}_cpu_avg_s"] = float(np.mean(cpu_arr)) if cpu_arr.size else 0.0
                summary[f"{name}_cpu_max_s"] = float(np.max(cpu_arr)) if cpu_arr.size else 0.0

            for name, vals in self.gpu_history.items():
                gpu_arr = np.array(vals)
                summary[f"{name}_gpu_avg_s"] = float(np.mean(gpu_arr)) if gpu_arr.size else 0.0
                summary[f"{name}_gpu_max_s"] = float(np.max(gpu_arr)) if gpu_arr.size else 0.0

            return summary
        except Exception as e:
            self.logger.error(f"[TraceML] StepTime summary calculation error: {e}")
            return {"error": str(e)}