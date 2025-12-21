from dataclasses import dataclass
from typing import Dict, List, Optional
from queue import Queue, Full
import sys
import torch

# Shared queue for timing events
step_time_queue: Queue = Queue(maxsize=2048)


@dataclass
class StepTimeEvent:
    """
    Represents a single timing measurement of a user function.

    For GPU timing, CUDA events are stored and resolved later
    by the sampler whereas CPU timing is resolved immediately.
    """

    name: str
    device: str
    cpu_start: float
    cpu_end: float
    gpu_start: Optional[torch.cuda.Event] = None
    gpu_end: Optional[torch.cuda.Event] = None
    gpu_time_ms: Optional[float] = None
    resolved: bool = False

    def try_resolve(self) -> bool:
        """
        Attempt to resolve GPU timings, non-blocking.
        Returns:
            bool: True if fully resolved (CPU-only or GPU completed).
        """
        if self.resolved:
            return True

        if self.gpu_start is not None and self.gpu_end is not None:
            if self.gpu_end.query():  # non-blocking readiness check
                # GPU finished then safe to measure
                self.gpu_time_ms = self.gpu_start.elapsed_time(self.gpu_end)
                # Free CUDA event references to release GPU memory
                self.gpu_start = None
                self.gpu_end = None
                self.resolved = True
        else:
            # CPU-only timing
            self.resolved = True

        return self.resolved


def get_steptimer_queue() -> Queue:
    """Return the shared queue for step timing events."""
    return step_time_queue


def record_step_time_event(evt: StepTimeEvent):
    """Try to enqueue a timing event without blocking."""
    try:
        step_time_queue.put_nowait(evt)
    except Full:
        print(
            f"[TraceML:StepTimer] Queue full, dropping event {evt.name}",
            file=sys.stderr,
        )
