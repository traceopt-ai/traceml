import time
import sys
from dataclasses import dataclass, field
from typing import Optional, Callable
from queue import Queue, Full

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


def trace_timestep(name: str, use_gpu: bool = True) -> Callable:
    """
    Decorator to measure execution time of a function.

    Args:
        name (str): Label for this timer.
        use_gpu (bool): If True and CUDA is available, record GPU timing
                        via CUDA events. Otherwise, only CPU wall-time.
    Returns:
        Callable: Wrapped function with timing instrumentation.
    """

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            cpu_start = time.time()
            evt: StepTimeEvent

            if use_gpu and torch.cuda.is_available():
                # Record CUDA events for GPU timing
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()          # queued in current CUDA stream
                result = func(*args, **kwargs)
                end_event.record()            # queued after kernels

                cpu_end = time.time()

                evt = StepTimeEvent(
                    name=name,
                    cpu_start=cpu_start,
                    cpu_end=cpu_end,
                    gpu_start=start_event,
                    gpu_end=end_event,
                )
            else:
                # CPU-only timing
                result = func(*args, **kwargs)
                cpu_end = time.time()

                evt = StepTimeEvent(
                    name=name,
                    cpu_start=cpu_start,
                    cpu_end=cpu_end,
                )

            try:
                step_time_queue.put_nowait(evt)
            except Full:
                print(
                    f"[TraceML:StepTimer] Queue full, dropping event {name}",
                    file=sys.stderr,
                )

            return result

        return wrapper

    return decorator
