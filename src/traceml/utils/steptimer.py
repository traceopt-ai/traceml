from dataclasses import dataclass
from typing import Optional
from queue import Queue, Full
from collections import deque
import sys
import torch
import time
from contextlib import contextmanager
from traceml.utils.cuda_event_pool import get_cuda_event, return_cuda_event


_temp_step_time_buffer: deque = deque()

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
    step: int = -1

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

                return_cuda_event(self.gpu_start)
                return_cuda_event(self.gpu_end)

                self.gpu_start = None
                self.gpu_end = None
                self.resolved = True
        else:
            # CPU-only timing
            self.resolved = True

        return self.resolved

def begin_timed_region(name: str, use_gpu: bool = True):
    """Starts a timer and returns the state needed to stop it later."""
    cpu_start = time.time()
    gpu_start, gpu_end = None, None
    device = "cpu"

    if use_gpu and torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
        gpu_start, gpu_end = get_cuda_event(), get_cuda_event()
        gpu_start.record()
    
    return {
        "name": name, 
        "device": device, 
        "cpu_start": cpu_start,
        "gpu_start": gpu_start, 
        "gpu_end": gpu_end
    }

def end_timed_region(state: dict):
    """Stops the timer using the state from begin_timed_region."""
    cpu_end = time.time()
    if state["gpu_end"]:
        state["gpu_end"].record()
    
    evt = StepTimeEvent(
        name=state["name"], 
        device=state["device"],
        cpu_start=state["cpu_start"], 
        cpu_end=cpu_end,
        gpu_start=state["gpu_start"], 
        gpu_end=state["gpu_end"],
    )
    record_step_time_event(evt)

def get_steptimer_queue() -> Queue:
    """Return the shared queue for step timing events."""
    return step_time_queue


def record_step_time_event(evt: StepTimeEvent, on_queue=False):
    """Try to enqueue a timing event without blocking."""
    if on_queue:
        try:
            step_time_queue.put_nowait(evt)
        except Full:
            print(
                f"[TraceML:StepTimer] Queue full, dropping event {evt.name}",
                file=sys.stderr,
            )
    else:
        _temp_step_time_buffer.append(evt)


@contextmanager
def timed_region(name: str, use_gpu: bool = True):
    """Context manager for timing a block of code."""
    state = begin_timed_region(name, use_gpu)
    try:
        yield
    finally:
        end_timed_region(state)

def flush_step_time_buffer(step: int) -> None:
    """Flush the temporary step time buffer to the shared queue."""
    while _temp_step_time_buffer:
        evt = _temp_step_time_buffer.popleft()
        evt.step = step
        try:
            step_time_queue.put_nowait(evt)
        except Full:
            print(
                f"[TraceML:StepTimer] Queue full, dropping event {evt.name}",
                file=sys.stderr,
            )
