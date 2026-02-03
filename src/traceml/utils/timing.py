"""
TraceML Timing Core

This module defines the unified timing pipeline used by TraceML.
It supports both step-scoped and global timing while preserving
a single ordered event stream per rank.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Deque
from queue import Queue, Full
from collections import deque
import sys
import time
import torch
from contextlib import contextmanager

from traceml.utils.cuda_event_pool import get_cuda_event, return_cuda_event


class TimeScope(str, Enum):
    """
    Semantic scope of a timing event.

    STEP belongs to a specific training step
    GLOBAL occurs outside the step loop (init, checkpoint, etc.)
    """
    STEP = "step"
    GLOBAL = "global"


@dataclass
class TimeEvent:
    """
    Represents a single timing measurement.

    GPU timing uses CUDA events and is resolved asynchronously
    by the sampler to avoid synchronization.
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
    scope: TimeScope = TimeScope.STEP


    def try_resolve(self) -> bool:
        """
        Attempt to resolve GPU timing without blocking.

        Returns
        -------
        bool
            True if the event is fully resolved.
        """
        if self.resolved:
            return True

        if self.gpu_start and self.gpu_end:
            if self.gpu_end.query():
                self.gpu_time_ms = self.gpu_start.elapsed_time(self.gpu_end)

                return_cuda_event(self.gpu_start)
                return_cuda_event(self.gpu_end)

                self.gpu_start = None
                self.gpu_end = None
                self.resolved = True
        else:
            self.resolved = True

        return self.resolved


_TIME_QUEUE: Queue = Queue(maxsize=2048)
_STEP_BUFFER: Deque[TimeEvent] = deque()


def get_time_queue() -> Queue:
    """Return the shared timing queue."""
    return _TIME_QUEUE


def _enqueue(evt: TimeEvent) -> None:
    """Best-effort enqueue without blocking."""
    try:
        _TIME_QUEUE.put_nowait(evt)
    except Full:
        print(
            f"[TraceML:Timing] Queue full, dropping event '{evt.name}'",
            file=sys.stderr,
        )


def record_event(evt: TimeEvent) -> None:
    """
    Record a timing event.

    STEP events are buffered until flush.
    GLOBAL events are enqueued immediately.
    """
    if evt.scope == TimeScope.STEP:
        _STEP_BUFFER.append(evt)
    else:
        _enqueue(evt)


def flush_step_time_buffer(step: int) -> None:
    """
    Flush buffered step events and assign step index.
    """
    while _STEP_BUFFER:
        evt = _STEP_BUFFER.popleft()
        evt.step = step
        _enqueue(evt)



@contextmanager
def timed_region(
    name: str,
    scope: TimeScope = TimeScope.STEP,
    use_gpu: bool = True,
):
    """
    Context manager for timing arbitrary code regions.

    Guarantees
    ----------
    - User code always runs
    - Timing is best-effort
    - User exceptions are never swallowed
    """

    cpu_start = time.time()
    evt = None

    # Best-effort setup
    try:
        if use_gpu and torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            start_evt = get_cuda_event()
            end_evt = get_cuda_event()
            start_evt.record()
        else:
            device = "cpu"
            start_evt = end_evt = None
    except Exception as e:
        # Timing setup failed → disable timing for this region
        print(
            f"[TraceML] timed_region setup failed for '{name}': {e}",
            file=sys.stderr,
        )
        start_evt = end_evt = None
        device = "cpu"

    try:
        # User code ALWAYS runs
        yield
    finally:
        try:
            cpu_end = time.time()

            if start_evt and end_evt:
                end_evt.record()
                evt = TimeEvent(
                    name=name,
                    device=device,
                    cpu_start=cpu_start,
                    cpu_end=cpu_end,
                    gpu_start=start_evt,
                    gpu_end=end_evt,
                    scope=scope,
                )
            else:
                evt = TimeEvent(
                    name=name,
                    device=device,
                    cpu_start=cpu_start,
                    cpu_end=cpu_end,
                    scope=scope,
                )

            record_event(evt)

        except Exception as e:
            # Absolutely nothing here may break training
            print(
                f"[TraceML] timed_region teardown failed for '{name}': {e}",
                file=sys.stderr,
            )