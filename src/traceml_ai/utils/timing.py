"""Timing measurement for TraceML instrumentation.

Step-scoped events are submitted to the active ``StepCapture``. The capture
owns finalization and queue publication; the timing sampler owns CUDA
resolution and aggregation. Global timing is currently not persisted.
"""

import os
import sys
import time
from contextlib import contextmanager
from queue import Full, Queue

import torch

from traceml_ai.instrumentation.step_events import (
    TimeEvent,
    TimeScope,
    record_step_time_event,
)
from traceml_ai.runtime.state import should_record_trace_events
from traceml_ai.utils.cuda_event_pool import get_cuda_event


def _traceml_disabled() -> bool:
    return os.environ.get("TRACEML_DISABLED") == "1"


_GLOBAL_TIME_QUEUE: Queue = Queue(maxsize=2048)


def get_global_time_queue() -> Queue:
    """Return the shared GLOBAL timing queue."""
    return _GLOBAL_TIME_QUEUE


def _enqueue_global(evt: TimeEvent) -> None:
    """Best-effort enqueue GLOBAL event without blocking."""
    try:
        pass
        # TODO: implement sampler for it if required
        # _GLOBAL_TIME_QUEUE.put_nowait(evt)
    except Full:
        print(
            f"[TraceML:Timing] Global queue full, dropping event '{evt.name}'",
            file=sys.stderr,
        )


def record_event(evt: TimeEvent) -> None:
    """
    Record a timing event.

    STEP events belong to the active capture until successful completion.
    GLOBAL timing is currently not persisted.
    """
    if _traceml_disabled() or not should_record_trace_events():
        return
    if evt.scope == TimeScope.STEP:
        record_step_time_event(evt)
    else:
        _enqueue_global(evt)


@contextmanager
def timed_region(
    name: str,
    scope: TimeScope = TimeScope.STEP,
    record_gpu_events: bool = True,
):
    """
    Context manager for timing arbitrary code regions.

    Guarantees
    ----------
    - User code always runs
    - Timing is best-effort
    - User exceptions are never swallowed

    Timing clocks
    -------------
    CPU wall time is always recorded. When ``record_gpu_events`` is true and
    PyTorch CUDA is available, TraceML also records CUDA stream events and
    resolves them later without synchronizing training. Timing events do not
    currently include GPU backend metadata; CUDA and ROCm-specific labeling can
    be added as a separate schema change when needed.
    """
    if _traceml_disabled() or not should_record_trace_events():
        yield
        return

    cpu_start = time.time()

    try:
        if record_gpu_events and torch.cuda.is_available():
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
