# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

"""Step capture, event contracts, and the training-to-sampler handoff.

Measurement utilities submit timing events and the final memory snapshot to one
active capture. A successful step publishes that capture through the existing
timing and memory queues; an aborted step publishes nothing. Samplers continue
to drain the queues independently, and only the timing sampler resolves CUDA
events.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Full, Queue
from typing import Optional, TypeVar

import torch

from traceml_ai.runtime.state import should_record_trace_events
from traceml_ai.utils.cuda_event_pool import return_cuda_event


class TimeScope(str, Enum):
    """Whether a timing region belongs to a training step or global work."""

    STEP = "step"
    GLOBAL = "global"


@dataclass
class TimeEvent:
    """One timing measurement, possibly holding unresolved CUDA events."""

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
        """Resolve on the sampler side without blocking; return readiness."""
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


@dataclass
class StepTimeBatch:
    """Timing events grouped by the producer's existing step boundary.

    Capture completion assigns both ``step`` and each event's step number.
    Publication transfers the batch to the sampler without copying its events.
    Producers must not modify its membership or step after publication.
    """

    step: int
    events: list[TimeEvent] = field(default_factory=list)


@dataclass
class StepMemoryEvent:
    """Process allocator peak on a device, measured at the step boundary.

    Peaks are bytes, or ``None`` when CUDA memory is not applicable. Capture
    completion assigns ``step`` before publication; producers must not modify
    the event afterward. ``timestamp`` is the measurement time, not drain time.
    """

    step: int
    device: str
    timestamp: float
    peak_allocated: Optional[float]
    peak_reserved: Optional[float]


@dataclass
class StepCapture:
    """Own measurements for one pending caller-defined training step.

    Input timing may be recorded before a framework opens its explicit step
    callback, so ``begin_step_capture`` adopts the current capture rather than
    replacing it. Callers must close timing regions before finalizing through
    ``complete_step_capture`` or ``abort_step_capture``. These helpers detach
    the capture, making repeated completion or abort calls harmless.
    """

    timing_events: list[TimeEvent] = field(default_factory=list)
    memory_event: Optional[StepMemoryEvent] = None
    _closed: bool = field(default=False, init=False, repr=False)

    def record_timing(self, event: TimeEvent) -> None:
        """Add a timing event while this capture is active."""
        if not self._closed:
            self.timing_events.append(event)

    def record_memory(self, event: StepMemoryEvent) -> None:
        """Retain the final memory snapshot measured for this step."""
        if not self._closed:
            self.memory_event = event

    def _complete(self, step: int) -> bool:
        """Assign one step number and publish this capture once."""
        if self._closed:
            return False
        self._closed = True

        memory_event = self.memory_event
        timing_events = self.timing_events
        self.memory_event = None
        self.timing_events = []

        if memory_event is not None:
            memory_event.step = step
            publish_step_memory_event(memory_event)

        if timing_events:
            for event in timing_events:
                event.step = step
            publish_step_time_batch(
                StepTimeBatch(step=step, events=timing_events)
            )
        return True

    def _abort(self) -> bool:
        """Discard this capture once without publishing partial telemetry."""
        if self._closed:
            return False
        self._closed = True

        # Unresolved CUDA events must not return to the reuse pool. Dropping
        # these references lets PyTorch retire them safely after an uncommon
        # failed step without synchronizing training.
        self.timing_events = []
        self.memory_event = None
        return True


_STEP_TIME_QUEUE: Queue[StepTimeBatch] = Queue(maxsize=2048)
_STEP_MEMORY_QUEUE: Queue[StepMemoryEvent] = Queue(maxsize=2048)
_ACTIVE_STEP_CAPTURE = StepCapture()
_T = TypeVar("_T")


def _drain_queue(queue: Queue[_T]) -> list[_T]:
    """Transfer available records without waiting or resolving CUDA events."""
    items: list[_T] = []
    while True:
        try:
            item = queue.get_nowait()
        except Empty:
            break
        except Exception:  # noqa: BLE001 - preserve best-effort queue draining
            break

        if item is not None:
            items.append(item)
    return items


def begin_step_capture() -> StepCapture:
    """Return the active capture, including any input events already recorded."""
    return _ACTIVE_STEP_CAPTURE


def record_step_time_event(event: TimeEvent) -> None:
    """Submit a timing event to the active step capture."""
    _ACTIVE_STEP_CAPTURE.record_timing(event)


def record_step_memory_event(event: StepMemoryEvent) -> None:
    """Submit the latest memory snapshot to the active step capture."""
    _ACTIVE_STEP_CAPTURE.record_memory(event)


def _detach_step_capture(capture: StepCapture) -> bool:
    """Replace the active capture only when ``capture`` still owns it."""
    global _ACTIVE_STEP_CAPTURE
    if capture is not _ACTIVE_STEP_CAPTURE or capture._closed:
        return False
    _ACTIVE_STEP_CAPTURE = StepCapture()
    return True


def complete_step_capture(capture: StepCapture, step: int) -> bool:
    """Detach a completed capture and publish it while recording is enabled."""
    if not _detach_step_capture(capture):
        return False
    if (
        os.environ.get("TRACEML_DISABLED") == "1"
        or not should_record_trace_events()
    ):
        return capture._abort()
    return capture._complete(step)


def abort_step_capture(capture: StepCapture) -> bool:
    """Detach and discard an unfinished capture exactly once."""
    if not _detach_step_capture(capture):
        return False
    return capture._abort()


def publish_step_time_batch(batch: StepTimeBatch) -> None:
    """Enqueue without blocking; retain the existing drop-on-full policy."""
    try:
        _STEP_TIME_QUEUE.put_nowait(batch)
    except Full:
        print(
            f"[TraceML:Timing] Step queue full, dropping step batch {batch.step}",
            file=sys.stderr,
        )


def drain_step_time_batches() -> list[StepTimeBatch]:
    """Take available batches in FIFO order, including after recording stops.

    This only transfers references. CUDA readiness and the pending FIFO remain
    the timing sampler's responsibility.
    """
    return _drain_queue(_STEP_TIME_QUEUE)


def publish_step_memory_event(event: StepMemoryEvent) -> None:
    """Enqueue without blocking; retain the existing drop-on-full policy."""
    try:
        _STEP_MEMORY_QUEUE.put_nowait(event)
    except Full:
        print(
            f"[TraceML:StepMemory] Queue full, dropping event for step "
            f"{event.step} on {event.device}",
            file=sys.stderr,
        )


def drain_step_memory_events() -> list[StepMemoryEvent]:
    """Take available memory snapshots, including after recording stops."""
    return _drain_queue(_STEP_MEMORY_QUEUE)
