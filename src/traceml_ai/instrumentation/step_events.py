# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

"""Timing event contracts and the training-to-sampler handoff.

Producers publish already-grouped step batches. The sampler drains them in
insertion order and owns CUDA resolution; publishing and draining never wait
for the GPU. Measurement and pending-step buffering remain in ``utils.timing``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Full, Queue
from typing import Optional

import torch

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

    Flush assigns both ``step`` and each event's step number. Publication
    transfers the batch to the sampler without copying its events. Producers
    must not append to it or change its step identity after publication.
    """

    step: int
    events: list[TimeEvent] = field(default_factory=list)


_STEP_TIME_QUEUE: Queue[StepTimeBatch] = Queue(maxsize=2048)


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
    batches: list[StepTimeBatch] = []
    while True:
        try:
            batch = _STEP_TIME_QUEUE.get_nowait()
        except Empty:
            break
        except Exception:  # noqa: BLE001 - preserve best-effort queue draining
            break

        if batch is not None:
            batches.append(batch)
    return batches
