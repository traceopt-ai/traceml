"""
TraceML Batch Size (Bytes) Capture

Buffers the tensor bytes of each dataloader fetch per training step;
flushed once per step as a BatchSizeBatch onto a shared queue. Producer =
training thread (the dataloader fetch path), consumer = sampling thread.
Multiple fetches in a step (gradient accumulation) are summed by the
sampler.

Sizing at the dataloader keeps the metric device-agnostic: CPU-only
training records the same batch bytes as GPU training.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Full, Queue
from typing import Deque, List

from traceml_ai.runtime.state import should_record_trace_events


def _traceml_disabled() -> bool:
    return os.environ.get("TRACEML_DISABLED") == "1"


def should_record_batch_size() -> bool:
    """
    Cheap gate for the fetch-path call sites, checked BEFORE sizing a
    batch (mirroring how ``timed_region`` checks its gate before doing
    any work) so fetches outside a recorded window cost nothing.
    """
    return not _traceml_disabled() and should_record_trace_events()


@dataclass
class BatchSizeEvent:
    """A single dataloader fetch observed inside a training step."""

    bytes_count: int
    step: int = -1


@dataclass
class BatchSizeBatch:
    """One optimizer-step worth of dataloader-fetch byte events."""

    step: int
    timestamp: float = 0.0
    events: List[BatchSizeEvent] = field(default_factory=list)


_BATCH_SIZE_QUEUE: Queue = Queue(maxsize=2048)
# Bounded like the queue: a loop that fetches without ever flushing
# (no trace_step) must not grow memory without limit.
_BATCH_SIZE_BUFFER: Deque[BatchSizeEvent] = deque(maxlen=2048)


def get_batch_size_queue() -> Queue:
    """Return the shared cross-thread BatchSizeBatch queue."""
    return _BATCH_SIZE_QUEUE


def record_batch_size_bytes(bytes_count: int) -> None:
    """
    Buffer one dataloader-fetch byte observation for the current step.

    The value is appended to the per-step buffer and flushed as part of a
    BatchSizeBatch at the next call to flush_batch_size_buffer(step).
    Recording is gated on the trace recording state (mirroring
    ``timed_region``) so fetches outside a recorded window are dropped.
    Best-effort: invalid values are ignored.
    """
    if _traceml_disabled() or not should_record_trace_events():
        return

    try:
        n = int(bytes_count)
    except Exception:
        return

    if n <= 0:
        return

    _BATCH_SIZE_BUFFER.append(BatchSizeEvent(bytes_count=n))


def flush_batch_size_buffer(step: int) -> None:
    """
    Flush buffered BatchSizeEvents as a single BatchSizeBatch.

    Called once per optimizer step, after trace_step exits.
    """
    if _traceml_disabled():
        return
    if not _BATCH_SIZE_BUFFER:
        return

    events: List[BatchSizeEvent] = []
    while _BATCH_SIZE_BUFFER:
        evt = _BATCH_SIZE_BUFFER.popleft()
        evt.step = step
        events.append(evt)

    try:
        # Stamped here, at step flush, so the persisted row carries the
        # step-end time rather than the sampler thread's drain time.
        _BATCH_SIZE_QUEUE.put_nowait(
            BatchSizeBatch(step=step, timestamp=time.time(), events=events)
        )
    except Full:
        print(
            f"[TraceML:BatchSize] Queue full, dropping step batch {step}",
            file=sys.stderr,
        )


def tensor_bytes(obj: object) -> int:
    """
    Best-effort byte sizing for a batch as it leaves the dataloader.

    Handles:
    - torch.Tensor: element_size() * numel()
    - dict / list / tuple of tensors (1 level deep): sum of contained tensors
    - everything else: 0 (caller decides whether to record)

    Never raises: this runs on the user's fetch path, and container
    subclasses can fail in arbitrary ways during iteration.
    """
    try:
        import torch  # local import: utils must be import-safe without torch

        if isinstance(obj, torch.Tensor):
            return int(obj.element_size()) * int(obj.numel())

        if isinstance(obj, dict):
            total = 0
            for v in obj.values():
                if isinstance(v, torch.Tensor):
                    total += int(v.element_size()) * int(v.numel())
            return total

        if isinstance(obj, (list, tuple)):
            total = 0
            for v in obj:
                if isinstance(v, torch.Tensor):
                    total += int(v.element_size()) * int(v.numel())
            return total
    except Exception:
        return 0

    return 0
