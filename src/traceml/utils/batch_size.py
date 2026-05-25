"""
TraceML Batch Size (Bytes) Capture

Buffers H2D byte counts per training step; flushed once per step as a
BatchSizeBatch onto a shared queue. Producer = training thread (inside
trace_step), consumer = sampling thread. Multiple transfers in a step
are summed by the sampler.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass, field
from queue import Full, Queue
from typing import Deque, List

TRACEML_DISABLED = os.environ.get("TRACEML_DISABLED") == "1"


@dataclass
class BatchSizeEvent:
    """A single host-to-device transfer observed inside a training step."""

    bytes_count: int
    step: int = -1


@dataclass
class BatchSizeBatch:
    """One optimizer-step worth of H2D byte events."""

    step: int
    events: List[BatchSizeEvent] = field(default_factory=list)


_BATCH_SIZE_QUEUE: Queue = Queue(maxsize=2048)
_BATCH_SIZE_BUFFER: Deque[BatchSizeEvent] = deque()


def get_batch_size_queue() -> Queue:
    """Return the shared cross-thread BatchSizeBatch queue."""
    return _BATCH_SIZE_QUEUE


def record_batch_size_bytes(bytes_count: int) -> None:
    """
    Buffer one H2D byte observation for the current step.

    The value is appended to the per-step buffer and flushed as part of a
    BatchSizeBatch at the next call to flush_batch_size_buffer(step).
    Best-effort: invalid values are ignored.
    """
    if TRACEML_DISABLED:
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
    if TRACEML_DISABLED:
        return
    if not _BATCH_SIZE_BUFFER:
        return

    events: List[BatchSizeEvent] = []
    while _BATCH_SIZE_BUFFER:
        evt = _BATCH_SIZE_BUFFER.popleft()
        evt.step = step
        events.append(evt)

    try:
        _BATCH_SIZE_QUEUE.put_nowait(BatchSizeBatch(step=step, events=events))
    except Full:
        print(
            f"[TraceML:BatchSize] Queue full, dropping step batch {step}",
            file=sys.stderr,
        )


def tensor_bytes(obj: object) -> int:
    """
    Best-effort byte sizing for an object passed to ``.to(device)``.

    Handles:
    - torch.Tensor: element_size() * numel()
    - dict / list / tuple of tensors (1 level deep): sum of contained tensors
    - everything else: 0 (caller decides whether to record)
    """
    try:
        import torch  # local import: utils must be import-safe without torch
    except Exception:
        return 0

    if isinstance(obj, torch.Tensor):
        try:
            return int(obj.element_size()) * int(obj.numel())
        except Exception:
            return 0

    if isinstance(obj, dict):
        total = 0
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                try:
                    total += int(v.element_size()) * int(v.numel())
                except Exception:
                    pass
        return total

    if isinstance(obj, (list, tuple)):
        total = 0
        for v in obj:
            if isinstance(v, torch.Tensor):
                try:
                    total += int(v.element_size()) * int(v.numel())
                except Exception:
                    pass
        return total

    return 0
