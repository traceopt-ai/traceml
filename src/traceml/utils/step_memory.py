import sys
from dataclasses import dataclass
from queue import Full, Queue
from typing import Dict

import torch
import torch.nn as nn

step_memory_queue: Queue = Queue(maxsize=2048)

_temp_step_memory_buffer: Dict = {}


@dataclass
class StepMemoryEvent:
    """
    Peak GPU memory during a TraceML step.
    """

    step: int
    model_id: int
    device: str
    peak_allocated_mb: float
    peak_reserved_mb: float


class StepMemoryTracker:
    """
    Tracks peak CUDA memory across a train step.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model_id = id(model)

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

    def reset(self):
        """
        Reset CUDA peak memory counters at step start.
        """
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def record(self):
        """
        Record peak memory at step end.

        Semantics:
        - CUDA: record real peak allocated / reserved memory
        - Non-CUDA: emit a sentinel event with 0 values
          (means: step memory not applicable on this device)
        """
        if self.device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
        else:
            peak_allocated = 0.0
            peak_reserved = 0.0

        evt = StepMemoryEvent(
            model_id=self.model_id,
            device=str(self.device),
            peak_allocated_mb=float(peak_allocated),
            peak_reserved_mb=float(peak_reserved),
            step=-1,  # filled during flush
        )
        _temp_step_memory_buffer[self.model_id] = evt


def flush_step_memory_buffer(model: nn.Module, step: int) -> None:
    model_id = id(model)

    evt = _temp_step_memory_buffer.pop(model_id, None)
    if evt is None:
        return

    evt.step = step
    try:
        step_memory_queue.put_nowait(evt)
    except Full:
        print(
            f"[TraceML:StepMemory] Queue full, dropping event for model {evt.model_id}",
            file=sys.stderr,
        )
