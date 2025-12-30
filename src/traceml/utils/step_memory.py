from dataclasses import dataclass
from queue import Queue, Full
import sys
import torch
import torch.nn as nn


# Shared queue for step-level memory events
step_memory_queue: Queue = Queue(maxsize=128)


def get_step_memory_queue() -> Queue:
    return step_memory_queue


@dataclass
class StepMemoryEvent:
    """
    Peak GPU memory during a TraceML step.
    """

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
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        """
        Reset CUDA peak memory counters at step start.
        """
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def record(self):
        """
        Record peak memory at step end.
        """
        if self.device.type != "cuda":
            return

        evt = StepMemoryEvent(
            model_id=self.model_id,
            device=str(self.device),
            peak_allocated_mb=torch.cuda.max_memory_allocated(self.device),
            peak_reserved_mb=torch.cuda.max_memory_reserved(self.device),
        )
        try:
            step_memory_queue.put_nowait(evt)
        except Full:
            print("[TraceML] Error in StepMemoryTracker", file=sys.stderr)
