"""Step memory measurement for TraceML instrumentation.

Each read submits an allocator snapshot to the active ``StepCapture``. The
capture retains the final snapshot until the caller completes or aborts the
step. Device identity stays in the event. This path assumes one sequential
training-step producer and one tracked device per process.
"""

import os
import time

import torch
import torch.nn as nn

from traceml_ai.instrumentation.step_events import (
    StepMemoryEvent,
    record_step_memory_event,
)
from traceml_ai.runtime.state import should_record_trace_events


def _traceml_disabled() -> bool:
    return os.environ.get("TRACEML_DISABLED") == "1"


class StepMemoryTracker:
    """Track process allocator peaks on the model's device across a step.

    The model selects the device; these counters do not measure memory owned
    by an individual model.
    """

    def __init__(self, model: nn.Module):
        if _traceml_disabled() or not should_record_trace_events():
            return  # BYPASS: Do not attach any trackers

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
        if _traceml_disabled() or not should_record_trace_events():
            return

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def record(self):
        """
        Record peak memory at step end.

        Semantics:
        - CUDA: record real peak allocated / reserved memory
        - Non-CUDA: record `None` for both fields so downstream samplers can
          treat step memory as not applicable instead of a real zero-valued
          measurement
        """
        if _traceml_disabled() or not should_record_trace_events():
            return

        if self.device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
        else:
            peak_allocated = None
            peak_reserved = None

        record_step_memory_event(
            StepMemoryEvent(
                device=str(self.device),
                peak_allocated=(
                    float(peak_allocated)
                    if peak_allocated is not None
                    else None
                ),
                peak_reserved=(
                    float(peak_reserved) if peak_reserved is not None else None
                ),
                step=-1,  # assigned when the capture completes
                # Capture occurrence time here; queue draining can happen later.
                timestamp=time.time(),
            )
        )
