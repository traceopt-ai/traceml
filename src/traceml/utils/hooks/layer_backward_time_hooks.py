"""
Layer backward timing instrumentation for TraceML.

This module captures *per-layer backward execution time* using PyTorch
full backward pre/post hooks.

Architecture
------------
- Hooks emit low-level timing events (per backward invocation).
- Events are buffered during the step.
- At step boundaries, all events are flushed together as a step snapshot.
- Aggregation (per-layer summation) is intentionally deferred to the sampler.

Design Principles
-----------------
- No GPU synchronization
- Correct handling of shared / re-entered modules
- Step semantics handled outside hooks
- Single-device assumption (V1)
"""

import sys
import time
from collections import deque
from dataclasses import dataclass
from queue import Full, Queue
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from traceml.utils.cuda_event_pool import get_cuda_event, return_cuda_event
from traceml.utils.shared_utils import model_is_on_cuda

# Shared queue for backward timing events
layer_backward_time_queue: Queue = Queue(maxsize=4906)


def get_layer_backward_time_queue() -> Queue:
    return layer_backward_time_queue


# Prevent double hook attachment
_backward_time_hook_registry: Dict[int, bool] = {}

# Pre-hook FIFO buffer:
#   model_id -> layer_name -> deque[start_records]
_layer_backward_time_start_buffer: Dict[int, Dict[str, deque]] = {}

# Post-hook event buffer:
#   model_id -> List[LayerBackwardTimeEvent]
_layer_backward_time_event_buffer: Dict[
    int, List["LayerBackwardTimeEvent"]
] = {}


@dataclass
class LayerBackwardTimeEvent:
    """
    Timing event for a *single backward invocation* of a layer.

    This is an internal, transient object:
    - Created by hooks
    - Paired via FIFO
    - Resolved asynchronously
    - Never written directly to the database
    """

    layer_name: str
    on_gpu: bool

    cpu_start: float
    cpu_end: float
    cpu_duration_ms: float

    gpu_start: Optional[torch.cuda.Event]
    gpu_end: Optional[torch.cuda.Event]

    gpu_duration_ms: Optional[float] = None
    resolved: bool = False

    def try_resolve(self) -> bool:
        """
        Attempt to resolve GPU timing without blocking.
        """
        if self.resolved:
            return True

        if not self.on_gpu:
            self.resolved = True
            return True

        if self.gpu_end.query():
            self.gpu_duration_ms = self.gpu_start.elapsed_time(self.gpu_end)

            return_cuda_event(self.gpu_start)
            return_cuda_event(self.gpu_end)

            self.gpu_start = None
            self.gpu_end = None
            self.resolved = True

        return self.resolved


@dataclass
class LayerBackwardTimeStepEvent:
    """
    Backward timing snapshot for a *single training step*.
    """

    model_id: int
    step: int
    device: str
    layers: List[LayerBackwardTimeEvent]


class LayerBackwardTimePreHook:
    """
    Full backward *pre* hook.

    Signature:
        (module, grad_output) -> None
    """

    def __init__(self, model_id: int, layer_name: str, on_gpu: bool):
        self.model_id = model_id
        self.layer_name = layer_name
        self.on_gpu = on_gpu

    def __call__(self, module, grad_output):
        try:
            cpu_start = time.perf_counter()
            gpu_start = None

            if self.on_gpu:
                gpu_start = get_cuda_event()
                gpu_start.record()

            model_buf = _layer_backward_time_start_buffer.setdefault(
                self.model_id, {}
            )
            layer_q = model_buf.setdefault(self.layer_name, deque())
            layer_q.append(
                {
                    "cpu_start": cpu_start,
                    "gpu_start": gpu_start,
                }
            )

        except Exception:
            print(
                f"[TraceML] Error in LayerBackwardTimePreHook ({self.layer_name})",
                file=sys.stderr,
            )


class LayerBackwardTimePostHook:
    """
    Full backward *post* hook.

    Signature:
        (module, grad_input, grad_output) -> None
    """

    def __init__(self, model_id: int, layer_name: str, on_gpu: bool):
        self.model_id = model_id
        self.layer_name = layer_name
        self.on_gpu = on_gpu

    def __call__(self, module, grad_input, grad_output):
        try:
            cpu_end = time.perf_counter()

            layer_q = _layer_backward_time_start_buffer.get(
                self.model_id, {}
            ).get(self.layer_name)
            if not layer_q:
                return

            start = layer_q.popleft()
            cpu_start = start["cpu_start"]
            cpu_duration_ms = (cpu_end - cpu_start) * 1000.0

            gpu_start = start["gpu_start"]
            gpu_end = None
            if self.on_gpu:
                gpu_end = get_cuda_event()
                gpu_end.record()

            event = LayerBackwardTimeEvent(
                layer_name=self.layer_name,
                on_gpu=self.on_gpu,
                cpu_start=cpu_start,
                cpu_end=cpu_end,
                cpu_duration_ms=cpu_duration_ms,
                gpu_start=gpu_start,
                gpu_end=gpu_end,
            )

            _layer_backward_time_event_buffer.setdefault(
                self.model_id, []
            ).append(event)

        except Exception:
            print(
                f"[TraceML] Error in LayerBackwardTimePostHook ({self.layer_name})",
                file=sys.stderr,
            )


def flush_layer_backward_time_buffers(model: nn.Module, step: int) -> None:
    """
    Flush all backward timing events for `model` at a step boundary.
    Emits a single LayerBackwardTimeStepEvent into the shared queue.
    """
    model_id = id(model)
    layers = _layer_backward_time_event_buffer.pop(model_id, None)
    if not layers:
        return

    try:
        device = str(next(model.parameters()).device)
    except StopIteration:
        device = "unknown"

    event = LayerBackwardTimeStepEvent(
        model_id=model_id,
        step=step,
        device=device,
        layers=layers,
    )

    try:
        layer_backward_time_queue.put_nowait(event)
    except Full:
        pass


def attach_layer_backward_time_hooks(model: nn.Module):
    """
    Attach backward pre/post hooks for backward timing.
    """
    model_id = id(model)
    if _backward_time_hook_registry.get(model_id):
        return

    on_gpu = model_is_on_cuda(model)

    for name, module in model.named_modules():
        if any(module.children()):
            continue  # leaf-only

        module.register_full_backward_pre_hook(
            LayerBackwardTimePreHook(model_id, name, on_gpu=on_gpu)
        )
        module.register_full_backward_hook(
            LayerBackwardTimePostHook(model_id, name, on_gpu=on_gpu)
        )

    _backward_time_hook_registry[model_id] = True
