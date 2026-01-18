from dataclasses import dataclass
from collections import deque
from queue import Queue, Full
from typing import Dict, Optional
import time
import sys

import torch
import torch.nn as nn
from traceml.utils.shared_utils import model_is_on_cuda
from traceml.utils.cuda_event_pool import get_cuda_event, return_cuda_event
from traceml.utils.shared_utils import get_hookable_modules

# Shared queue (consumer-facing)
layer_forward_time_queue: Queue = Queue(maxsize=4096)

# Prevent double hook attachment
_layer_forward_time_hook_registry: Dict[int, bool] = {}

# Temporary pre-hook buffer: model_id -> layer -> FIFO start records
_layer_forward_time_start_buffer: Dict[int, Dict[str, deque]] = {}

# Main FIFO event buffer: model_id -> deque[]
_layer_forward_time_event_buffer: Dict[int, deque] = {}


@dataclass
class LayerForwardTimeEvent:
    """
    Time event for a single forward pass of a layer.
    """

    step: int
    model_id: int
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
        Attempt to resolve GPU timings, non-blocking.
        Returns:
            bool: True if fully resolved (CPU-only or GPU completed).
        """
        if self.resolved:
            return True
        # On CPU only
        if not self.on_gpu:
            self.resolved = True
            return True

        # On GPU (non-blocking)
        if self.gpu_end.query():
            self.gpu_duration_ms = self.gpu_start.elapsed_time(self.gpu_end)

            return_cuda_event(self.gpu_start)
            return_cuda_event(self.gpu_end)

            # Release CUDA event handles
            self.gpu_start = None
            self.gpu_end = None
            self.resolved = True
        return self.resolved


def get_layer_forward_time_queue() -> Queue:
    return layer_forward_time_queue


class LayerForwardTimePreHook:
    def __init__(self, model_id: int, layer_name: str, on_gpu: bool):
        self.model_id = model_id
        self.layer_name = layer_name
        self.on_gpu = on_gpu

    def __call__(self, module, inputs):
        try:
            cpu_start = time.perf_counter()
            gpu_start = None

            if self.on_gpu:
                gpu_start = get_cuda_event()
                gpu_start.record()

            model_buf = _layer_forward_time_start_buffer.setdefault(self.model_id, {})
            layer_q = model_buf.setdefault(self.layer_name, deque())
            layer_q.append(
                {
                    "cpu_start": cpu_start,
                    "gpu_start": gpu_start,
                }
            )
        except Exception:
            print(
                f"[TraceML] Error in LayerForwardTimePreHook for layer {self.layer_name}",
                file=sys.stderr,
            )


class LayerForwardTimePostHook:
    def __init__(self, model_id: int, layer_name: str, on_gpu: bool):
        self.model_id = model_id
        self.layer_name = layer_name
        self.on_gpu = on_gpu

    def __call__(self, module, inputs, output):
        try:
            cpu_end = time.perf_counter()

            layer_q = _layer_forward_time_start_buffer.get(self.model_id, {}).get(
                self.layer_name
            )
            if not layer_q:
                return  # No start recorded

            start_record = layer_q.popleft()  # FIFO match
            cpu_start = start_record["cpu_start"]
            cpu_duration_ms = (cpu_end - cpu_start) * 1000

            gpu_start = start_record["gpu_start"]
            gpu_end = None
            if self.on_gpu:
                gpu_end = get_cuda_event()
                gpu_end.record()

            event = LayerForwardTimeEvent(
                model_id=self.model_id,
                layer_name=self.layer_name,
                on_gpu=self.on_gpu,
                cpu_start=cpu_start,
                cpu_end=cpu_end,
                cpu_duration_ms=cpu_duration_ms,
                gpu_start=gpu_start,
                gpu_end=gpu_end,
                step=-1,
            )
            _layer_forward_time_event_buffer.setdefault(self.model_id, deque()).append(
                event
            )

        except Exception:
            print(
                f"[TraceML] Error in ActivationTimePostHook for layer {self.layer_name}",
                file=sys.stderr,
            )


def flush_layer_forward_time_buffers(model: nn.Module, step: int) -> None:
    """
    Drain the forward-time buffer for `model` and enqueue it as a NEW deque.

    - Preserves order
    - Producer buffer is fully emptied
    - Consumer receives independent ownership
    - No GPU resolve here
    """
    model_id = id(model)
    src = _layer_forward_time_event_buffer.get(model_id, None)
    if not src:
        return

    # Create a new deque for the consumer
    dst = deque()
    while src:
        event = src.popleft()
        event.step = step  ## Step is updated to correct value here
        dst.append(event)

    # Remove empty buffer entry
    _layer_forward_time_event_buffer.pop(model_id, None)

    try:
        layer_forward_time_queue.put_nowait(dst)
    except Full:
        pass


def attach_layer_forward_time_hooks(
    model: nn.Module,
    include_names=None, 
    exclude_names=None, 
    leaf_only=True
):
    """
    Attach pre and post hooks for timing.
    """

    model_id = id(model)
    if _layer_forward_time_hook_registry.get(model_id):
        return

    on_gpu = model_is_on_cuda(model)
    for name, module in get_hookable_modules(model, include_names, exclude_names, leaf_only):
        
        module.register_forward_pre_hook(
            LayerForwardTimePreHook(model_id, name, on_gpu=on_gpu)
        )
        module.register_forward_hook(
            LayerForwardTimePostHook(model_id, name, on_gpu=on_gpu)
        )

    _layer_forward_time_hook_registry[model_id] = True
