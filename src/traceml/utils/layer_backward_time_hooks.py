from dataclasses import dataclass
from queue import Queue, Full
from typing import Dict, Deque, Optional
from collections import deque
import time
import sys
import torch
import torch.nn as nn
from traceml.utils.shared_utils import model_is_on_cuda
from traceml.utils.cuda_event_pool import get_cuda_event, return_cuda_event


# Shared queue for backward timing events
layer_backward_time_queue: Queue = Queue(maxsize=2048)

# Prevent double hook attachment
_backward_time_hook_registry: Dict[int, bool] = {}

# Temporary buffer to match backward pre <-> post
# model_id -> layer_name -> deque[{cpu_start, gpu_start}]
_backward_time_start_buffer: Dict[int, Dict[str, Deque[dict]]] = {}

# Main in-memory FIFO buffer
# model_id -> deque
_backward_time_buffer: Dict[int, Deque] = {}



def get_layer_backward_time_queue() -> Queue:
    return layer_backward_time_queue



@dataclass
class LayerBackwardTimeEvent:
    """
    Time event for a single backward pass of a layer.
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
        if self.resolved:
            return True

        if not self.on_gpu:
            self.resolved = True
            return True

        # Non-blocking readiness check
        if self.gpu_end.query():
            self.gpu_duration_ms = self.gpu_start.elapsed_time(self.gpu_end)

            return_cuda_event(self.gpu_start)
            return_cuda_event(self.gpu_end)

            # release CUDA event handles
            self.gpu_start = None
            self.gpu_end = None
            self.resolved = True

        return self.resolved




class LayerBackwardTimePreHook:
    """
    Full backward *pre* hook: record start markers.
    Signature: (module, grad_output) -> None
    """
    def __init__(self, model_id: int, layer_name: str, on_gpu: bool):
        self.model_id = model_id
        self.layer_name = layer_name
        self.on_gpu = on_gpu

    def __call__(self, module, grad_output):
        try:
            cpu_start = time.perf_counter()
            gpu_start = None

            if self.on_gpu and torch.cuda.is_available():
                gpu_start = get_cuda_event()
                gpu_start.record()

            model_buf = _backward_time_start_buffer.setdefault(self.model_id, {})
            model_buf.setdefault(self.layer_name, deque()).append(
                {
                    "cpu_start": cpu_start,
                    "gpu_start": gpu_start,
                }
            )
        except Exception:
            print(
                f"[TraceML] Error in LayerBackwardTimePreHook for layer {self.layer_name}",
                file=sys.stderr,
            )


class LayerBackwardTimePostHook:
    """
    Full backward hook: record end markers and enqueue event.
    Signature: (module, grad_input, grad_output) -> None
    """
    def __init__(self, model_id: int, layer_name: str, on_gpu: bool):
        self.model_id = model_id
        self.layer_name = layer_name
        self.on_gpu = on_gpu

    def __call__(self, module, grad_input, grad_output):
        try:
            cpu_end = time.perf_counter()

            layer_q = (
                _backward_time_start_buffer
                .get(self.model_id, {})
                .get(self.layer_name)
            )
            if not layer_q:
                return

            start_rec = layer_q.popleft()  # FIFO match
            cpu_start = start_rec["cpu_start"]
            cpu_duration_ms = (cpu_end - cpu_start) * 1000.0

            gpu_start = start_rec["gpu_start"]
            gpu_end = None

            if self.on_gpu:
                gpu_end = get_cuda_event()
                gpu_end.record()

            event = LayerBackwardTimeEvent(
                model_id=self.model_id,
                layer_name=self.layer_name,
                on_gpu=self.on_gpu,
                cpu_start=cpu_start,
                cpu_end=cpu_end,
                cpu_duration_ms=cpu_duration_ms,
                gpu_start=gpu_start,
                gpu_end=gpu_end,
                step=-1
            )

            _backward_time_buffer.setdefault(self.model_id, deque()).append(event)

        except Exception:
            print(
                f"[TraceML] Error in GradientTimePostHook for layer {self.layer_name}",
                file=sys.stderr,
            )




def flush_layer_backward_time_buffers(model: nn.Module, step: int) -> None:
    """
    Drain the backward-time buffer for `model` and enqueue as a NEW deque.
    - Preserves FIFO order
    - Producer buffer is fully emptied
    - Consumer gets independent ownership
    """
    model_id = id(model)
    src = _backward_time_buffer.get(model_id, None)
    if not src:
        return

    dst: Deque = deque()

    while src:
        event = src.popleft()
        event.step = step       ## step is updated during flush
        dst.append(event)

    _backward_time_buffer.pop(model_id, None)

    try:
        layer_backward_time_queue.put_nowait(dst)
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