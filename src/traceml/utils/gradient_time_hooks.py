from dataclasses import dataclass
from queue import Queue, Full
from typing import Dict, List, Optional
import time
import sys
import torch
import torch.nn as nn
from traceml.utils.shared_utils import model_is_on_cuda


# Shared queue for gradient timing events
gradient_time_queue: Queue = Queue(maxsize=2048)
_gradient_time_registry: Dict[int, bool] = {}

# Temporary buffer to match backward pre <-> backward post events
_temp_grad_time_buffer: Dict[int, Dict[str, List[dict]]] = {}


@dataclass
class GradientTimeEvent:
    """
    Time event for a single backward pass of a layer.
    """
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

        if self.gpu_start is None or self.gpu_end is None:
            # Shouldn't happen on GPU path, but keep it safe
            self.resolved = True
            return True

        # Non-blocking readiness check
        if self.gpu_end.query():
            self.gpu_duration_ms = self.gpu_start.elapsed_time(self.gpu_end)
            # release CUDA event handles
            self.gpu_start = None
            self.gpu_end = None
            self.resolved = True

        return self.resolved


def get_gradient_time_queue() -> Queue:
    return gradient_time_queue


class GradientTimePreHook:
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
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_start.record()

            buf = _temp_grad_time_buffer.setdefault(self.model_id, {})
            buf.setdefault(self.layer_name, []).append(
                {"cpu_start": cpu_start, "gpu_start": gpu_start}
            )
        except Exception:
            print(
                f"[TraceML] Error in GradientTimePreHook for layer {self.layer_name}",
                file=sys.stderr,
            )


class GradientTimePostHook:
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

            start_list = _temp_grad_time_buffer.get(self.model_id, {}).get(self.layer_name, [])
            if not start_list:
                return

            start_record = start_list.pop(0)  # FIFO
            cpu_start = start_record["cpu_start"]
            cpu_duration_ms = (cpu_end - cpu_start) * 1000.0

            gpu_start = start_record.get("gpu_start")
            gpu_end = None

            if self.on_gpu:
                gpu_end = torch.cuda.Event(enable_timing=True)
                gpu_end.record()  # stream-ordered end marker

            event = GradientTimeEvent(
                model_id=self.model_id,
                layer_name=self.layer_name,
                on_gpu=self.on_gpu,
                cpu_start=cpu_start,
                cpu_end=cpu_end,
                cpu_duration_ms=cpu_duration_ms,
                gpu_start=gpu_start,
                gpu_end=gpu_end,
            )
            try:
                gradient_time_queue.put_nowait(event)
            except Full:
                pass

        except Exception:
            print(
                f"[TraceML] Error in GradientTimePostHook for layer {self.layer_name}",
                file=sys.stderr,
            )


def attach_gradient_time_hooks(model: nn.Module):
    """
    Attach backward pre/post hooks for gradient timing.

    Note:
      - Hooks only fire for modules that actually participate in backward
        (requires_grad path).
      - Uses full backward hooks (recommended).
    """
    model_id = id(model)
    if _gradient_time_registry.get(model_id):
        return

    on_gpu = model_is_on_cuda(model)

    for name, module in model.named_modules():
        if any(module.children()):
            continue  # leaf-only like your activation timers

        module.register_full_backward_pre_hook(
            GradientTimePreHook(model_id, name, on_gpu=on_gpu)
        )
        module.register_full_backward_hook(
            GradientTimePostHook(model_id, name, on_gpu=on_gpu)
        )

    _gradient_time_registry[model_id] = True