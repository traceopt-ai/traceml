from dataclasses import dataclass
from queue import Queue, Full
from typing import Dict, List, Optional
import time
import sys

import torch
import torch.nn as nn
from traceml.utils.shared_utils import model_is_on_cuda

activation_time_queue: Queue = Queue(maxsize=2048)
_activation_time_registry: Dict[int, bool] = {}

# Temporary buffer to match pre <-> post events
_temp_time_buffer: Dict[int, Dict[str, List[dict]]] = {}


@dataclass
class ActivationTimeEvent:
    """
    Time event for a single forward pass of a layer.
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

            # Release CUDA event handles
            self.gpu_start = None
            self.gpu_end = None
            self.resolved = True
        return self.resolved


def get_activation_time_queue() -> Queue:
    return activation_time_queue


class ActivationTimePreHook:
    def __init__(self, model_id: int, layer_name: str, on_gpu:bool):
        self.model_id = model_id
        self.layer_name = layer_name
        self.on_gpu = on_gpu

    def __call__(self, module, inputs):
        try:
            cpu_start = time.perf_counter()
            gpu_start = None

            if self.on_gpu:
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_start.record()

            buf = _temp_time_buffer.setdefault(self.model_id, {})
            buf.setdefault(self.layer_name, []).append({
                    "cpu_start": cpu_start,
                    "gpu_start": gpu_start
                }
            )
        except Exception:
            print(
                f"[TraceML] Error in ActivationTimePreHook for layer {self.layer_name}",
                file=sys.stderr,
            )


class ActivationTimePostHook:
    def __init__(self, model_id: int, layer_name: str, on_gpu:bool):
        self.model_id = model_id
        self.layer_name = layer_name
        self.on_gpu = on_gpu

    def __call__(self, module, inputs, output):
        try:
            cpu_end = time.perf_counter()

            start_list = _temp_time_buffer.get(self.model_id, {}).get(
                self.layer_name, []
            )

            if not start_list:
                return  # No start recorded

            start_record = start_list.pop(0)  # FIFO match
            cpu_start = start_record["cpu_start"]
            cpu_duration_ms = (cpu_end - cpu_start)*1000

            gpu_start = start_record["gpu_start"]
            gpu_end = None
            if self.on_gpu:
                gpu_end = torch.cuda.Event(enable_timing=True)
                gpu_end.record()

            event = ActivationTimeEvent(
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
                activation_time_queue.put_nowait(event)
            except Full:
                pass

        except Exception:
            print(
                f"[TraceML] Error in ActivationTimePostHook for layer {self.layer_name}",
                file=sys.stderr,
            )


def attach_activation_time_hooks(model: nn.Module):
    """
    Attach pre and post hooks for timing.
    """

    model_id = id(model)
    if _activation_time_registry.get(model_id):
        return

    on_gpu = model_is_on_cuda(model)
    for name, module in model.named_modules():
        if any(module.children()):
            continue

        module.register_forward_pre_hook(
            ActivationTimePreHook(model_id, name, on_gpu=on_gpu))
        module.register_forward_hook(
            ActivationTimePostHook(model_id, name, on_gpu=on_gpu))

    _activation_time_registry[model_id] = True
