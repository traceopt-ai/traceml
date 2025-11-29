from dataclasses import dataclass
from queue import Queue, Full
from typing import Dict, List
import time
import sys

import torch.nn as nn

activation_time_queue: Queue = Queue(maxsize=2048)
_activation_time_registry: Dict[int, bool] = {}

# Temporary buffer to match pre <-> post events
_temp_time_buffer: Dict[int, Dict[str, List[float]]] = {}
# structure:
# _temp_time_buffer[model_id][layer_name] = [list of start_times]


@dataclass
class ActivationTimeEvent:
    """
    Time event for a single forward pass of a layer.
    """

    model_id: int
    layer_name: str
    start_time: float
    end_time: float
    duration_ms: float


def get_activation_time_queue() -> Queue:
    return activation_time_queue


class ActivationTimePreHook:
    def __init__(self, model_id: int, layer_name: str):
        self.model_id = model_id
        self.layer_name = layer_name

    def __call__(self, module, inputs):
        try:
            now = time.perf_counter()
            if self.model_id not in _temp_time_buffer:
                _temp_time_buffer[self.model_id] = {}
            _temp_time_buffer[self.model_id].setdefault(self.layer_name, []).append(now)

        except Exception:
            print(
                f"[TraceML] Error in ActivationTimePreHook for layer {self.layer_name}",
                file=sys.stderr,
            )


class ActivationTimePostHook:
    def __init__(self, model_id: int, layer_name: str):
        self.model_id = model_id
        self.layer_name = layer_name

    def __call__(self, module, inputs, output):
        try:
            now = time.perf_counter()
            # Retrieve earliest start time
            start_list = _temp_time_buffer.get(self.model_id, {}).get(
                self.layer_name, []
            )

            if not start_list:
                return  # No start recorded

            start_time = start_list.pop(0)  # FIFO match
            duration_ms = now - start_time

            event = ActivationTimeEvent(
                model_id=self.model_id,
                layer_name=self.layer_name,
                start_time=start_time,
                end_time=now,
                duration_ms=duration_ms,
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

    for name, module in model.named_modules():
        if any(module.children()):
            continue

        module.register_forward_pre_hook(ActivationTimePreHook(model_id, name))
        module.register_forward_hook(ActivationTimePostHook(model_id, name))

    _activation_time_registry[model_id] = True
