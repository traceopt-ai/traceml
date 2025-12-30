from dataclasses import dataclass
from queue import Queue, Full
from typing import Dict, Optional
import sys

import torch
import torch.nn as nn

# Shared queue for model forward peak memory events
model_forward_memory_queue: Queue = Queue(maxsize=128)

# Registry to prevent multiple hook attachments per model
_model_forward_memory_hook_registry: Dict[int, bool] = {}

# In-memory buffer: model_id -> last event (optional)
_model_forward_memory_buffer: Dict = {}


def get_model_forward_memory_queue() -> Queue:
    return model_forward_memory_queue


@dataclass
class ModelForwardMemoryEvent:
    """
    Peak GPU memory during model forward pass.
    """

    step: int
    model_id: int
    device: str
    peak_allocated_mb: float
    peak_reserved_mb: float


class ModelForwardMemoryPreHook:
    """
    Reset CUDA peak memory before model forward.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, module: nn.Module, inputs):
        try:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
        except Exception:
            print("[TraceML] Error in ModelForwardMemoryPreHook", file=sys.stderr)


class ModelForwardMemoryPostHook:
    """
    Record CUDA peak memory after model forward.
    """

    def __init__(self, model_id: int, device: torch.device):
        self.model_id = model_id
        self.device = device

    def __call__(self, module: nn.Module, inputs, output):
        try:
            if self.device.type != "cuda":
                return

            evt = ModelForwardMemoryEvent(
                model_id=self.model_id,
                device=str(self.device),
                peak_allocated_mb=torch.cuda.max_memory_allocated(self.device),
                peak_reserved_mb=torch.cuda.max_memory_reserved(self.device),
                step=-1,
            )
            _model_forward_memory_buffer[self.model_id] = evt

        except Exception:
            print("[TraceML] Error in ModelForwardMemoryPostHook", file=sys.stderr)


def flush_model_forward_memory_buffers(model: nn.Module, step: int) -> None:
    model_id = id(model)
    evt: Optional[ModelForwardMemoryEvent] = _model_forward_memory_buffer.pop(
        model_id, None
    )
    if evt is None:
        return
    evt.step = step
    try:
        model_forward_memory_queue.put_nowait(evt)
    except Full:
        pass


def attach_model_forward_memory_hooks(model: nn.Module):
    """
    Attach model-level forward hooks to capture peak memory.
    """
    model_id = id(model)
    if _model_forward_memory_hook_registry.get(model_id):
        return

    try:
        device = next(model.parameters()).device
    except StopIteration:
        # Model without parameters (edge case)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.register_forward_pre_hook(ModelForwardMemoryPreHook(device))
    model.register_forward_hook(ModelForwardMemoryPostHook(model_id, device))

    _model_forward_memory_hook_registry[model_id] = True
