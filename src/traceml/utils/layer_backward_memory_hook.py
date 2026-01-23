"""
Layer backward (gradient) activation memory instrumentation for TraceML.

This module captures *backward-pass gradient memory* at a per-layer level.
It uses PyTorch full backward hooks to record the size of gradient tensors
produced during backpropagation for each leaf module.

Design principles
-----------------
- Backward hooks are attached once per model (idempotent).
- Memory is accumulated per layer per step.
- Data is buffered during backward execution and flushed explicitly
  at step boundaries.
- Only scalar memory values are tracked (single-device assumption, V1).
- No live tensors or modules escape this module.
"""

from dataclasses import dataclass
from queue import Queue, Full
from typing import Any, Dict, Tuple, List
import sys

import torch
import torch.nn as nn

# Shared queue for gradient events
layer_backward_memory_queue: Queue = Queue(maxsize=2048)

# Registries to prevent multiple hook attachments per model
_layer_backward_hook_registry: Dict[int, bool] = {}

# In-memory buffer:
#   model_id -> List[(layer_name, gradient_memory_bytes)]
_layer_backward_memory_buffer: Dict[int, List[Tuple[str, float]]] = {}


@dataclass
class LayerBackwardMemoryEvents:
    """
    Represents a single backward-pass gradient memory snapshot.

    Attributes
    ----------
    model_id : int
        Identity of the model instance (id(model)).
    layers : List[Tuple[str, float]]
        List of (layer_name, gradient_memory_bytes).
    device : str
        Device on which the backward pass executed (e.g. "cuda:0", "cpu").
    step : int
        Training step index.
    """

    model_id: int
    layers: List[Tuple[str, float]]
    device: str
    step: int


def get_layer_backward_queue() -> Queue:
    """
    Return the shared queue containing flushed backward memory events.

    This queue is drained by the corresponding sampler.
    """
    return layer_backward_memory_queue


def _tensor_size(t: torch.Tensor) -> float:
    """
    Compute the memory footprint of a tensor in bytes.

    Parameters
    ----------
    tensor : torch.Tensor

    Returns
    -------
    float
        Tensor size in bytes.
    """
    try:
        return float(t.numel() * t.element_size())
    except Exception:
        return 0.0


def _accumulate_tensor_bytes(obj: Any) -> float:
    """
    Accumulate tensor memory from a tensor or shallow container of tensors.

    Supported types:
      - torch.Tensor
      - list / tuple of tensors
      - dict of tensors

    Returns
    -------
    float
        Total memory in bytes.
    """
    total = 0.0
    if obj is None:
        return total

    if isinstance(obj, torch.Tensor):
        return _tensor_size(obj)

    if isinstance(obj, (list, tuple)):
        for x in obj:
            total += _accumulate_tensor_bytes(x)
        return total

    if isinstance(obj, dict):
        for x in obj.values():
            total += _accumulate_tensor_bytes(x)
        return total

    return total


class LayerBackwardModuleHook:
    """
    Full backward hook that records gradient memory for a single layer.

    The hook accumulates the size of gradient tensors produced during
    backpropagation and stores the result in a per-model buffer.
    """

    def __init__(self, model_id: int, layer_name: str):
        self.model_id = model_id
        self.layer_name = layer_name

    def __call__(self, module: nn.Module, grad_input: Any, grad_output: Any):
        try:
            total_bytes = _accumulate_tensor_bytes(grad_output)

            if total_bytes > 0:
                _layer_backward_memory_buffer.setdefault(
                    self.model_id, []
                ).append((self.layer_name, total_bytes))

        except Exception:
            print(
                f"[TraceML] Error in LayerBackwardHook for layer {self.layer_name}",
                file=sys.stderr,
            )


def flush_layer_backward_memory_buffers(model: nn.Module, step: int) -> None:
    """
    Flush buffered backward (gradient) memory for a model into the queue.

    This function should be called at a step boundary
    (typically before optimizer.step()).

    Parameters
    ----------
    model : nn.Module
        Model whose backward buffers should be flushed.
    step : int
        Current training step.
    """
    model_id = id(model)
    buf = _layer_backward_memory_buffer.pop(model_id, None)

    if not buf:
        return

    # Resolve execution device
    try:
        device = str(next(model.parameters()).device)
    except StopIteration:
        device = "unknown"

    event = LayerBackwardMemoryEvents(
        model_id=model_id,
        layers=buf,
        device=device,
        step=step,
    )
    try:
        layer_backward_memory_queue.put_nowait(event)
    except Full:
        # Drop silently to avoid training backpressure
        pass


def attach_layer_backward_memory_hooks(model: nn.Module) -> None:
    """
    Attach backward hooks to all leaf modules of a model.

    Hooks are idempotent: calling this function multiple times for the same
    model instance has no effect.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to instrument.
    """
    model_id = id(model)
    if _layer_backward_hook_registry.get(model_id):
        return

    try:
        for name, module in model.named_modules():
            # Only attach to leaf modules
            if any(module.children()):
                continue
            # full backward hook works on module outputs
            module.register_full_backward_hook(LayerBackwardModuleHook(model_id, name))
        _layer_backward_hook_registry[model_id] = True
    except Exception as e:
        print(f"[TraceML] Failed to attach layer backward hooks: {e}", file=sys.stderr)
