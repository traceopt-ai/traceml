"""
Layer forward activation memory instrumentation for TraceML.

This module captures *forward-pass activation memory* at a per-layer level.
It uses PyTorch forward hooks to record the size of tensors produced by
each leaf module during forward execution.

Design principles
-----------------
- Forward hooks are attached once per model (idempotent).
- Memory is accumulated per layer per step.
- Data is buffered during forward execution and flushed explicitly
  at step boundaries.
- Only scalar memory values are tracked (single-device assumption, V1).
- No live tensors or modules escape this module.
"""

import sys
from dataclasses import dataclass
from queue import Full, Queue
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from traceml.utils.shared_utils import get_hookable_modules

# Shared queue for forward events
layer_forward_memory_queue: Queue = Queue(maxsize=4096)

# Registry to prevent multiple hook attachments per model
_layer_forward_memory_hook_registry: Dict[int, bool] = {}

# In-memory buffer: model_id -> List[(layer_name, memory_per_device, timestamp)
_layer_forward_memory_buffer: Dict[int, List] = {}


@dataclass
class LayerForwardMemoryEvents:
    """
    Represents a single forward-pass activation memory snapshot.

    Attributes
    ----------
    model_id : int
        Identity of the model instance (id(model)).
    layers : List[Tuple[str, float]]
        List of (layer_name, activation_memory_bytes).
    device : str
        Device on which the forward pass executed (e.g. "cuda:0", "cpu").
    step : int
    """

    model_id: int
    layers: List[Tuple[str, float]]
    device: str
    step: int


def get_layer_forward_memory_queue() -> Queue:
    """
    Return the shared queue containing flushed forward memory events.

    This queue is drained by the corresponding sampler.
    """
    return layer_forward_memory_queue


def _tensor_size(tensor: torch.Tensor) -> float:
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
    return float(tensor.numel() * tensor.element_size())


class LayerForwardMemoryHook:
    """
    Forward hook that records activation memory for a single layer.

    The hook accumulates the size of all tensors produced by the layer
    during a forward pass and stores the result in a per-model buffer.
    """

    def __init__(self, model_id: int, layer_name: str):
        self.model_id = model_id
        self.layer_name = layer_name

    def __call__(self, module: nn.Module, inputs: Any, output: Any):
        try:
            total_bytes = 0.0

            def accumulate(obj: Any) -> None:
                nonlocal total_bytes
                if isinstance(obj, torch.Tensor):
                    total_bytes += _tensor_size(obj)

            # Handle tensor or collection of tensors
            if isinstance(output, torch.Tensor):
                accumulate(output)
            elif isinstance(output, (list, tuple)):
                for o in output:
                    accumulate(o)
            elif isinstance(output, dict):
                for o in output.values():
                    accumulate(o)

            if total_bytes > 0:
                _layer_forward_memory_buffer.setdefault(
                    self.model_id,
                    [],
                ).append(
                    (self.layer_name, total_bytes),
                )

        except Exception:
            print(
                f"[TraceML] Error in LayerForwardMemoryHook for layer {self.layer_name}",
                file=sys.stderr,
            )


def flush_layer_forward_memory_buffers(model: nn.Module, step: int) -> None:
    """
    Flush buffered forward activation memory for a model into the queue.

    This function should be called at a step boundary (e.g. before or after
    optimizer.step()).

    Parameters
    ----------
    model : nn.Module
        Model whose forward buffers should be flushed.
    step : int
        Current training step.
    """
    model_id = id(model)
    buf = _layer_forward_memory_buffer.pop(model_id, None)
    if not buf:
        return

    # Resolve execution device
    try:
        device = str(next(model.parameters()).device)
    except StopIteration:
        device = "unknown"

    event = LayerForwardMemoryEvents(
        model_id=model_id,
        layers=buf,
        device=device,
        step=step,
    )
    try:
        layer_forward_memory_queue.put_nowait(event)
    except Full:
        # Drop event silently to avoid backpressure on training
        pass


def attach_layer_forward_memory_hooks(
    model: nn.Module,
    include_names=None,
    exclude_names=None,
    leaf_only=True,
):
    """
    Attach forward hooks to specific modules of `model` based on filtering criteria.
    Hooks are idempotent: repeated calls for the same model instance do nothing.

    Args:
        model (nn.Module): PyTorch model to instrument.
        include_names (list, optional): Only hook modules containing these strings in their name.
        exclude_names (list, optional): Skip modules containing these strings.
        leaf_only (bool): If True, only considers leaf modules (default behavior).
    """
    model_id = id(model)
    if _layer_forward_memory_hook_registry.get(model_id):
        # Hooks already attached
        return

    # Register ActivationHook on all leaf modules
    for name, module in get_hookable_modules(
        model,
        include_names,
        exclude_names,
        leaf_only,
    ):
        module.register_forward_hook(LayerForwardMemoryHook(model_id, name))

    _layer_forward_memory_hook_registry[model_id] = True
