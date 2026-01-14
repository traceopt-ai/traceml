from dataclasses import dataclass
from queue import Queue, Full
from typing import Dict, Any, List, Tuple
import sys
from traceml.utils.shared_utils import get_hookable_modules
import torch
import torch.nn as nn


# Shared queue for forward events
layer_forward_memory_queue: Queue = Queue(maxsize=4096)

# Registry to prevent multiple hook attachments per model
_layer_forward_memory_hook_registry: Dict[int, bool] = {}

# In-memory buffer: model_id -> List[(layer_name, memory_per_device, timestamp)
_layer_forward_memory_buffer: Dict[int, List] = {}


@dataclass
class LayerForwardMemoryEvents:
    """
    Represents a single forward-pass snapshot for a model layer.
    """

    model_id: int
    layers: List[Tuple[str, Dict[str, float]]]
    step: int


def get_layer_forward_memory_queue() -> Queue:
    """Return the shared queue of activation events."""
    return layer_forward_memory_queue


def _tensor_size(tensor: torch.Tensor) -> float:
    """
    Compute the memory footprint of a tensor in megabytes.
    """
    return float(tensor.numel() * tensor.element_size())


class LayerForwardMemoryHook:
    """
    Callable class used as a forward hook to capture forward tensor sizes for a layer.
    """

    def __init__(self, model_id: int, layer_name: str):
        self.model_id = model_id
        self.layer_name = layer_name

    def __call__(self, module: nn.Module, inputs: Any, output: Any):
        try:
            layer_acc: Dict[str, float] = {}

            def accumulate(t):
                if isinstance(t, torch.Tensor):
                    device_str = str(t.device)
                    size_mb = _tensor_size(t)
                    layer_acc[device_str] = layer_acc.get(device_str, 0.0) + size_mb

            # Handle tensor or collection of tensors
            if isinstance(output, torch.Tensor):
                accumulate(output)
            elif isinstance(output, (list, tuple)):
                for o in output:
                    accumulate(o)
            elif isinstance(output, dict):
                for o in output.values():
                    accumulate(o)

            _layer_forward_memory_buffer.setdefault(self.model_id, []).append(
                (self.layer_name, layer_acc)
            )

        except Exception:
            print(
                f"[TraceML] Error in LayerForwardMemoryHook for layer {self.layer_name}",
                file=sys.stderr,
            )


def flush_layer_forward_memory_buffers(model: nn.Module, step: int):
    """
    Convert buffered activation memory into events and enqueue them.
    Called before optimizer.step().
    """
    model_id = id(model)
    buf = _layer_forward_memory_buffer.pop(model_id, None)

    if not buf:
        return

    event = LayerForwardMemoryEvents(
        model_id=model_id,
        layers=buf,
        step=step,
    )
    try:
        layer_forward_memory_queue.put_nowait(event)
    except Full:
        pass


def attach_layer_forward_memory_hooks(
    model: nn.Module,
    include_names=None, 
    exclude_names=None, 
    leaf_only=True
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
    for name, module in get_hookable_modules(model, include_names, exclude_names, leaf_only):
        module.register_forward_hook(LayerForwardMemoryHook(model_id, name))

    _layer_forward_memory_hook_registry[model_id] = True
