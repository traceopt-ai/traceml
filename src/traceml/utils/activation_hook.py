from dataclasses import dataclass
from queue import Queue, Full
from typing import Dict, Any
import sys
import time

import torch
import torch.nn as nn


activation_queue: Queue = Queue(maxsize=2048)

# Registry to prevent multiple hook attachments per model
_activation_hook_registry: Dict[int, bool] = {}


@dataclass
class ActivationEvent:
    """
    Represents a single forward-pass activation snapshot for a model layer.
    """

    model_id: int
    timestamp: float
    per_device_activation_memory: Dict[str, float]
    per_layer: Dict[str, Dict[str, float]]


def get_activation_queue() -> Queue:
    """Return the shared queue of activation events."""
    return activation_queue




def _tensor_size_mb(tensor: torch.Tensor) -> float:
    """
    Compute the memory footprint of a tensor in megabytes.
    """
    return float(tensor.numel() * tensor.element_size()) / (1024**2)


class ActivationHook:
    """
    Callable class used as a forward hook to capture activation sizes for a layer.
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
                    size_mb = _tensor_size_mb(t)
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

            # Create and enqueue event
            event = ActivationEvent(
                model_id=self.model_id,
                timestamp=time.time(),
                per_device_activation_memory=layer_acc.copy(),
                per_layer={self.layer_name: layer_acc.copy()},
            )
            try:
                activation_queue.put_nowait(event)
            except Full:
                pass  # drop if queue is full
        except Exception:
            print(
                f"[TraceML] Error in ActivationHook for layer {self.layer_name}",
                file=sys.stderr,
            )


def attach_activation_hooks(model: nn.Module):
    """
    Attach a class-based forward hook to all leaf modules of `model`.
    Hooks are idempotent: repeated calls do nothing.

    Args:
        model (nn.Module): PyTorch model to instrument.
    """
    model_id = id(model)
    if _activation_hook_registry.get(model_id):
        # Hooks already attached
        return

    # Register ActivationHook on all leaf modules
    for name, module in model.named_modules():
        if any(module.children()):  # skip non-leaf modules
            continue
        module.register_forward_hook(ActivationHook(model_id, name))

    _activation_hook_registry[model_id] = True

