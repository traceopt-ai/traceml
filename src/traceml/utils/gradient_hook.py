import sys
import time
from dataclasses import dataclass
from queue import Queue, Full
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# Shared queue for gradient events
gradient_queue: Queue = Queue(maxsize=2048)

# Registries to prevent multiple hook attachments per model
_grad_module_registry: Dict[int, bool] = {}
_grad_param_registry: Dict[int, bool] = {}


@dataclass
class GradientEvent:
    """
    Represents gradient memory for a single layer / param batch during backprop.

    For module hooks:
      - per_device_grad_memory sums sizes over grad_output tensors of that layer.
      - per_layer contains {layer_name: {device: mb}}.

    For parameter hooks:
      - layer_name is the parameter's qualified name.
      - per_param contains {param_name: {device: mb}}.
    """

    model_id: int
    timestamp: float
    layer_name: Optional[str]
    per_device_grad_memory: Dict[str, float]
    per_layer: Dict[str, Dict[str, float]]
    per_param: Dict[str, Dict[str, float]]


def get_gradient_queue() -> Queue:
    """Return the shared queue of gradient events."""
    return gradient_queue


def _tensor_size_mb(t: torch.Tensor) -> float:
    try:
        return float(t.numel() * t.element_size()) / (1024**2)
    except Exception:
        return 0.0


def _accumulate_tensor_sizes_mb(obj: Any, out: Dict[str, float]) -> None:
    """
    Accumulate tensor sizes by device into `out`.
    Accepts a Tensor or nested structures (list/tuple/dict) of Tensors.
    """
    try:
        if isinstance(obj, torch.Tensor):
            dev = str(obj.device)
            out[dev] = out.get(dev, 0.0) + _tensor_size_mb(obj)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                _accumulate_tensor_sizes_mb(x, out)
        elif isinstance(obj, dict):
            for x in obj.values():
                _accumulate_tensor_sizes_mb(x, out)
    except Exception:
        # continue; don't want hooks to break training
        pass


class ModuleGradientHook:
    """
    Full backward hook capturing gradient sizes from grad_output for a layer.
    """

    def __init__(self, model_id: int, layer_name: str):
        self.model_id = model_id
        self.layer_name = layer_name

    def __call__(self, module: nn.Module, grad_input: Any, grad_output: Any):
        try:
            device_mb: Dict[str, float] = {}
            _accumulate_tensor_sizes_mb(grad_output, device_mb)

            ev = GradientEvent(
                model_id=self.model_id,
                timestamp=time.time(),
                layer_name=self.layer_name,
                per_device_grad_memory=device_mb.copy(),
                per_layer={self.layer_name: device_mb.copy()},
                per_param={},
            )
            try:
                gradient_queue.put_nowait(ev)
            except Full:
                pass
        except Exception:
            print(
                f"[TraceML] Error in ModuleGradientHook for layer {self.layer_name}",
                file=sys.stderr,
            )


class ParamGradientHook:
    """
    Per-parameter grad hook (registered on Tensor) to capture grad sizes for that param.
    """

    def __init__(self, model_id: int, param_name: str):
        self.model_id = model_id
        self.param_name = param_name

    def __call__(self, grad: torch.Tensor):
        try:
            device_mb: Dict[str, float] = {}
            _accumulate_tensor_sizes_mb(grad, device_mb)

            ev = GradientEvent(
                model_id=self.model_id,
                timestamp=time.time(),
                layer_name=None,
                per_device_grad_memory=device_mb.copy(),
                per_layer={},  # not populated by param hook
                per_param={self.param_name: device_mb.copy()},
            )
            try:
                gradient_queue.put_nowait(ev)
            except Full:
                pass
        except Exception:
            print(
                f"[TraceML] Error in ParamGradientHook for param {self.param_name}",
                file=sys.stderr,
            )


def attach_module_gradient_hooks(model: nn.Module) -> None:
    """
    Attach `register_full_backward_hook` to all leaf modules to capture grad_output sizes.
    Idempotent per model object.
    """
    model_id = id(model)
    if _grad_module_registry.get(model_id):
        return

    try:
        for name, module in model.named_modules():
            # leaf only
            if any(module.children()):
                continue
            # full backward hook works on module outputs
            module.register_full_backward_hook(ModuleGradientHook(model_id, name))
        _grad_module_registry[model_id] = True
    except Exception as e:
        print(f"[TraceML] Failed to attach module gradient hooks: {e}", file=sys.stderr)


def attach_param_gradient_hooks(model: nn.Module) -> None:
    """
    Attach per-parameter grad hooks to capture grad tensor sizes of parameters.
    Idempotent per model object.
    """
    model_id = id(model)
    if _grad_param_registry.get(model_id):
        return

    try:
        for name, p in model.named_parameters(recurse=True):
            if p.requires_grad:
                # register_hook attaches to the Tensor that will receive .grad
                p.register_hook(ParamGradientHook(model_id, name))
        _grad_param_registry[model_id] = True
    except Exception as e:
        print(f"[TraceML] Failed to attach param gradient hooks: {e}", file=sys.stderr)


def attach_all_gradient_hooks(model: nn.Module) -> None:
    """
    Convenience: attach both module backward hooks and per-parameter hooks.
    """
    attach_module_gradient_hooks(model)
    attach_param_gradient_hooks(model)
