import sys
from dataclasses import dataclass
from queue import Queue, Full
from typing import Any, Dict, Optional

from traceml.loggers.error_log import get_error_logger, setup_error_logger

import torch
import torch.nn as nn

# Shared queue for gradient events
gradient_queue: Queue = Queue(maxsize=2048)

# Registries to prevent multiple hook attachments per model
_grad_layer_registry: Dict[int, bool] = {}
_grad_param_registry: Dict[int, bool] = {}


@dataclass
class GradientEvent:
    """
    Represents gradient memory for a single layer / param batch during backprop.

    For module hooks:
      - per_device_memory sizes over grad_output tensors of that layer.

    For parameter hooks:
      - layer_name is the parameter's qualified name.
    """

    model_id: int
    layer_name: str  # layer/module name
    param_name: Optional[str]
    per_device_memory: Dict[str, float]


def get_gradient_queue() -> Queue:
    """Return the shared queue of gradient events."""
    return gradient_queue


def _tensor_size_mb(t: torch.Tensor) -> float:
    try:
        return float(t.numel() * t.element_size())
    except Exception:
        return 0.0


def _accumulate_tensor_sizes_mb(obj: Any, out: Dict[str, float]) -> None:
    """
    Accumulate tensor sizes by device into `out`.
    Accepts a Tensor or nested structures (list/tuple/dict) of Tensors.
    """
    if obj is None:
        return

    if isinstance(obj, torch.Tensor):
        try:
            dev = str(obj.device)
            out[dev] = out.get(dev, 0.0) + _tensor_size_mb(obj)
        except Exception:
            pass
        return

    if isinstance(obj, (list, tuple)):
        for x in obj:
            _accumulate_tensor_sizes_mb(x, out)
        return

    if isinstance(obj, dict):
        for x in obj.values():
            _accumulate_tensor_sizes_mb(x, out)
        return


class LayerGradientHook:
    """
    Full backward hook capturing gradient sizes from grad_output for a layer.
    """

    def __init__(self, model_id: int, layer_name: str):
        self.model_id = model_id
        self.layer_name = layer_name
        setup_error_logger()
        self.logger = get_error_logger("LayerGradientHook")

    def __call__(self, module: nn.Module, grad_input: Any, grad_output: Any):
        try:
            device_mb: Dict[str, float] = {}
            _accumulate_tensor_sizes_mb(grad_output, device_mb)

            if device_mb:
                elem = GradientEvent(
                    model_id=self.model_id,
                    layer_name=self.layer_name,
                    param_name=None,
                    per_device_memory=device_mb.copy(),
                )
                try:
                    gradient_queue.put_nowait(elem)
                except Full:
                    pass
        except Exception:
            self.logger.error(
                f"[TraceML] Error in ModuleGradientHook for layer {self.layer_name}"
            )


class ParamGradientHook:
    """
    Per-parameter grad hook (registered on Tensor) to capture grad sizes for that param.
    """

    def __init__(self, model_id: int, layer_name: str, param_name: str):
        self.model_id = model_id
        self.layer_name = layer_name
        self.param_name = param_name
        setup_error_logger()
        self.logger = get_error_logger("ParamGradientHook")

    def __call__(self, grad: torch.Tensor):
        try:
            device_mb: Dict[str, float] = {}
            _accumulate_tensor_sizes_mb(grad, device_mb)

            if device_mb:
                elem = GradientEvent(
                    model_id=self.model_id,
                    layer_name=self.layer_name,
                    param_name=self.param_name,
                    per_device_memory=device_mb.copy(),
                )
                try:
                    gradient_queue.put_nowait(elem)
                except Full:
                    pass
        except Exception:
            self.logger.error(
                f"[TraceML] Error in ParamGradientHook for param {self.param_name}"
            )


def attach_layer_gradient_hooks(model: nn.Module) -> None:
    """
    Attach `register_full_backward_hook` to all leaf modules to capture grad_output sizes.
    Idempotent per model object.
    """
    model_id = id(model)
    if _grad_layer_registry.get(model_id):
        return

    try:
        for name, module in model.named_modules():
            # leaf only
            if any(module.children()):
                continue
            # full backward hook works on module outputs
            module.register_full_backward_hook(LayerGradientHook(model_id, name))
        _grad_layer_registry[model_id] = True
    except Exception as e:
        print(f"[TraceML] Failed to attach module gradient hooks: {e}", file=sys.stderr)


def _build_param_to_module_map(model: nn.Module) -> Dict[str, str]:
    mapping = {}
    for module_name, module in model.named_modules():
        for pname, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{pname}" if module_name else pname
            mapping[full_name] = module_name or "<root>"
    return mapping


def attach_param_gradient_hooks(model: nn.Module) -> None:
    """
    Attach per-parameter grad hooks to capture grad tensor sizes of parameters.
    Idempotent per model object.
    """
    model_id = id(model)
    if _grad_param_registry.get(model_id):
        return

    try:
        param_to_module = _build_param_to_module_map(model)

        for name, p in model.named_parameters(recurse=True):
            if p.requires_grad:
                # register_hook attaches to the Tensor that will receive .grad
                layer_name = param_to_module.get(name, "<unknown>")
                p.register_hook(
                    ParamGradientHook(
                        model_id,
                        layer_name,
                        name,
                    )
                )
        _grad_param_registry[model_id] = True
    except Exception as e:
        print(f"[TraceML] Failed to attach param gradient hooks: {e}", file=sys.stderr)


def attach_all_gradient_hooks(model: nn.Module) -> None:
    """
    Attach gradient hooks to a model.
    Always attaches param hooks (safe).
    Optionally attaches module backward hooks (risky with AMP).
    """
    ## param gradients are not used to commented
    ## attach_param_gradient_hooks(model)
    attach_layer_gradient_hooks(model)
