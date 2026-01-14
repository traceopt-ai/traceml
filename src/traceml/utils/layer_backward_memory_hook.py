import sys
from dataclasses import dataclass
from queue import Queue, Full
from typing import Any, Dict, Tuple, List
from traceml.utils.shared_utils import get_hookable_modules
import torch
import torch.nn as nn

# Shared queue for gradient events
layer_backward_memory_queue: Queue = Queue(maxsize=2048)

# Registries to prevent multiple hook attachments per model
_layer_backward_hook_registry: Dict[int, bool] = {}

# In-memory buffer:
# model_id -> List[(layer_name, param_name, per_device_memory)]
_layer_backward_memory_buffer: Dict[int, List[Tuple[str, Dict[str, float]]]] = {}


@dataclass
class LayerBackwardMemoryEvents:
    """
    Represents gradient memory for a single layer / param batch during backprop.

    For module hooks:
      - per_device_memory sizes over grad_output tensors of that layer.
    """

    model_id: int
    layers: List[Tuple[str, Dict[str, float]]]
    step: int


def get_layer_backward_queue() -> Queue:
    """Return the shared queue of gradient events."""
    return layer_backward_memory_queue


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


class LayerBackwardModuleHook:
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

            if device_mb:
                _layer_backward_memory_buffer.setdefault(self.model_id, []).append(
                    (self.layer_name, device_mb)
                )
        except Exception:
            print(
                f"[TraceML] Error in LayerBackwardHook for layer {self.layer_name}",
                file=sys.stderr,
            )


def flush_layer_backward_memory_buffers(model: nn.Module, step: int) -> None:
    """
    Convert buffered backward memory into GradientEvent objects and enqueue them.
    Intended to be called before optimizer.step().
    """
    model_id = id(model)
    buf = _layer_backward_memory_buffer.pop(model_id, None)

    if not buf:
        return

    event = LayerBackwardMemoryEvents(
        model_id=model_id,
        layers=buf,
        step=step,
    )
    try:
        layer_backward_memory_queue.put_nowait(event)
    except Full:
        pass


def attach_layer_backward_memory_hooks(
    model: nn.Module,
    include_names=None, 
    exclude_names=None, 
    leaf_only=True
) -> None:
    """
    Attach `register_full_backward_hook` to all leaf modules to capture grad_output sizes.
    Idempotent per model object.
    """
    model_id = id(model)
    if _layer_backward_hook_registry.get(model_id):
        return

    try:
        for name, module in get_hookable_modules(model, include_names, exclude_names, leaf_only):
            # full backward hook works on module outputs
            module.register_full_backward_hook(LayerBackwardModuleHook(model_id, name))
        _layer_backward_hook_registry[model_id] = True
    except Exception as e:
        print(f"[TraceML] Failed to attach layer backward hooks: {e}", file=sys.stderr)
