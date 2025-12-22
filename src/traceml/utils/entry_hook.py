import torch.nn as nn
import torch
from traceml.utils.shared_utils import EXECUTION_LAYER


_execution_entry_hook_registry = {}

class ForwardEntryHook:
    def __init__(self, layer_name: str):
        self.layer_name = layer_name

    def __call__(self, module: nn.Module, inputs):
        EXECUTION_LAYER.current = f"forward_{self.layer_name}"


class BackwardEntryHook:
    def __init__(self, layer_name: str):
        self.layer_name = layer_name

    def __call__(self, grad):
        EXECUTION_LAYER.current = f"backward_{self.layer_name}"
        return grad



def attach_execution_entry_hooks(model: nn.Module):
    model_id = id(model)
    if _execution_entry_hook_registry.get(model_id):
        return

    for name, module in model.named_modules():
        if any(module.children()):
            continue

        # forward
        module.register_forward_pre_hook(
            ForwardEntryHook(name)
        )

        # backward
        module.register_full_backward_pre_hook(
            BackwardEntryHook(name)
        )

    _execution_entry_hook_registry[model_id] = True