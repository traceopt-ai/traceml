import torch.nn as nn
from .activation_memory_hook import flush_activation_memory_buffers
from .gradient_memory_hook import flush_gradient_memory_buffers
from .activation_time_hooks import flush_activation_time_buffers
from .gradient_time_hooks import flush_gradient_time_buffers

def flush_traceml_buffers(attached: dict, model: nn.Module) -> None:
    if attached.get("activation_memory"):
        flush_activation_memory_buffers(model)
    if attached.get("gradient_memory"):
        flush_gradient_memory_buffers(model)
    if attached.get("activation_time"):
        flush_activation_time_buffers(model)
    if attached.get("gradient_time"):
        flush_gradient_time_buffers(model)
    if attached.get("execution"):
        flush_execution_buffers(model)          # TODO


def flush_execution_buffers(model: nn.Module):
    pass