import torch.nn as nn
from .layer_forward_memory_hook import flush_layer_forward_memory_buffers
from .layer_backward_memory_hook import flush_layer_backward_memory_buffers
from .layer_forward_time_hooks import flush_layer_forward_time_buffers
from .layer_backward_time_hooks import flush_layer_backward_time_buffers

def flush_traceml_buffers(attached: dict, model: nn.Module) -> None:
    if attached.get("layer_forward_memory"):
        flush_layer_forward_memory_buffers(model)
    if attached.get("layer_backward_memory"):
        flush_layer_backward_memory_buffers(model)
    if attached.get("layer_forward_time"):
        flush_layer_forward_time_buffers(model)
    if attached.get("layer_backward_time"):
        flush_layer_backward_time_buffers(model)
