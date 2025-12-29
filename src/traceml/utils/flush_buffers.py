import torch.nn as nn
from .layer_forward_memory_hook import flush_layer_forward_memory_buffers
from .layer_backward_memory_hook import flush_layer_backward_memory_buffers
from .layer_forward_time_hooks import flush_layer_forward_time_buffers
from .layer_backward_time_hooks import flush_layer_backward_time_buffers
from .model_forward_memory_hook import flush_model_forward_memory_buffers

def flush_traceml_buffers(attached: dict, model: nn.Module, step: int) -> None:
    if attached.get("layer_forward_memory", None):
        flush_layer_forward_memory_buffers(model, step)
    if attached.get("layer_backward_memory", None):
        flush_layer_backward_memory_buffers(model, step)
    if attached.get("layer_forward_time", None):
        flush_layer_forward_time_buffers(model, step)
    if attached.get("layer_backward_time", None):
        flush_layer_backward_time_buffers(model, step)
    if attached.get("model_forward_memory", None):
        flush_model_forward_memory_buffers(model, step)

