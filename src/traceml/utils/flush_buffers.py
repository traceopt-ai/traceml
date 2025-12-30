import torch.nn as nn
from .layer_forward_memory_hook import flush_layer_forward_memory_buffers
from .layer_backward_memory_hook import flush_layer_backward_memory_buffers
from .layer_forward_time_hooks import flush_layer_forward_time_buffers
from .layer_backward_time_hooks import flush_layer_backward_time_buffers
from .model_forward_memory_hook import flush_model_forward_memory_buffers


def flush_traceml_buffers(model: nn.Module, step: int) -> None:
    flush_layer_forward_memory_buffers(model, step)
    flush_layer_backward_memory_buffers(model, step)
    flush_layer_forward_time_buffers(model, step)
    flush_layer_backward_time_buffers(model, step)
    flush_model_forward_memory_buffers(model, step)
    flush_step_timers(model, step)
