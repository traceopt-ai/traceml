import torch.nn as nn
from .layer_forward_memory_hook import flush_layer_forward_memory_buffers
from .layer_backward_memory_hook import flush_layer_backward_memory_buffers
from .layer_forward_time_hooks import flush_layer_forward_time_buffers
from .layer_backward_time_hooks import flush_layer_backward_time_buffers
from .model_forward_memory_hook import flush_model_forward_memory_buffers
from .step_memory import flush_step_memory_buffer
from .steptimer import flush_step_time_buffer


def flush_step_events(model: nn.Module, step: int) -> None:
    flush_layer_forward_memory_buffers(model, step)
    flush_layer_backward_memory_buffers(model, step)
    flush_layer_forward_time_buffers(model, step)
    flush_layer_backward_time_buffers(model, step)
    flush_model_forward_memory_buffers(model, step)
    flush_step_memory_buffer(model, step)
    flush_step_time_buffer(step)
