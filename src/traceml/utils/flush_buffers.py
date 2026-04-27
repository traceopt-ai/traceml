import os

import torch.nn as nn

from traceml.hooks.layer_backward_memory_hooks import (
    flush_layer_backward_memory_buffers,
)
from traceml.hooks.layer_backward_time_hooks import (
    flush_layer_backward_time_buffers,
)
from traceml.hooks.layer_forward_memory_hooks import (
    flush_layer_forward_memory_buffers,
)
from traceml.hooks.layer_forward_time_hooks import (
    flush_layer_forward_time_buffers,
)

from .step_memory import flush_step_memory_buffer
from .timing import flush_step_time_buffer

TRACEML_DISABLED = os.environ.get("TRACEML_DISABLED") == "1"


def flush_step_events(model: nn.Module, step: int) -> None:
    if TRACEML_DISABLED:
        return

    flush_layer_forward_memory_buffers(model, step)
    flush_layer_backward_memory_buffers(model, step)
    flush_layer_forward_time_buffers(model, step)
    flush_layer_backward_time_buffers(model, step)
    flush_step_memory_buffer(model, step)
    flush_step_time_buffer(step)
