import functools
import sys
from typing import Callable
import torch.nn as nn


from traceml.utils.patch import model_queue
from traceml.utils.activation_hook import attach_activation_hooks
from traceml.utils.gradient_hook import attach_all_gradient_hooks


def trace_model(
    sample_layer_memory: bool = True,
    trace_activations: bool = True,
    trace_gradients: bool = True
) -> Callable:
    """
    Class decorator to automatically trace a PyTorch nn.Module.
    Enqueues the model for parameter-memory sampling and optionally
    attaches activation hooks.

    Args:
        sample_layer_memory: enqueue model for memory sampling.
        trace_activations: attach activation hooks to capture activations.
        trace_gradients: attach gradient hooks to capture grad sizes (module + param).
    """

    def decorator(cls):
        if not isinstance(cls, type) or not issubclass(cls, nn.Module):
            raise TypeError(
                "@trace_model can only be applied to nn.Module subclasses for now."
            )

        original_init = cls.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            try:
                if sample_layer_memory:
                    model_queue.put(self)
                if trace_activations:
                    attach_activation_hooks(self)
                if trace_gradients:
                    attach_all_gradient_hooks(self)
            except Exception as e:
                print(f"[TraceML] Failed to trace model: {e}", file=sys.stderr)

        cls.__init__ = wrapped_init
        return cls

    return decorator


def trace_model_instance(
    model: nn.Module,
    sample_layer_memory: bool = True,
    trace_activations: bool = True,
    trace_gradients: bool = True
):
    """
    Manually trace a PyTorch model instance (useful for functional or sequential models).

    Args:
        model (nn.Module): The model instance to trace.
        sample_layer_memory: enqueue model for memory sampling.
        trace_activations: attach activation hooks to capture activations.
        trace_gradients: attach gradient hooks to capture grad sizes (module + param).
    """
    try:
        if not isinstance(model, nn.Module):
            raise TypeError("trace_model_instance expects an nn.Module.")
        if sample_layer_memory:
            model_queue.put(model)
        if trace_activations:
            attach_activation_hooks(model)
        if trace_gradients:
            attach_all_gradient_hooks(model)
    except Exception as e:
        print(f"[TraceML] Failed to trace model instance: {e}", file=sys.stderr)
