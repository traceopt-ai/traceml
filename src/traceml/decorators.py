import functools
import sys
from typing import Callable
import torch.nn as nn
import time
import torch


from traceml.utils.patch import model_queue
from traceml.utils.activation_memory_hook import attach_activation_memory_hooks
from traceml.utils.gradient_hook import attach_all_gradient_hooks
from traceml.utils.activation_time_hooks import attach_activation_time_hooks
from traceml.utils.gradient_time_hooks import attach_gradient_time_hooks
from traceml.utils.steptimer import StepTimeEvent, record_step_time_event


def trace_model(
    sample_layer_memory: bool = True,
    trace_activation_memory: bool = True,
    trace_gradient_memory: bool = True,
    trace_activation_time: bool = True,
    trace_gradient_time: bool = True
) -> Callable:
    """
    Class decorator to automatically trace a PyTorch nn.Module.
    Enqueues the model for parameter-memory sampling and optionally
    attaches activation hooks.

    Args:
        sample_layer_memory: enqueue model for memory sampling.
        trace_activation_memory: attach activation hooks to capture activations.
        trace_gradient_memory: attach gradient hooks to capture grad sizes (module + param).
        trace_activation_time:attach activation *time* hooks (pre + post)
            (only CPU time so wwaiting time + execution time).
        trace_gradient_time:attach gradient *time* hooks (pre + post).
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

                if trace_activation_memory:
                    attach_activation_memory_hooks(self)

                if trace_gradient_memory:
                    attach_all_gradient_hooks(self)

                if trace_activation_time:
                    attach_activation_time_hooks(self)

                if trace_gradient_time:
                    attach_gradient_time_hooks(self)

            except Exception as e:
                print(f"[TraceML] Failed to trace model: {e}", file=sys.stderr)

        cls.__init__ = wrapped_init
        return cls

    return decorator


def trace_model_instance(
    model: nn.Module,
    sample_layer_memory: bool = True,
    trace_activation_memory: bool = True,
    trace_gradient_memory: bool = True,
    trace_activation_time: bool = True,
    trace_gradient_time: bool = True,
):
    """
    Manually trace a PyTorch model instance (useful for functional or sequential models).

    Args:
        model (nn.Module): The model instance to trace.
        sample_layer_memory: enqueue model for memory sampling.
        trace_activation_memory: attach activation hooks to capture activations.
        trace_gradient_memory: attach gradient hooks to capture grad sizes (module + param).
        trace_activation_time:attach activation *time* hooks (pre + post).
        trace_gradient_time:attach gradient *time* hooks (pre + post).
    """
    try:
        if not isinstance(model, nn.Module):
            raise TypeError("trace_model_instance expects an nn.Module.")
        if sample_layer_memory:
            model_queue.put(model)

        if trace_activation_memory:
            attach_activation_memory_hooks(model)

        if trace_gradient_memory:
            attach_all_gradient_hooks(model)

        if trace_activation_time:
            attach_activation_time_hooks(model)

        if trace_gradient_time:
            attach_gradient_time_hooks(model)

    except Exception as e:
        print(f"[TraceML] Failed to trace model instance: {e}", file=sys.stderr)


def trace_timestep(name: str, use_gpu: bool = True) -> Callable:
    """
    Decorator to measure execution time of a function.

    Args:
        name (str): Label for this timer.
        use_gpu (bool): If True and CUDA is available, record GPU timing
                        via CUDA events. Otherwise, only CPU wall-time.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cpu_start = time.time()
            if torch.cuda.is_available():
                device = f"cuda:{torch.cuda.current_device()}"
            else:
                device = "cpu"

            if use_gpu and torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()  # queued in current CUDA stream
                result = func(*args, **kwargs)
                end_event.record()  # queued after kernels

                cpu_end = time.time()

                evt = StepTimeEvent(
                    name=name,
                    device=device,
                    cpu_start=cpu_start,
                    cpu_end=cpu_end,
                    gpu_start=start_event,
                    gpu_end=end_event,
                )
            else:
                result = func(*args, **kwargs)
                cpu_end = time.time()
                evt = StepTimeEvent(
                    name=name, device=device, cpu_start=cpu_start, cpu_end=cpu_end
                )

            record_step_time_event(evt)
            return result

        return wrapper

    return decorator
