import functools
import sys
from typing import Callable, Optional
import torch.nn as nn
import time
import torch
from contextlib import contextmanager

from traceml.utils.patch import model_queue

from traceml.utils.layer_forward_memory_hook import attach_layer_forward_memory_hooks
from traceml.utils.layer_backward_memory_hook import attach_layer_backward_memory_hooks

from traceml.utils.layer_forward_time_hooks import attach_layer_forward_time_hooks
from traceml.utils.layer_backward_time_hooks import attach_layer_backward_time_hooks

from traceml.utils.steptimer import StepTimeEvent, record_step_time_event
from traceml.utils.entry_hook import attach_execution_entry_hooks
from traceml.utils.flush_buffers import flush_traceml_buffers



@contextmanager
def trace_step(model: nn.Module):
    """
    Defines a TraceML step boundary.
    Currently:
        - flushes TraceML buffers at step end
        - resets peak CUDA memory stats at step start (if available)

    Timing and metadata may be added later.
    """
    try:
        yield
    finally:
        attached = getattr(model, "_trace_attached")
        if attached:
            flush_traceml_buffers(attached, model)


def trace_model(
    sample_layer_memory: bool = True,
    trace_layer_forward__memory: bool = True,
    trace_layer_backward_memory: bool = True,
    trace_layer_forward_time: bool = True,
    trace_layer_backward_time: bool = True,
    trace_execution: bool = True,
) -> Callable:
    """
    Class decorator to automatically trace a PyTorch nn.Module.
    Enqueues the model for parameter-memory sampling and optionally
    attaches activation hooks.

    Args:
        sample_layer_memory: enqueue model for memory sampling.
        trace_layer_forward__memory: attach forward hooks to capture forward pass.
        trace_layer_backward_memory: attach backward hooks to capture grad sizes (module + param).
        trace_layer_forward_time:attach forward *time* hooks (pre + post)
            (only CPU time so waiting time + execution time).
        trace_layer_backward_time:attach backward *time* hooks (pre + post).
        trace_execution: attach execution hooks.
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

            attached = {
                "layer_forward_memory": False,
                "layer_backward_memory": False,
                "trace_layer_forward_time": False,
                "trace_layer_backward_time": False,
                "execution": False,
            }
            try:
                if sample_layer_memory:
                    model_queue.put(self)

                if trace_layer_forward__memory:
                    attach_layer_forward_memory_hooks(self)
                    attached["layer_forward_memory"] = True

                if trace_layer_backward_memory:
                    attach_layer_backward_memory_hooks(self)
                    attached["trace_layer_backward_memory"] = True

                if trace_layer_forward_time:
                    attach_layer_forward_time_hooks(self)
                    attached["trace_layer_forward_time"] = True

                if trace_layer_backward_time:
                    attach_layer_backward_time_hooks(self)
                    attached["trace_layer_backward_time"] = True

                if trace_execution:
                    attach_execution_entry_hooks(self)
                    attached["execution"] = True

                self._trace_attached = attached

            except Exception as e:
                print(f"[TraceML] Failed to trace model: {e}", file=sys.stderr)

        cls.__init__ = wrapped_init
        return cls

    return decorator


def trace_model_instance(
    model: nn.Module,
    sample_layer_memory: bool = True,
    trace_layer_forward__memory: bool = True,
    trace_layer_backward_memory: bool = True,
    trace_layer_forward_time: bool = True,
    trace_layer_backward_time: bool = True,
    trace_execution: bool = True,
):
    """
    Manually trace a PyTorch model instance (useful for functional or sequential models).

    Args:
        model (nn.Module): The model instance to trace.
        sample_layer_memory: enqueue model for memory sampling.
        trace_layer_forward__memory: attach activation hooks to capture activations.
        trace_layer_backward_memory: attach gradient hooks to capture grad sizes (module + param).
        trace_layer_forward_time: attach forward *time* hooks (pre + post).
        trace_layer_backward_time: attach backward *time* hooks (pre + post).
        trace_execution: attach execution hooks.
    """
    attached = {
        "layer_forward_memory": False,
        "layer_backward_memory": False,
        "layer_forward_time": False,
        "layer_backward_time": False,
        "execution": False,
    }
    try:
        if not isinstance(model, nn.Module):
            raise TypeError("trace_model_instance expects an nn.Module.")
        if sample_layer_memory:
            model_queue.put(model)

        if trace_layer_forward__memory:
            attach_layer_forward_memory_hooks(model)
            attached["layer_forward_memory"] = True

        if trace_layer_backward_memory:
            attach_layer_backward_memory_hooks(model)
            attached["layer_backward_memory"] = True

        if trace_layer_forward_time:
            attach_layer_forward_time_hooks(model)
            attached["layer_forward_time"] = True

        if trace_layer_backward_time:
            attach_layer_backward_time_hooks(model)
            attached["layer_backward_time"] = True

        if trace_execution:
            attach_execution_entry_hooks(model)
            attached["execution"] = True

        model._trace_attached = attached

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


