import functools
import sys
from typing import Callable, Optional, List
import torch.nn as nn
import time
import torch
from contextlib import contextmanager

from traceml.utils.layer_parameter_memory import (
    model_queue,
    collect_layer_parameter_memory,
)
from traceml.utils.step_memory import StepMemoryTracker
from traceml.utils.layer_forward_memory_hook import attach_layer_forward_memory_hooks
from traceml.utils.layer_backward_memory_hook import attach_layer_backward_memory_hooks

from traceml.utils.layer_forward_time_hooks import attach_layer_forward_time_hooks
from traceml.utils.layer_backward_time_hooks import attach_layer_backward_time_hooks

from traceml.utils.steptimer import (
    StepTimeEvent, 
    record_step_time_event, 
    timed_region,
    begin_timed_region,
    end_timed_region
)
from traceml.utils.entry_hook import attach_execution_entry_hooks
from traceml.utils.flush_buffers import flush_traceml_buffers

from traceml.utils.dataloader_patch import patch_dataloader
from traceml.utils.cuda_event_pool import get_cuda_event

# NOTE:
# We intentionally patch torch.utils.data.DataLoader.__iter__ at import time.
# This is a lightweight, observational patch used to infer batch metadata and,
# dataloader fetch time.It is idempotent and safe to import multiple times.

patch_dataloader()


try:
    from traceml.utils.lightning_patch import patch_lightning
    patch_lightning()
except Exception:
    pass

class TraceState:
    step = 0

def begin_trace_step(model: nn.Module):
    """Logic to run at the start of a training step."""
    mem_tracker = StepMemoryTracker(model)

    try:
        mem_tracker.reset() # Clears peak memory counters
    except Exception as e:
        print(f"[TraceML] memory reset failed: {e}", file=sys.stderr)

    timer_state = begin_timed_region("_traceml_internal:step_time")
    return {"mem_tracker": mem_tracker, "timer_state": timer_state}

def end_trace_step(model: nn.Module, state: dict):
    """Logic to run at the end of a training step."""
    TraceState.step += 1
    
    # End the timer
    end_timed_region(state["timer_state"])
    
    # Record peak memory
    try:
        state["mem_tracker"].record()
    except Exception as e:
        print(f"[TraceML] memory record failed: {e}", file=sys.stderr)

    # Flush all buffers (layer metrics, step metrics) to dashboard
    try:
        flush_traceml_buffers(model, TraceState.step)
    except Exception as e:
        print(f"[TraceML] buffer flush failed: {e}", file=sys.stderr)

@contextmanager
def trace_step(model: nn.Module):
    """Original context manager, now using split logic internally."""
    state = begin_trace_step(model)
    try:
        yield
    finally:
        end_trace_step(model, state)

def trace_model_instance(
    model: nn.Module,
    sample_layer_memory: bool = True,
    trace_layer_forward_memory: bool = True,
    trace_layer_backward_memory: bool = True,
    trace_layer_forward_time: bool = True,
    trace_layer_backward_time: bool = True,
    trace_execution: bool = True,
    include_names: Optional[List[str]] = None,
    exclude_names: Optional[List[str]] = None,
    leaf_only: bool = True
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
    try:
        if not isinstance(model, nn.Module):
            raise TypeError("trace_model_instance expects an nn.Module.")
        if sample_layer_memory:
            model._traceml_include_names = include_names
            model._traceml_exclude_names = exclude_names
            model._traceml_leaf_only = leaf_only
            layer_memory = collect_layer_parameter_memory(model)
            model_queue.put(layer_memory)

        if trace_layer_forward_memory:
            attach_layer_forward_memory_hooks(
                model,
                include_names=include_names,
                exclude_names=exclude_names,
                leaf_only=leaf_only,
            )

        if trace_layer_backward_memory:
            attach_layer_backward_memory_hooks(
                model,
                include_names=include_names,
                exclude_names=exclude_names,
                leaf_only=leaf_only,
            )

        if trace_layer_forward_time:
            attach_layer_forward_time_hooks(
                model,
                include_names=include_names,
                exclude_names=exclude_names,
                leaf_only=leaf_only,
            )

        if trace_layer_backward_time:
            attach_layer_backward_time_hooks(
                model,
                include_names=include_names,
                exclude_names=exclude_names,
                leaf_only=leaf_only,
            )

        if trace_execution:
            attach_execution_entry_hooks(model)

    except Exception as e:
        print(f"[TraceML] Failed to trace model instance: {e}", file=sys.stderr)


def trace_time(name: str, use_gpu: bool = True) -> Callable:
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

            if use_gpu and torch.cuda.is_available():
                device = f"cuda:{torch.cuda.current_device()}"
                start_event = get_cuda_event()
                end_event = get_cuda_event()

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
                device = "cpu"
                result = func(*args, **kwargs)
                cpu_end = time.time()
                evt = StepTimeEvent(
                    name=name, device=device, cpu_start=cpu_start, cpu_end=cpu_end
                )

            record_step_time_event(evt)
            return result

        return wrapper

    return decorator
