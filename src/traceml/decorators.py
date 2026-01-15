import functools
import sys
from typing import Callable
import torch.nn as nn
import time
import torch
from contextlib import contextmanager

from traceml.utils.patch import model_queue
from traceml.utils.step_memory import StepMemoryTracker
from traceml.utils.layer_forward_memory_hook import attach_layer_forward_memory_hooks
from traceml.utils.layer_backward_memory_hook import attach_layer_backward_memory_hooks

from traceml.utils.layer_forward_time_hooks import attach_layer_forward_time_hooks
from traceml.utils.layer_backward_time_hooks import attach_layer_backward_time_hooks

from traceml.utils.steptimer import StepTimeEvent, record_step_time_event, timed_region
from traceml.utils.entry_hook import attach_execution_entry_hooks
from traceml.utils.flush_buffers import flush_traceml_buffers

from traceml.utils.dataloader_patch import patch_dataloader
from traceml.utils.cuda_event_pool import get_cuda_event

# NOTE:
# We intentionally patch torch.utils.data.DataLoader.__iter__ at import time.
# This is a lightweight, observational patch used to infer batch metadata and,
# dataloader fetch time.It is idempotent and safe to import multiple times.

patch_dataloader()


class TraceState:
    step = 0


@contextmanager
def trace_step(model: nn.Module):
    mem_tracker = StepMemoryTracker(model)

    try:
        mem_tracker.reset()
    except Exception as e:
        print(f"[TraceML] reset failed: {e}", file=sys.stderr)

    start_timed = False
    step_completed = False

    try:    ## User code block
        try: ## timed_region
            with timed_region("_traceml_internal:step_time", use_gpu=True):
                start_timed = True
                yield
                step_completed = True
        except Exception:
            if not start_timed:
                yield  # timed_region failed to enter
                step_completed = True
            else:
                raise  # user code failed → propagate
    finally:
        if step_completed:
            TraceState.step += 1
        try:
            mem_tracker.record()
        except Exception as e:
            print(f"[TraceML] record failed: {e}", file=sys.stderr)

        try:
            flush_traceml_buffers(model, TraceState.step)
        except Exception as e:
            print(f"[TraceML] flush failed: {e}", file=sys.stderr)



def trace_model_instance(
    model: nn.Module,
    sample_layer_memory: bool = True,
    trace_layer_forward_memory: bool = True,
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
    try:
        if not isinstance(model, nn.Module):
            raise TypeError("trace_model_instance expects an nn.Module.")
        if sample_layer_memory:
            model_queue.put(model)

        if trace_layer_forward_memory:
            attach_layer_forward_memory_hooks(model)

        if trace_layer_backward_memory:
            attach_layer_backward_memory_hooks(model)

        if trace_layer_forward_time:
            attach_layer_forward_time_hooks(model)

        if trace_layer_backward_time:
            attach_layer_backward_time_hooks(model)

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
