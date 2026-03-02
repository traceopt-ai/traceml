import functools
import sys
from contextlib import contextmanager
from typing import Callable, List, Optional

import torch.nn as nn

from traceml.utils.entry_hook import attach_execution_entry_hooks
from traceml.utils.flush_buffers import flush_step_events
from traceml.utils.hooks.layer_backward_memory_hook import (
    attach_layer_backward_memory_hooks,
)
from traceml.utils.hooks.layer_backward_time_hooks import (
    attach_layer_backward_time_hooks,
)
from traceml.utils.hooks.layer_forward_memory_hook import (
    attach_layer_forward_memory_hooks,
)
from traceml.utils.hooks.layer_forward_time_hooks import (
    attach_layer_forward_time_hooks,
)
from traceml.utils.hooks.optimizer_hook import (
    ensure_optimizer_timing_installed,
)
from traceml.utils.layer_parameter_memory import (
    collect_layer_parameter_memory,
    model_queue,
)
from traceml.utils.patches.backward_auto_timer_patch import (
    backward_auto_timer,
    patch_backward,
)
from traceml.utils.patches.dataloader_patch import patch_dataloader
from traceml.utils.patches.forward_auto_timer_patch import (
    forward_auto_timer,
    patch_forward,
)
from traceml.utils.step_memory import StepMemoryTracker
from traceml.utils.timing import timed_region

# NOTE:
# We intentionally patch torch.utils.data.DataLoader.__iter__ at import time.
# This is a lightweight, observational patch used to infer batch metadata and,
# dataloader fetch time.It is idempotent and safe to import multiple times.

patch_dataloader()
patch_forward()
patch_backward()


class TraceState:
    step = 0


@contextmanager
def trace_step(model: nn.Module):
    """
    Defines a single training step boundary.

    Responsibilities
    ----------------
    - Marks the semantic start/end of a training step
    - Attributes step-scoped timing events
    - Advances the global step counter
    - Triggers step-end memory sampling
    - Flushes buffered step timing events

    Safety
    ------
    - Never blocks training
    - Never swallows user exceptions
    - Best-effort instrumentation only

    """

    mem_tracker = StepMemoryTracker(model)
    step_completed = False

    # Step begin (best effort)
    try:
        mem_tracker.reset()
    except Exception as e:
        print(f"[TraceML] reset failed: {e}", file=sys.stderr)

    try:
        with timed_region(
            "_traceml_internal:step_time", scope="step", use_gpu=False
        ):
            with forward_auto_timer(), backward_auto_timer():
                ensure_optimizer_timing_installed()
                yield
                step_completed = True
    finally:
        # Step end
        if step_completed:
            TraceState.step += 1

        # Memory sampling is step-end scoped
        try:
            mem_tracker.record()
        except Exception as e:
            print(f"[TraceML] record failed: {e}", file=sys.stderr)

        try:
            flush_step_events(model, TraceState.step)
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
    include_names: Optional[List[str]] = None,
    exclude_names: Optional[List[str]] = None,
    leaf_only: bool = True,
):
    """
    Manually trace a PyTorch model instance (useful for functional or sequential models).

    Args:
        model (nn.Module): The model instance to trace.
        sample_layer_memory: enqueue model for memory sampling.
        trace_layer_forward_memory: attach activation hooks to capture activations.
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
        print(
            f"[TraceML] Failed to trace model instance: {e}", file=sys.stderr
        )


def trace_time(
    name: str, scope: str = "global", use_gpu: bool = True
) -> Callable:
    """
    Decorator to time a function.

    Parameters
    ----------
    name : str
        Human-readable label for this timed region.
    scope : {"step", "global"}, optional
        Semantic scope of this timing:
        - "step": attributed to the current training step
        - "global": not step-attributed (default)
    use_gpu : bool, optional
        If True, records CUDA timing when available.
    """
    if scope not in ("step", "global"):
        raise ValueError(
            f"Invalid scope '{scope}'. " "Expected 'step' or 'global'."
        )

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timed_region(name, scope=scope, use_gpu=use_gpu):
                return func(*args, **kwargs)

        return wrapper

    return decorator
