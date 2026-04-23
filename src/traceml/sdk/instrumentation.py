"""
Core training instrumentation helpers used by TraceML.

This module contains the actual tracing context managers and hook attachment
logic. It intentionally has no import-time side effects. Patch installation is
owned by `traceml.init(...)` and the legacy `traceml.decorators` compatibility
layer.

New path
--------
- `import traceml`
- `traceml.init(...)`
- `traceml.trace_step(...)`

Legacy compatibility path
-------------------------
- `from traceml.decorators import trace_step`
- import side effects remain enabled there for backward compatibility
"""

from __future__ import annotations

import functools
import os
import sys
from contextlib import contextmanager
from typing import Callable, List, Optional

import torch.nn as nn

from traceml.instrumentation.hooks.layer_backward_memory_hooks import (
    attach_layer_backward_memory_hooks,
)
from traceml.instrumentation.hooks.layer_backward_time_hooks import (
    attach_layer_backward_time_hooks,
)
from traceml.instrumentation.hooks.layer_forward_memory_hooks import (
    attach_layer_forward_memory_hooks,
)
from traceml.instrumentation.hooks.layer_forward_time_hooks import (
    attach_layer_forward_time_hooks,
)
from traceml.instrumentation.hooks.optimizer_hooks import (
    ensure_optimizer_timing_installed,
)
from traceml.instrumentation.patches.backward_auto_timer_patch import (
    backward_auto_timer,
)
from traceml.instrumentation.patches.forward_auto_timer_patch import (
    forward_auto_timer,
)
from traceml.instrumentation.patches.h2d_auto_timer_patch import (
    h2d_auto_timer,
)
from traceml.runtime.state import TraceSessionState, get_trace_session_state
from traceml.utils.entry_hook import attach_execution_entry_hooks
from traceml.utils.flush_buffers import flush_step_events
from traceml.utils.layer_parameter_memory import (
    collect_layer_parameter_memory,
    model_queue,
)
from traceml.utils.step_memory import StepMemoryTracker
from traceml.utils.timing import timed_region


def _log_instrumentation_error(message: str, exc: Exception) -> None:
    """
    Log instrumentation failures without interrupting user training.

    The TraceML runtime configures the file-backed error logger when running
    under the launcher. Direct SDK users may not have configured it, so stderr
    remains as a tiny fallback signal for the same behavior this module had
    before the state refactor.
    """
    try:
        from traceml.loggers.error_log import get_error_logger

        get_error_logger("TraceInstrumentation").exception(
            "[TraceML] %s", message
        )
    except Exception:
        pass

    print(f"[TraceML] {message}: {exc}", file=sys.stderr)


def _traceml_disabled() -> bool:
    return os.environ.get("TRACEML_DISABLED", "0") == "1"


def _traceml_profile() -> str:
    return (os.environ.get("TRACEML_PROFILE", "run") or "run").strip().lower()


def _should_auto_install_optimizer_timing() -> bool:
    """
    Return True when `trace_step(...)` should install global optimizer timing.

    Behavior
    --------
    - Without an explicit init config, preserve historical behavior.
    - With explicit init:
      - auto      -> install optimizer hooks automatically
      - manual    -> do not install hooks
      - selective -> do not install hooks

    Rationale
    ---------
    Optimizer timing ownership must be unambiguous:
    - legacy / auto path uses global optimizer hooks
    - manual / selective path uses `traceml.wrap_optimizer(...)`
    """
    try:
        from traceml.sdk.initial import get_init_config
    except Exception:
        return True

    cfg = get_init_config()
    if cfg is None:
        return True

    return getattr(cfg, "mode", "auto") == "auto"


class _TraceStateMeta(type):
    @property
    def step(cls) -> int:
        return get_trace_session_state().step

    @step.setter
    def step(cls, value: int) -> None:
        get_trace_session_state().set_step(value)


class TraceState(metaclass=_TraceStateMeta):
    """
    Compatibility facade for TraceML's process-local step counter.

    New code should use ``traceml.runtime.state.TraceSessionState`` or
    ``get_trace_session_state()``. This class stays intentionally small so
    existing imports and assignments such as ``TraceState.step += 1`` continue
    to resolve to the same underlying session state.
    """

    @classmethod
    def session(cls) -> TraceSessionState:
        """Return the active TraceML session state."""
        return get_trace_session_state()

    @classmethod
    def reset(cls, step: int = 0) -> int:
        """Reset the active TraceML step counter."""
        return get_trace_session_state().reset(step)

    @classmethod
    def advance(cls, delta: int = 1) -> int:
        """Advance the active TraceML step counter."""
        return get_trace_session_state().advance_step(delta)


@contextmanager
def trace_step(model: nn.Module):
    """
    Define a single training step boundary.

    Responsibilities
    ----------------
    - Mark the semantic start/end of a training step
    - Attribute step-scoped timing events
    - Advance the global step counter
    - Trigger step-end memory sampling
    - Flush buffered step timing events

    Important
    ---------
    This function does not install automatic framework patches on its own.
    The new SDK path expects callers to choose an explicit init policy via
    `traceml.init(...)`. Legacy `traceml.decorators` imports still preserve
    automatic patch installation for backward compatibility.

    Safety
    ------
    - Never blocks training
    - Never swallows user exceptions
    - Best-effort instrumentation only
    """
    if _traceml_disabled():
        yield
        return

    trace_state = get_trace_session_state()
    mem_tracker = StepMemoryTracker(model)
    step_completed = False

    try:
        mem_tracker.reset()
    except Exception as exc:
        _log_instrumentation_error("reset failed", exc)

    try:
        with timed_region(
            "_traceml_internal:step_time", scope="step", use_gpu=False
        ):
            with forward_auto_timer(), backward_auto_timer(), h2d_auto_timer():
                if _should_auto_install_optimizer_timing():
                    ensure_optimizer_timing_installed()
                yield
                step_completed = True
    finally:
        if step_completed:
            trace_state.advance_step()

        try:
            mem_tracker.record()
        except Exception as exc:
            _log_instrumentation_error("record failed", exc)

        try:
            flush_step_events(model, trace_state.step)
        except Exception as exc:
            _log_instrumentation_error("flush failed", exc)


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
) -> None:
    """
    Manually trace a PyTorch model instance.

    This is primarily used by the deep profile and integration layers for
    model-level hook attachment. It is independent of the automatic patch
    policy configured by `traceml.init(...)`.
    """
    if _traceml_disabled() or _traceml_profile() != "deep":
        return

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

    except Exception as exc:
        _log_instrumentation_error(
            "Failed to trace model instance",
            exc,
        )


def trace_time(
    name: str,
    scope: str = "global",
    use_gpu: bool = True,
) -> Callable:
    """
    Decorator to time a function.

    Parameters
    ----------
    name:
        Human-readable label for this timed region.
    scope:
        Semantic scope of this timing:
        - "step": attributed to the current training step
        - "global": not step-attributed
    use_gpu:
        If True, record CUDA timing when available.
    """
    if _traceml_disabled():
        return lambda func: func

    if scope not in ("step", "global"):
        raise ValueError(
            f"Invalid scope {scope!r}. Expected 'step' or 'global'."
        )

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timed_region(name, scope=scope, use_gpu=use_gpu):
                return func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "TraceState",
    "TraceSessionState",
    "get_trace_session_state",
    "trace_step",
    "trace_model_instance",
    "trace_time",
]
