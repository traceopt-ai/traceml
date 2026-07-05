"""
Core training instrumentation helpers used by TraceML.

This module contains the actual tracing context managers and hook attachment
logic. It intentionally has no import-time side effects. Patch installation is
owned by `traceml.init(...)`.

Public path
-----------
- `import traceml_ai as traceml`
- `traceml.init(...)`
- `traceml.trace_step(...)`

The old `import traceml` path remains as a deprecated compatibility alias.
"""

from __future__ import annotations

import functools
import os
import sys
from contextlib import contextmanager
from typing import Callable

import torch.nn as nn

from traceml_ai.instrumentation.hooks.optimizer_hooks import (
    ensure_optimizer_timing_installed,
)
from traceml_ai.instrumentation.patches.backward_auto_timer_patch import (
    backward_auto_timer,
)
from traceml_ai.instrumentation.patches.forward_auto_timer_patch import (
    forward_auto_timer,
)
from traceml_ai.instrumentation.patches.h2d_auto_timer_patch import (
    h2d_auto_timer,
)
from traceml_ai.runtime.state import (
    TraceSessionState,
    get_trace_session_state,
    mark_trace_step_flushed,
)
from traceml_ai.utils.flush_buffers import flush_step_events
from traceml_ai.utils.step_memory import StepMemoryTracker
from traceml_ai.utils.timing import timed_region


def _log_instrumentation_error(message: str, exc: Exception) -> None:
    """
    Log instrumentation failures without interrupting user training.

    The TraceML runtime configures the file-backed error logger when running
    under the launcher. Direct SDK users may not have configured it, so stderr
    remains as a tiny fallback signal for the same behavior this module had
    before the state refactor.
    """
    try:
        from traceml_ai.loggers.error_log import get_error_logger

        get_error_logger("TraceInstrumentation").exception(
            "[TraceML] %s", message
        )
    except Exception:
        pass

    print(f"[TraceML] {message}: {exc}", file=sys.stderr)


def _traceml_disabled() -> bool:
    return os.environ.get("TRACEML_DISABLED", "0") == "1"


def _should_auto_install_optimizer_timing() -> bool:
    """
    Return True when `trace_step(...)` should install global optimizer timing.

    Behavior
    --------
    - Without an explicit init config, do not install hooks.
    - With explicit init:
      - auto      -> install optimizer hooks automatically
      - manual    -> do not install hooks
      - selective -> do not install hooks

    Rationale
    ---------
    Optimizer timing ownership must be unambiguous:
    - auto path uses global optimizer hooks
    - manual / selective path uses `traceml.wrap_optimizer(...)`
    """
    try:
        from traceml_ai.sdk.initial import get_init_config
    except Exception:
        return False

    cfg = get_init_config()
    if cfg is None:
        return False

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
    """Define a single training step boundary."""
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
            "_traceml_internal:step_time",
            scope="step",
            record_gpu_events=False,
        ):
            with (
                forward_auto_timer(model),
                backward_auto_timer(),
                h2d_auto_timer(),
            ):
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

        if step_completed:
            try:
                mark_trace_step_flushed(trace_state.step)
            except Exception as exc:
                _log_instrumentation_error(
                    "recording state update failed", exc
                )


def trace_time(
    name: str,
    scope: str = "global",
    record_gpu_events: bool = True,
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
    record_gpu_events:
        If True, record PyTorch CUDA stream events when available.
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
            with timed_region(
                name,
                scope=scope,
                record_gpu_events=record_gpu_events,
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "TraceState",
    "TraceSessionState",
    "get_trace_session_state",
    "trace_step",
    "trace_time",
]
