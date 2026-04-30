"""Public TraceML module-level API."""

from __future__ import annotations

from typing import Any, Dict, Optional

from traceml.sdk.initial import TraceMLInitConfig
from traceml.sdk.summary_client import final_summary as _final_summary


def trace_step(*args: Any, **kwargs: Any) -> Any:
    """Return the TraceML step tracing context manager."""
    from traceml.sdk.instrumentation import trace_step as _trace_step

    return _trace_step(*args, **kwargs)


def trace_model_instance(*args: Any, **kwargs: Any) -> Any:
    """Attach TraceML model hooks."""
    from traceml.sdk.instrumentation import (
        trace_model_instance as _trace_model_instance,
    )

    return _trace_model_instance(*args, **kwargs)


def final_summary(
    *,
    timeout_sec: float = 30.0,
    poll_interval_sec: float = 0.1,
    print_text: bool = False,
    rank0_only: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Return a finalized TraceML summary for the active session.
    """
    return _final_summary(
        timeout_sec=timeout_sec,
        poll_interval_sec=poll_interval_sec,
        print_text=print_text,
        rank0_only=rank0_only,
    )


def wrap_dataloader_fetch(*args: Any, **kwargs: Any) -> Any:
    """Wrap dataloader fetch timing."""
    from traceml.sdk.wrappers import (
        wrap_dataloader_fetch as _wrap_dataloader_fetch,
    )

    return _wrap_dataloader_fetch(*args, **kwargs)


def wrap_forward(*args: Any, **kwargs: Any) -> Any:
    """Wrap forward-pass timing."""
    from traceml.sdk.wrappers import wrap_forward as _wrap_forward

    return _wrap_forward(*args, **kwargs)


def wrap_backward(*args: Any, **kwargs: Any) -> Any:
    """Wrap backward-pass timing."""
    from traceml.sdk.wrappers import wrap_backward as _wrap_backward

    return _wrap_backward(*args, **kwargs)


def wrap_optimizer(*args: Any, **kwargs: Any) -> Any:
    """Wrap optimizer-step timing."""
    from traceml.sdk.wrappers import wrap_optimizer as _wrap_optimizer

    return _wrap_optimizer(*args, **kwargs)


def init(
    *,
    mode: str = "auto",
    patch_dataloader: Optional[bool] = None,
    patch_forward: Optional[bool] = None,
    patch_backward: Optional[bool] = None,
) -> TraceMLInitConfig:
    """Initialize TraceML instrumentation for this process."""
    from traceml.sdk.initial import init as _init

    return _init(
        mode=mode,
        patch_dataloader=patch_dataloader,
        patch_forward=patch_forward,
        patch_backward=patch_backward,
    )


def start(
    *,
    mode: str = "auto",
    patch_dataloader: Optional[bool] = None,
    patch_forward: Optional[bool] = None,
    patch_backward: Optional[bool] = None,
) -> TraceMLInitConfig:
    """Alias for `traceml.init(...)`."""
    from traceml.sdk.initial import start as _start

    return _start(
        mode=mode,
        patch_dataloader=patch_dataloader,
        patch_forward=patch_forward,
        patch_backward=patch_backward,
    )


__all__ = [
    "trace_step",
    "trace_model_instance",
    "final_summary",
    "start",
    "init",
    "wrap_dataloader_fetch",
    "wrap_forward",
    "wrap_backward",
    "wrap_optimizer",
]
