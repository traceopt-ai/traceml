"""
Public TraceML module-level API.

This module provides the stable user-facing entrypoints exposed at
`import traceml`.

Design goals
------------
- Keep the public API simple and discoverable
- Preserve backward compatibility with existing decorator imports
- Support an explicit initialization model
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from traceml.sdk.initial import TraceMLInitConfig
from traceml.sdk.summary_client import final_summary as _final_summary


def trace_step(*args: Any, **kwargs: Any) -> Any:
    """
    Lazily resolve and return the TraceML step tracing context manager.

    This path intentionally avoids import-time patch installation. The new
    model expects callers to opt into automatic patching explicitly via
    `traceml.init(...)`.
    """
    from traceml.sdk.instrumentation import trace_step as _trace_step

    return _trace_step(*args, **kwargs)


def trace_model_instance(*args: Any, **kwargs: Any) -> Any:
    """
    Lazily resolve and invoke TraceML model hook attachment.
    """
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
    """
    Lazily resolve and apply TraceML dataloader-fetch wrapping.
    """
    from traceml.sdk.wrappers import (
        wrap_dataloader_fetch as _wrap_dataloader_fetch,
    )

    return _wrap_dataloader_fetch(*args, **kwargs)


def wrap_forward(*args: Any, **kwargs: Any) -> Any:
    """
    Lazily resolve and apply TraceML forward wrapping.
    """
    from traceml.sdk.wrappers import wrap_forward as _wrap_forward

    return _wrap_forward(*args, **kwargs)


def wrap_backward(*args: Any, **kwargs: Any) -> Any:
    """
    Lazily resolve and apply TraceML backward wrapping.
    """
    from traceml.sdk.wrappers import wrap_backward as _wrap_backward

    return _wrap_backward(*args, **kwargs)


def wrap_optimizer(*args: Any, **kwargs: Any) -> Any:
    """
    Lazily resolve and apply TraceML optimizer-step wrapping.
    """
    from traceml.sdk.wrappers import wrap_optimizer as _wrap_optimizer

    return _wrap_optimizer(*args, **kwargs)


def wrap_h2d(*args: Any, **kwargs: Any) -> Any:
    """
    Lazily resolve and apply TraceML H2D transfer wrapping.
    """
    from traceml.sdk.wrappers import wrap_h2d as _wrap_h2d

    return _wrap_h2d(*args, **kwargs)


def init(
    *,
    mode: str = "auto",
    patch_dataloader: Optional[bool] = None,
    patch_forward: Optional[bool] = None,
    patch_backward: Optional[bool] = None,
    patch_h2d: Optional[bool] = None,
) -> TraceMLInitConfig:
    """
    Initialize TraceML instrumentation for the current process.

    Parameters
    ----------
    mode:
        One of:
        - "auto"
        - "manual"
        - "selective"

        The alias "custom" is also accepted and maps to "selective".
    patch_dataloader:
        Selective-mode-only override controlling automatic DataLoader patching.
    patch_forward:
        Selective-mode-only override controlling automatic forward patching.
    patch_backward:
        Selective-mode-only override controlling automatic backward patching.
    patch_h2d:
        Selective-mode-only override controlling automatic H2D patching.

    Returns
    -------
    TraceMLInitConfig
        The effective init config for this process.

    Notes
    -----
    - `mode="auto"` patches all supported automatic instrumentation points.
    - `mode="manual"` patches none of them.
    - `mode="selective"` patches only the explicitly enabled subset.
    """
    from traceml.sdk.initial import init as _init

    return _init(
        mode=mode,
        patch_dataloader=patch_dataloader,
        patch_forward=patch_forward,
        patch_backward=patch_backward,
        patch_h2d=patch_h2d,
    )


def start(
    *,
    mode: str = "auto",
    patch_dataloader: Optional[bool] = None,
    patch_forward: Optional[bool] = None,
    patch_backward: Optional[bool] = None,
    patch_h2d: Optional[bool] = None,
) -> TraceMLInitConfig:
    """
    Alias for `traceml.init(...)` during the transition.
    """
    from traceml.sdk.initial import start as _start

    return _start(
        mode=mode,
        patch_dataloader=patch_dataloader,
        patch_forward=patch_forward,
        patch_backward=patch_backward,
        patch_h2d=patch_h2d,
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
    "wrap_h2d",
]
