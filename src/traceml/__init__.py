"""
Public TraceML package surface.

This module intentionally keeps import-time work minimal so lightweight entry
points such as `traceml.cli` do not pull in optional training dependencies
like PyTorch unless the user actually accesses the instrumentation API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.2.12"

__all__ = [
    "__version__",
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


if TYPE_CHECKING:
    from traceml.api import (
        final_summary,
        init,
        start,
        trace_model_instance,
        trace_step,
        wrap_backward,
        wrap_dataloader_fetch,
        wrap_forward,
        wrap_optimizer,
    )


def __getattr__(name: str) -> Any:
    """
    Lazily resolve public API symbols from `traceml.api`.

    This avoids importing the full SDK stack when callers only need light
    utilities such as CLI helpers.
    """
    if name in __all__:
        from traceml import api as _api

        return getattr(_api, name)
    raise AttributeError(f"module 'traceml' has no attribute {name!r}")
