"""
Host-to-Device (H2D) transfer timing auto-patch.

Patches ``torch.Tensor.to()`` globally so that calls which move a tensor from
CPU to a CUDA device are timed during an active TraceML step.

Design
------
The patch is structured identically to the forward and backward auto-timer
patches:

1. The patch is installed **once** at ``traceml.init()`` time via
   ``patch_h2d()``.
2. A thread-local flag (``_H2D_TLS._traceml_h2d_enabled``) gates whether
   timing is active. The flag is normally raised while inside
   ``trace_step()``.
3. The ``h2d_auto_timer`` context manager toggles the flag.  It is entered by
   ``trace_step`` the same way ``forward_auto_timer`` and
   ``backward_auto_timer`` are used.

Framework integrations whose input transfer occurs before ``trace_step`` may
opt in to a paired ``step_time`` segment for each observed H2D transfer. This
keeps the transfer in Traced Step Time without placing surrounding DataLoader
work inside the step-time region.

GPU timing
----------
CUDA events are recorded on the current stream around the ``.to()`` call.
``start.record()`` enqueues a timestamp marker before the DMA op;
``end.record()`` enqueues one after.  Once ``end`` fires (resolved later via
``event.query()`` in the sampler — see
``instrumentation/step_events.py::TimeEvent.try_resolve``),
``start.elapsed_time(end)`` returns the GPU-side wall-clock duration between
the two markers — including the asynchronous DMA itself.  Accuracy is the
same for ``non_blocking=True`` and ``non_blocking=False``.

Filtering
---------
Only calls whose target resolves to a CUDA device are timed.  CPU-to-CPU
copies, dtype-only casts, and memory-format conversions are passed through
without instrumentation overhead.

Event name
----------
``_traceml_internal:h2d_time``  (same namespace as all other internal events)
"""

from __future__ import annotations

import threading
from contextlib import ExitStack
from typing import Any

import torch

from traceml_ai.instrumentation.h2d import should_time_h2d
from traceml_ai.runtime.arming import is_tracing_armed
from traceml_ai.utils.timing import timed_region

_H2D_TLS = threading.local()

_ORIG_TENSOR_TO = torch.Tensor.to

# TLS helpers


def _enabled() -> bool:
    return bool(getattr(_H2D_TLS, "_traceml_h2d_enabled", False))


def _include_step_time() -> bool:
    """Return whether each H2D event also contributes a step-time segment."""
    return bool(getattr(_H2D_TLS, "_traceml_h2d_include_step_time", False))


# Patched method


def _traceml_tensor_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
    if not is_tracing_armed() or not _enabled():
        return _ORIG_TENSOR_TO(self, *args, **kwargs)

    if not should_time_h2d(self, args, kwargs):
        return _ORIG_TENSOR_TO(self, *args, **kwargs)

    with ExitStack() as stack:
        if _include_step_time():
            stack.enter_context(
                timed_region(
                    "_traceml_internal:step_time",
                    scope="step",
                    record_gpu_events=True,
                )
            )
        stack.enter_context(
            timed_region(
                "_traceml_internal:h2d_time",
                scope="step",
                record_gpu_events=True,
            )
        )
        return _ORIG_TENSOR_TO(self, *args, **kwargs)


# Public API


def patch_h2d() -> None:
    """
    Patch ``torch.Tensor.to`` once.  Safe to call multiple times.
    """
    if getattr(torch.Tensor, "_traceml_h2d_patched", False):
        return

    torch.Tensor.to = _traceml_tensor_to  # type: ignore[assignment]
    torch.Tensor._traceml_h2d_patched = True  # type: ignore[attr-defined]


class h2d_auto_timer:
    """
    Context manager that enables H2D timing during its scope.

    Assumes ``patch_h2d()`` has been called once at process startup. When the
    patch is not installed this context manager is a no-op so code paths that
    don't call ``patch_h2d`` (manual / selective mode without H2D) are
    unaffected.

    ``include_step_time=True`` is for a framework-owned input transfer that
    occurs outside its main ``trace_step`` envelope. Each qualifying transfer
    then emits a matching short ``step_time`` segment. The option is false by
    default because transfers performed inside ``trace_step`` are already
    covered by its outer region.

    Each scope should use a fresh context-manager instance. The saved prior
    state belongs to that instance, while the active timing flags themselves
    remain thread-local.
    """

    def __init__(self, *, include_step_time: bool = False) -> None:
        self._include_step_time = bool(include_step_time)
        self._previous_state: tuple[bool, bool] | None = None

    def __enter__(self) -> "h2d_auto_timer":
        self._previous_state = (_enabled(), _include_step_time())
        _H2D_TLS._traceml_h2d_enabled = True
        _H2D_TLS._traceml_h2d_include_step_time = self._include_step_time
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        enabled, include_step_time = self._previous_state or (False, False)
        _H2D_TLS._traceml_h2d_enabled = enabled
        _H2D_TLS._traceml_h2d_include_step_time = include_step_time
        self._previous_state = None
        return False
