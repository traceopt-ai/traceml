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
   timing is active.  The flag is raised only while inside ``trace_step()``.
3. The ``h2d_auto_timer`` context manager toggles the flag.  It is entered by
   ``trace_step`` the same way ``forward_auto_timer`` and
   ``backward_auto_timer`` are used.

GPU timing
----------
CUDA events are recorded around the ``.to()`` call via ``timed_region``.  For
blocking transfers (``non_blocking=False``, the default) this accurately
captures DMA time.  For ``non_blocking=True`` the events bound the time until
the DMA is queued on the current stream, which is a useful lower bound.

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
from typing import Any

import torch

from traceml.utils.timing import timed_region

_H2D_TLS = threading.local()

_ORIG_TENSOR_TO = torch.Tensor.to

# TLS helpers


def _enabled() -> bool:
    return bool(getattr(_H2D_TLS, "_traceml_h2d_enabled", False))


# Device-target detection


def _is_cuda_target(args: tuple, kwargs: dict) -> bool:
    """
    Return True when the ``.to()`` call is moving data to a CUDA device.

    Handles the common calling conventions:
      tensor.to("cuda:0")
      tensor.to(torch.device("cuda"))
      tensor.to(device="cuda:0")
      tensor.to(other_cuda_tensor)   # copies device from other_tensor
    """
    first = args[0] if args else None

    if isinstance(first, str) and "cuda" in first:
        return True
    if isinstance(first, torch.device) and first.type == "cuda":
        return True
    if isinstance(first, torch.Tensor) and first.is_cuda:
        return True

    device = kwargs.get("device")
    if isinstance(device, str) and "cuda" in device:
        return True
    if isinstance(device, torch.device) and device.type == "cuda":
        return True

    return False


# Patched method


def _traceml_tensor_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
    if not _enabled() or not _is_cuda_target(args, kwargs):
        return _ORIG_TENSOR_TO(self, *args, **kwargs)

    with timed_region(
        "_traceml_internal:h2d_time",
        scope="step",
        use_gpu=True,
    ):
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

    Must be used inside ``trace_step``; assumes ``patch_h2d()`` has been
    called once at process startup.  When the patch is not installed this
    context manager is a no-op so code paths that don't call ``patch_h2d``
    (manual / selective mode without H2D) are unaffected.
    """

    def __enter__(self) -> "h2d_auto_timer":
        _H2D_TLS._traceml_h2d_enabled = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        _H2D_TLS._traceml_h2d_enabled = False
        return False
