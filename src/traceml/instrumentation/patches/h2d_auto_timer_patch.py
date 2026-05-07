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
CUDA events are recorded on the current stream around the ``.to()`` call.
``start.record()`` enqueues a timestamp marker before the DMA op;
``end.record()`` enqueues one after.  Once ``end`` fires (resolved later via
``event.query()`` in the sampler — see ``utils/timing.py::TimeEvent.try_resolve``),
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
from typing import Any, Optional

import torch

from traceml.utils.timing import timed_region

_H2D_TLS = threading.local()

_ORIG_TENSOR_TO = torch.Tensor.to

# TLS helpers


def _enabled() -> bool:
    return bool(getattr(_H2D_TLS, "_traceml_h2d_enabled", False))


# Device-target detection


def _device_type(value: Any) -> Optional[str]:
    """Return the device type string for a value, or None if not recognisable."""
    if isinstance(value, torch.device):
        return value.type
    if isinstance(value, torch.Tensor):
        return value.device.type
    if isinstance(value, str):
        try:
            return torch.device(value).type
        except (RuntimeError, TypeError):
            return None
    return None


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
    if _device_type(first) == "cuda":
        return True
    if _device_type(kwargs.get("device")) == "cuda":
        return True
    return False


# Patched method


def _traceml_tensor_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
    if not _enabled():
        return _ORIG_TENSOR_TO(self, *args, **kwargs)
    # D2D — source is already on CUDA; skip before parsing the destination.
    if self.is_cuda:
        return _ORIG_TENSOR_TO(self, *args, **kwargs)
    #  _apply traversal — model.to(device) calls tensor.to() once per
    # Parameter, inflating n_calls by the parameter count.
    if isinstance(self, torch.nn.Parameter):
        return _ORIG_TENSOR_TO(self, *args, **kwargs)
    # Destination check last — involves torch.device() parsing.
    if not _is_cuda_target(args, kwargs):
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
