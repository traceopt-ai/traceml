import threading
from typing import Any

import torch

from traceml_ai.utils.timing import timed_region

_BACKWARD_TLS = threading.local()

_ORIG_TENSOR_BACKWARD = torch.Tensor.backward
_ORIG_AUTOGRAD_BACKWARD = torch.autograd.backward


def _enabled() -> bool:
    return bool(getattr(_BACKWARD_TLS, "_traceml_backward_enabled", False))


def _depth() -> int:
    return int(getattr(_BACKWARD_TLS, "_traceml_backward_depth", 0))


def _set_depth(v: int) -> None:
    setattr(_BACKWARD_TLS, "_traceml_backward_depth", v)


def _traceml_tensor_backward(
    self: torch.Tensor, *args: Any, **kwargs: Any
) -> Any:
    if not _enabled():
        return _ORIG_TENSOR_BACKWARD(self, *args, **kwargs)

    # Optional: only time the OUTERMOST backward call (avoid double timing)
    if _depth() > 0:
        return _ORIG_TENSOR_BACKWARD(self, *args, **kwargs)

    _set_depth(_depth() + 1)
    try:
        with timed_region(
            "_traceml_internal:backward_time", scope="step", use_gpu=True
        ):
            return _ORIG_TENSOR_BACKWARD(self, *args, **kwargs)
    finally:
        _set_depth(_depth() - 1)


def _traceml_autograd_backward(*args: Any, **kwargs: Any) -> Any:
    if not _enabled():
        return _ORIG_AUTOGRAD_BACKWARD(*args, **kwargs)

    # Optional: only time the OUTERMOST backward call (avoid double timing)
    if _depth() > 0:
        return _ORIG_AUTOGRAD_BACKWARD(*args, **kwargs)

    _set_depth(_depth() + 1)
    try:
        with timed_region(
            "_traceml_internal:backward_time", scope="step", use_gpu=True
        ):
            return _ORIG_AUTOGRAD_BACKWARD(*args, **kwargs)
    finally:
        _set_depth(_depth() - 1)


def patch_backward() -> None:
    """Patch backward entry points once. Safe to call multiple times."""
    if getattr(torch, "_traceml_backward_patched", False):
        return

    torch.Tensor.backward = _traceml_tensor_backward  # type: ignore[assignment]
    torch.autograd.backward = _traceml_autograd_backward  # type: ignore[assignment]

    torch._traceml_backward_patched = True  # type: ignore[attr-defined]


class backward_auto_timer:
    """
    Enables backward timing during its scope.

    The previous thread-local state is restored on exit so nested tracing
    contexts preserve the outer context's timing state.

    Assumes patch_backward() has been called once at startup/runtime init.
    """

    def __init__(self):
        self._prev_enabled = False
        self._prev_depth = 0

    def __enter__(self):
        self._prev_enabled = _enabled()
        self._prev_depth = _depth()

        _BACKWARD_TLS._traceml_backward_enabled = True
        _BACKWARD_TLS._traceml_backward_depth = 0
        return self

    def __exit__(self, exc_type, exc, tb):
        _BACKWARD_TLS._traceml_backward_enabled = self._prev_enabled
        _BACKWARD_TLS._traceml_backward_depth = self._prev_depth
        return False
