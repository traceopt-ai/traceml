import threading
from typing import Any

import torch

from traceml.utils.timing import timed_region

_OPT_TLS = threading.local()
_ORIG_OPT_STEP = torch.optim.Optimizer.step
_ORIG_SCALER_STEP = torch.cuda.amp.GradScaler.step


def _enabled() -> bool:
    return bool(getattr(_OPT_TLS, "_traceml_opt_enabled", False))


def _depth() -> int:
    return int(getattr(_OPT_TLS, "_traceml_opt_depth", 0))


def _set_depth(v: int) -> None:
    setattr(_OPT_TLS, "_traceml_opt_depth", v)


def _traceml_optimizer_step(self: torch.optim.Optimizer, *args: Any, **kwargs: Any) -> Any:
    if not _enabled():
        return _ORIG_OPT_STEP(self, *args, **kwargs)

    # Only time OUTERMOST step() in case something nests/calls step internally
    if _depth() > 0:
        return _ORIG_OPT_STEP(self, *args, **kwargs)

    _set_depth(_depth() + 1)
    try:
        with timed_region("_traceml_internal:optimizer_step", scope="step", use_gpu=True):
            return _ORIG_OPT_STEP(self, *args, **kwargs)
    finally:
        _set_depth(_depth() - 1)


def _traceml_scaler_step(self, optimizer, *args: Any, **kwargs: Any) -> Any:
    if not _enabled():
        return _ORIG_SCALER_STEP(self, optimizer, *args, **kwargs)

    # Outermost semantic optimizer step
    if _depth() > 0:
        return _ORIG_SCALER_STEP(self, optimizer, *args, **kwargs)

    _set_depth(_depth() + 1)
    try:
        with timed_region("_traceml_internal:optimizer_step", scope="step", use_gpu=True):
            return _ORIG_SCALER_STEP(self, optimizer, *args, **kwargs)
    finally:
        _set_depth(_depth() - 1)


def patch_optimizer() -> None:
    """Patch Optimizer.step once. Safe to call multiple times."""
    if getattr(torch.optim.Optimizer, "_traceml_opt_patched", False):
        return

    torch.optim.Optimizer.step = _traceml_optimizer_step
    torch.cuda.amp.GradScaler.step = _traceml_scaler_step
    torch.optim.Optimizer._traceml_opt_patched = True


class optimizer_auto_timer:
    """
    Enables optimizer step timing during its scope.
    Assumes patch_optimizer() has been called once at startup/runtime init.
    """

    def __enter__(self):
        _OPT_TLS._traceml_opt_enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        _OPT_TLS._traceml_opt_enabled = False
        _OPT_TLS._traceml_opt_depth = 0
        return False
