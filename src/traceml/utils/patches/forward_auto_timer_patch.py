import torch.nn as nn
import threading
from traceml.utils.timing import timed_region


_TLS = threading.local()
_ORIG_MODULE_CALL = nn.Module.__call__


def _enabled() -> bool:
    # Set this flag True only inside trace_step
    return bool(getattr(_TLS, "_traceml_forward_enabled", False))


def _depth() -> int:
    return int(getattr(_TLS, "_traceml_forward_depth", 0))


def _set_depth(v: int) -> None:
    setattr(_TLS, "_traceml_forward_depth", v)


def _traceml_module_call(self: nn.Module, *args, **kwargs):
    if not _enabled():
        return _ORIG_MODULE_CALL(self, *args, **kwargs)

    # Only time the OUTERMOST forward to avoid submodule spam
    if _depth() > 0:
        return _ORIG_MODULE_CALL(self, *args, **kwargs)

    _set_depth(_depth() + 1)
    try:
        with timed_region("_traceml_internal:forward_time", scope="step", use_gpu=True):
            return _ORIG_MODULE_CALL(self, *args, **kwargs)
    finally:
        _set_depth(_depth() - 1)


def patch_forward() -> None:
    """Patch nn.Module.__call__ once."""
    if getattr(nn.Module, "_traceml_forward_patched", False):
        return
    nn.Module.__call__ = _traceml_module_call  # type: ignore[assignment]
    nn.Module._traceml_forward_patched = True


class forward_auto_timer:
    """
    Context manager that enables forward timing during its scope.
    Assumes patch_forward() has been called once at startup/runtime init.
    """

    def __enter__(self):
        _TLS._traceml_forward_enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        _TLS._traceml_forward_enabled = False
        _TLS._traceml_forward_depth = 0
        return False
