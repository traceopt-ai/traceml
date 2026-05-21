import threading

import torch.nn as nn

from traceml.utils.timing import timed_region

_TLS = threading.local()
_ORIG_MODULE_CALL = nn.Module.__call__


def _enabled() -> bool:
    # Set this flag True only inside trace_step
    return bool(getattr(_TLS, "_traceml_forward_enabled", False))


def _target_ids() -> set[int]:
    return getattr(_TLS, "_traceml_forward_target_ids", set())


def _is_target(module: nn.Module) -> bool:
    ids = _target_ids()
    return not ids or id(module) in ids


def _depth() -> int:
    return int(getattr(_TLS, "_traceml_forward_depth", 0))


def _set_depth(v: int) -> None:
    setattr(_TLS, "_traceml_forward_depth", v)


def _collect_forward_target_ids(model: nn.Module | None) -> set[int]:
    if model is None:
        return set()

    targets = {id(model)}

    ddp_module = getattr(model, "module", None)
    if isinstance(ddp_module, nn.Module):
        targets.add(id(ddp_module))

    fsdp_module = getattr(model, "_fsdp_wrapped_module", None)
    if isinstance(fsdp_module, nn.Module):
        targets.add(id(fsdp_module))

    return targets


def _traceml_module_call(self: nn.Module, *args, **kwargs):
    if not _enabled() or not _is_target(self):
        return _ORIG_MODULE_CALL(self, *args, **kwargs)

    # Only time the OUTERMOST forward to avoid submodule spam
    if _depth() > 0:
        return _ORIG_MODULE_CALL(self, *args, **kwargs)

    _set_depth(_depth() + 1)
    try:
        with timed_region(
            "_traceml_internal:forward_time", scope="step", use_gpu=True
        ):
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

    def __init__(self, model: nn.Module | None = None):
        self.model = model
        self._prev_enabled = False
        self._prev_depth = 0
        self._prev_target_ids = set()

    def __enter__(self):
        self._prev_enabled = _enabled()
        self._prev_depth = _depth()
        self._prev_target_ids = _target_ids()

        _TLS._traceml_forward_enabled = True
        _TLS._traceml_forward_depth = 0
        _TLS._traceml_forward_target_ids = _collect_forward_target_ids(
            self.model
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        _TLS._traceml_forward_enabled = self._prev_enabled
        _TLS._traceml_forward_depth = self._prev_depth
        _TLS._traceml_forward_target_ids = self._prev_target_ids
        return False
