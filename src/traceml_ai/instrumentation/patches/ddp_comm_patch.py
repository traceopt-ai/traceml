"""
DDP gradient-sync comm-hook auto-patch.

Patches ``DistributedDataParallel.forward`` once at ``traceml.init()`` time so
each DDP instance gets the TraceML comm hook installed on its first forward.
Mirrors the forward / h2d auto-patches: a single class-level patch installed by
``patch_ddp_comm()``.

Why first-forward (lazy), not ``__init__``
------------------------------------------
- A class-level patch catches DDP instances regardless of construction order
  (even ones an integration created before ``init()`` ran).
- Installing at first forward (inside the training loop) rather than at
  construction lets a user register their own comm hook first; if they did,
  ``install_ddp_comm_hook`` fail-opens instead of preempting it.
- It removes the old fragility where auto-install keyed on the model object
  passed to ``trace_step`` and silently skipped when an integration passed the
  unwrapped model (e.g. the HuggingFace Trainer callback).

Wire name: ``_traceml_comm:ddp_grad_sync`` (emitted by the installed hook).
"""

from typing import Any

import torch.nn as nn

from traceml_ai.instrumentation.hooks.ddp_comm_hook import (
    ensure_ddp_comm_hook_installed,
)

_ORIG_DDP_FORWARD = None


def _traceml_ddp_forward(self: nn.Module, *args: Any, **kwargs: Any) -> Any:
    # One-shot lazy install on first forward (covers success and fail-open so
    # a pre-registered user hook does not retry / spam stderr every step).
    if not getattr(self, "_traceml_ddp_comm_patch_attempted", False):
        self._traceml_ddp_comm_patch_attempted = True
        ensure_ddp_comm_hook_installed(self)
    return _ORIG_DDP_FORWARD(self, *args, **kwargs)


def patch_ddp_comm() -> None:
    """Patch ``DistributedDataParallel.forward`` once. Safe to call repeatedly."""
    from torch.nn.parallel import DistributedDataParallel as _DDP

    if getattr(_DDP, "_traceml_ddp_comm_patched", False):
        return

    global _ORIG_DDP_FORWARD
    _ORIG_DDP_FORWARD = _DDP.forward
    _DDP.forward = _traceml_ddp_forward  # type: ignore[assignment]
    _DDP._traceml_ddp_comm_patched = True  # type: ignore[attr-defined]
