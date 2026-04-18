"""
Official TraceML manual instrumentation wrappers.

This module provides the manual counterparts to TraceML's automatic
patch-based instrumentation. The wrappers emit the same internal event names
used by the automatic path so that downstream samplers, summaries, and
diagnostics continue to work unchanged.

Design goals
------------
- Keep wrappers explicit and easy to use in custom training loops
- Preserve object identity where that matters operationally
- Refuse ambiguous or duplicate instrumentation configurations
- Remain safe for large production training workflows

Important
---------
These wrappers are intended for `manual` or `selective` initialization modes.
They should not be used for a feature whose automatic patch or hook is already
active in the current process.
"""

from __future__ import annotations

import functools
from typing import Any, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from traceml.utils.timing import TimeScope, timed_region


def _raise_duplicate_instrumentation(feature: str, reason: str) -> None:
    raise RuntimeError(
        f"TraceML cannot apply manual wrapper instrumentation for {feature} "
        f"because automatic instrumentation is already active. {reason} "
        "Disable the automatic path for this feature before using the wrapper."
    )


def _is_torch_dataloader_iterator(obj: Any) -> bool:
    """
    Best-effort detection for iterators returned by torch DataLoader.

    We intentionally keep this lightweight and non-invasive. This check is only
    used to prevent obvious duplicate wrapping when the global DataLoader patch
    is already active.
    """
    typ = type(obj)
    module = str(getattr(typ, "__module__", "") or "")
    name = str(getattr(typ, "__name__", "") or "")
    return module.startswith("torch.utils.data") and name.endswith(
        "DataLoaderIter"
    )


def _ensure_dataloader_wrapper_allowed(obj: Any) -> None:
    """
    Reject manual wrapping only for the torch DataLoader path that is already
    automatically patched.

    This intentionally still allows custom loaders / custom iterators even when
    the global torch DataLoader patch is active.
    """
    if not getattr(DataLoader, "_traceml_patched", False):
        return

    if isinstance(obj, DataLoader) or _is_torch_dataloader_iterator(obj):
        _raise_duplicate_instrumentation(
            "dataloader fetch",
            "torch DataLoader fetch timing is already patched.",
        )


def _ensure_forward_wrapper_allowed() -> None:
    if getattr(nn.Module, "_traceml_forward_patched", False):
        _raise_duplicate_instrumentation(
            "forward",
            "nn.Module.__call__ has already been patched.",
        )


def _ensure_backward_wrapper_allowed() -> None:
    if getattr(torch, "_traceml_backward_patched", False):
        _raise_duplicate_instrumentation(
            "backward",
            "torch backward entry points have already been patched.",
        )


def _ensure_optimizer_wrapper_allowed() -> None:
    if getattr(torch.optim.Optimizer, "_traceml_opt_hooks_installed", False):
        _raise_duplicate_instrumentation(
            "optimizer step",
            "global optimizer step hooks are already installed.",
        )


class _WrappedDataLoaderIterator:
    """
    Iterator proxy that times `next(...)` as TraceML dataloader fetch.

    This emits the same event name currently used by the automatic DataLoader
    patch path: `_traceml_internal:dataloader_next`.
    """

    def __init__(self, iterator: Iterator[Any]) -> None:
        self._iterator = iterator

    def __iter__(self) -> "_WrappedDataLoaderIterator":
        return self

    def __next__(self) -> Any:
        with timed_region(
            name="_traceml_internal:dataloader_next",
            scope=TimeScope.STEP,
            use_gpu=False,
        ):
            return next(self._iterator)


class _WrappedDataLoaderFetch:
    """
    Loader proxy that returns wrapped iterators.

    This is intended for manual Python training loops. It forwards unknown
    attributes to the original loader so existing user code remains natural.
    """

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def __iter__(self) -> _WrappedDataLoaderIterator:
        return _WrappedDataLoaderIterator(iter(self._loader))

    def __len__(self) -> int:
        return len(self._loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


class _WrappedBackwardHandle:
    """
    Thin proxy that times `.backward(...)` on a loss-like object.

    This wrapper is intentionally lightweight and is designed for the common
    pattern:

        loss = traceml.wrap_backward(loss)
        loss.backward()
    """

    def __init__(self, loss: Any) -> None:
        self._loss = loss

    def backward(self, *args: Any, **kwargs: Any) -> Any:
        with timed_region(
            name="_traceml_internal:backward_time",
            scope=TimeScope.STEP,
            use_gpu=True,
        ):
            return self._loss.backward(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loss, name)


def wrap_dataloader_fetch(obj: Any) -> Any:
    """
    Wrap a dataloader or iterator for step-scoped fetch timing.

    Supported inputs
    ----------------
    - loader objects that implement `__iter__`
    - iterator objects that implement `__next__`

    Notes
    -----
    - If you pass a torch DataLoader while automatic DataLoader patching is
      active, this raises to prevent duplicate instrumentation.
    - Custom iterators remain allowed even when torch DataLoader auto patching
      is active.
    """
    _ensure_dataloader_wrapper_allowed(obj)

    if hasattr(obj, "__next__"):
        return _WrappedDataLoaderIterator(obj)

    if hasattr(obj, "__iter__"):
        return _WrappedDataLoaderFetch(obj)

    raise TypeError(
        "wrap_dataloader_fetch() expects a loader or iterator object."
    )


def wrap_forward(model: nn.Module) -> nn.Module:
    """
    Wrap one model instance's `forward(...)` for step-scoped timing.

    This wrapper mutates the provided model instance in place, but only for that
    instance. It does not patch `nn.Module` globally.
    """
    _ensure_forward_wrapper_allowed()

    if not isinstance(model, nn.Module):
        raise TypeError("wrap_forward() expects an nn.Module instance.")

    if getattr(model, "_traceml_forward_instance_wrapped", False):
        return model

    original_forward = getattr(model, "forward", None)
    if original_forward is None or not callable(original_forward):
        raise TypeError("wrap_forward() requires a callable model.forward.")

    @functools.wraps(original_forward)
    def _wrapped_forward(*args: Any, **kwargs: Any) -> Any:
        with timed_region(
            name="_traceml_internal:forward_time",
            scope=TimeScope.STEP,
            use_gpu=True,
        ):
            return original_forward(*args, **kwargs)

    try:
        model.forward = _wrapped_forward  # type: ignore[method-assign]
        model._traceml_forward_instance_wrapped = True  # type: ignore[attr-defined]
        model._traceml_original_forward = original_forward  # type: ignore[attr-defined]
    except Exception as exc:
        raise RuntimeError(
            "TraceML failed to wrap model.forward() in place. "
            "This is fatal for manual forward instrumentation because the "
            "model would otherwise run with inconsistent timing behavior."
        ) from exc

    return model


def wrap_backward(loss: Any) -> Any:
    """
    Wrap a loss-like object so `.backward(...)` emits TraceML backward timing.
    """
    _ensure_backward_wrapper_allowed()

    backward = getattr(loss, "backward", None)
    if backward is None or not callable(backward):
        raise TypeError(
            "wrap_backward() expects an object with a callable backward() method."
        )

    return _WrappedBackwardHandle(loss)


def wrap_optimizer(optimizer: Any) -> Any:
    """
    Wrap one optimizer instance's `.step(...)` in place for TraceML timing.

    This preserves optimizer identity, which is important for integrations like
    `torch.cuda.amp.GradScaler.step(optimizer)` and other tooling that expects
    the real optimizer object.
    """
    _ensure_optimizer_wrapper_allowed()

    step_fn = getattr(optimizer, "step", None)
    if step_fn is None or not callable(step_fn):
        raise TypeError(
            "wrap_optimizer() expects an object with a callable step() method."
        )

    if getattr(optimizer, "_traceml_step_instance_wrapped", False):
        return optimizer

    @functools.wraps(step_fn)
    def _wrapped_step(*args: Any, **kwargs: Any) -> Any:
        with timed_region(
            name="_traceml_internal:optimizer_step",
            scope=TimeScope.STEP,
            use_gpu=True,
        ):
            return step_fn(*args, **kwargs)

    try:
        optimizer.step = _wrapped_step  # type: ignore[method-assign]
        optimizer._traceml_step_instance_wrapped = True  # type: ignore[attr-defined]
        optimizer._traceml_original_step = step_fn  # type: ignore[attr-defined]
    except Exception as exc:
        raise RuntimeError(
            "TraceML failed to wrap optimizer.step() in place. "
            "This is fatal for manual optimizer instrumentation because the "
            "optimizer would otherwise run with inconsistent timing behavior."
        ) from exc

    return optimizer


__all__ = [
    "wrap_dataloader_fetch",
    "wrap_forward",
    "wrap_backward",
    "wrap_optimizer",
]
