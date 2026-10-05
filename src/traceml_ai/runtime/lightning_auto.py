"""Lazy automatic attachment for standard Lightning ``Trainer.fit`` runs."""

from __future__ import annotations

import sys
from functools import wraps
from importlib import import_module

from traceml_ai.runtime import _import_hook

_WARNINGS = set()


def _warn_once(key, message):
    if key in _WARNINGS:
        return
    _WARNINGS.add(key)
    try:
        print(f"[TraceML] {message}", file=sys.stderr)
    except Exception:
        pass


def _is_compiled(module) -> bool:
    if module is None:
        return False
    if getattr(module, "_compiler_ctx", None) is not None:
        return True
    try:
        from torch._dynamo import OptimizedModule

        return any(
            isinstance(child, OptimizedModule) for child in module.modules()
        )
    except Exception:
        return False


def _is_rfdetr(module) -> bool:
    return module is not None and any(
        cls.__module__ == "rfdetr" or cls.__module__.startswith("rfdetr.")
        for cls in type(module).__mro__
    )


def _supported_strategy(strategies, strategy) -> bool:
    deepspeed = getattr(strategies, "DeepSpeedStrategy", ())
    if isinstance(strategy, deepspeed):
        return False
    return isinstance(strategy, strategies.SingleDeviceStrategy) or (
        isinstance(strategy, strategies.DDPStrategy)
        and getattr(strategy, "_start_method", None) == "popen"
    )


def _prepare_callback(trainer) -> None:
    from traceml_ai.integrations import lightning
    from traceml_ai.sdk.initial import get_init_config

    if lightning._traceml_disabled():
        return
    config = get_init_config()
    if (config is not None and config.disabled) or getattr(
        trainer, "_traceml_auto_skip", False
    ):
        return

    callbacks = trainer.callbacks
    existing = any(
        isinstance(callback, lightning.TraceMLCallback)
        for callback in callbacks
    )
    model = getattr(trainer, "lightning_module", None)
    if _is_rfdetr(model):
        if not existing:
            _warn_once(
                "rfdetr",
                "RF-DETR uses its dedicated TraceML adapter; call "
                "traceml_ai.integrations.rfdetr.init() before training. "
                "Generic Lightning attachment was skipped.",
            )
        return
    if _is_compiled(model):
        if not existing:
            _warn_once(
                "compiled",
                "Lightning automatic instrumentation does not cover "
                "torch.compile; keeping existing callbacks unchanged.",
            )
        return
    if config is not None and not (
        config.mode == "selective"
        and config.patch_dataloader
        and config.patch_h2d
        and not config.patch_forward
        and not config.patch_backward
    ):
        _warn_once(
            "configuration",
            "Lightning automatic instrumentation found an incompatible TraceML "
            "init configuration; keeping the user configuration and callbacks "
            "unchanged. Use the Lightning integration's init() and callback "
            "for manual setup.",
        )
        return

    namespace = (
        "pytorch_lightning"
        if any(
            cls.__module__.startswith("pytorch_lightning.")
            for cls in type(trainer).__mro__
        )
        else "lightning.pytorch"
    )
    strategies = import_module(f"{namespace}.strategies")
    strategy = trainer.strategy
    supported = _supported_strategy(strategies, strategy)
    if not supported or getattr(strategy.root_device, "type", None) not in {
        "cpu",
        "cuda",
    }:
        if not existing:
            _warn_once(
                "strategy",
                "Lightning automatic instrumentation supports CPU/CUDA "
                "single-device and ordinary DDP runs; keeping existing "
                "callbacks unchanged for this strategy or launch mode.",
            )
        return

    if not existing:
        connector = import_module(
            f"{namespace}.trainer.connectors.callback_connector"
        )
        # Register first: init arms process-wide timing patches, which must not
        # be left active if automatic callback attachment cannot complete.
        trainer.callbacks = connector._CallbackConnector._reorder_callbacks(
            [*callbacks, lightning.TraceMLCallback()]
        )
    if lightning.init().disabled:
        return
    trainer._traceml_auto_status = (
        "using existing TraceML callback"
        if existing
        else "TraceML callback added automatically"
    )


def _activate(module):
    try:
        connector = module._CallbackConnector
        original = connector._attach_model_callbacks
        if getattr(original, "_traceml_auto_attach", False):
            return

        @wraps(original)
        def attach(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if getattr(self.trainer.state.fn, "value", None) == "fit":
                try:
                    _prepare_callback(self.trainer)
                except Exception as exc:
                    _warn_once(
                        "attachment",
                        f"Lightning automatic attachment unavailable: {exc}",
                    )
            return result

        attach._traceml_auto_attach = True
        connector._attach_model_callbacks = attach
    except Exception as exc:
        _warn_once(
            "attachment",
            f"Lightning automatic attachment unavailable: {exc}",
        )


def install():
    return _import_hook.install(
        (
            "lightning.pytorch.trainer.trainer",
            "pytorch_lightning.trainer.trainer",
        ),
        _activate,
    )


uninstall = _import_hook.uninstall
