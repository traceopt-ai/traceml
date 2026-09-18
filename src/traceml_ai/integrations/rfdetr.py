"""Opt-in TraceML instrumentation for RF-DETR's ``model.train()`` API."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from functools import wraps
from importlib.metadata import PackageNotFoundError, version

__all__ = ["init"]

_TraceMLCallback = None


def _warn(message):
    try:
        print(f"[TraceML] RF-DETR: {message}", file=sys.stderr)
    except Exception:
        pass


def _callback_class():
    """Keep importing this integration free of optional training imports."""
    if _TraceMLCallback is not None:
        return _TraceMLCallback
    from traceml_ai.integrations import lightning

    class TraceMLCallback(lightning.TraceMLCallback):
        """Time RF-DETR's inner model using Lightning step semantics.

        The base installs the wrapper at training start, after RF-DETR's EMA
        copy. An earlier wrapper could bind that copy to the live model.
        """

        def _forward_target(self, pl_module):
            return pl_module.model

    # Give the private cached class a readable module-level identity.
    TraceMLCallback.__name__ = "_TraceMLCallback"
    TraceMLCallback.__qualname__ = "_TraceMLCallback"
    globals()["_TraceMLCallback"] = TraceMLCallback
    return TraceMLCallback


def _validate_mode(train_config, model_config, accelerator, trainer_kwargs):
    """Reject modes whose phase measurements this adapter does not support."""
    unsupported = []
    if model_config.compile:
        unsupported.append("compiled training")
    if model_config.segmentation_head:
        unsupported.append("segmentation")
    if model_config.use_grouppose_keypoints:
        unsupported.append("keypoint training")
    # Added after RF-DETR 1.10.1; absent on the pinned eager-only revision.
    if getattr(model_config, "cuda_graphs", False):
        unsupported.append("CUDA graphs")

    accelerator = accelerator or getattr(train_config, "accelerator", "auto")
    if str(accelerator).lower() not in {"auto", "cpu", "gpu", "cuda"}:
        unsupported.append(f"accelerator={accelerator!r}")
    strategy = trainer_kwargs.get(
        "strategy", getattr(train_config, "strategy", "auto")
    )
    if isinstance(strategy, str):
        supported_strategy = strategy.lower() in {
            "auto",
            "ddp",
            "ddp_find_unused_parameters_true",
            "ddp_find_unused_parameters_false",
        }
    else:
        # Explicit DDPStrategy instances are supported by build_trainer's
        # lower-level API; other strategies need separate timing validation.
        from pytorch_lightning.strategies import DDPStrategy

        supported_strategy = (
            isinstance(strategy, DDPStrategy)
            and getattr(strategy, "_start_method", "popen") == "popen"
        )
    if not supported_strategy:
        unsupported.append(f"strategy={strategy!r}")

    if unsupported:
        raise ValueError(
            "requires eager detection on CPU/CUDA with single-device or "
            f"ordinary DDP training; unsupported: {', '.join(unsupported)}"
        )


def _install_factory(training, original):
    """Append instrumentation without replacing RF-DETR's callback stack."""
    from traceml_ai.integrations import lightning

    callback_class = _callback_class()
    factory_signature = inspect.signature(original)

    @wraps(original)
    def build_trainer(*args, **kwargs):
        bound = factory_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        enabled = (
            bound.arguments["include_training_callbacks"]
            and not lightning._traceml_disabled()
        )
        if enabled:
            try:
                _validate_mode(
                    bound.arguments["train_config"],
                    bound.arguments["model_config"],
                    bound.arguments["accelerator"],
                    bound.arguments.get("trainer_kwargs", {}),
                )
            except Exception as exc:
                _warn(f"skipping instrumentation: {exc}")
                enabled = False
        # Native factory errors must propagate, even when tracing is skipped.
        trainer = original(*args, **kwargs)
        if not enabled:
            return trainer

        try:
            device = getattr(trainer.strategy, "root_device", None)
            if getattr(device, "type", str(device).split(":")[0]) not in {
                "cpu",
                "cuda",
            }:
                raise ValueError(f"unsupported resolved device {device!r}")
            callbacks = trainer.callbacks
            trace_callbacks = [
                callback
                for callback in callbacks
                if isinstance(callback, lightning.TraceMLCallback)
            ]
            if trace_callbacks:
                if len(trace_callbacks) == 1 and isinstance(
                    trace_callbacks[0], callback_class
                ):
                    return trainer
                raise ValueError(
                    "existing generic or duplicate TraceML callbacks; "
                    "remove them to use the RF-DETR callback"
                )
            callbacks.append(callback_class())
        except Exception as exc:
            _warn(f"skipping instrumentation: {exc}")
        return trainer

    build_trainer._traceml_rfdetr_factory = True
    training.build_trainer = build_trainer


def init():
    """Enable TraceML for subsequent RF-DETR ``model.train()`` calls.

    Call once in each worker's training script, then launch with
    ``traceml run``. RF-DETR remains an optional dependency. The process-wide
    factory hook is idempotent and leaves standalone evaluation unchanged;
    per-fit model and transfer wrappers are restored by the callback.

    Supports eager detection on CPU/CUDA with single-device or torchrun DDP
    execution. The native ``rfdetr fit`` CLI does not use this factory hook.
    Returns the effective TraceML initialization configuration.
    """
    if os.environ.get("TRACEML_DISABLED") == "1":
        from traceml_ai.integrations import lightning

        return lightning.init()
    try:
        training = importlib.import_module("rfdetr.training")
    except ModuleNotFoundError as exc:
        if exc.name == "rfdetr" or (
            exc.name and not exc.name.startswith("rfdetr.")
        ):
            raise ImportError(
                "RF-DETR training dependencies are required. Install "
                "them with `pip install 'rfdetr[train]==1.10.1'`."
            ) from exc
        raise

    from traceml_ai.integrations import lightning

    config = lightning.init()
    if not config.disabled:
        from traceml_ai.instrumentation.patches.dataloader_patch import (
            require_dataloader_timing_scope,
        )

        # A skipped adapter has no callback to complete DataLoader captures.
        require_dataloader_timing_scope()
    original = getattr(training, "build_trainer", None)
    if config.disabled or getattr(original, "_traceml_rfdetr_factory", False):
        return config
    try:
        required_parameters = {
            "train_config",
            "model_config",
            "accelerator",
            "include_training_callbacks",
            "trainer_kwargs",
        }
        if not callable(original) or not required_parameters.issubset(
            inspect.signature(original).parameters
        ):
            raise ValueError("unsupported build_trainer interface")
        try:
            installed_version = version("rfdetr")
        except PackageNotFoundError:
            installed_version = "unknown"
        if installed_version != "1.10.1":
            _warn(
                f"version {installed_version} has not been validated with "
                "this integration. Tested versions are listed in "
                "docs/user_guide/integrations/rfdetr.md."
            )
        _install_factory(training, original)
    except Exception as exc:
        _warn(f"skipping instrumentation: {exc}")
    return config
