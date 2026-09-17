"""Opt-in TraceML instrumentation for RF-DETR's ``model.train()`` API."""

from __future__ import annotations

import importlib
import inspect
import os
import warnings
from functools import wraps
from importlib.metadata import PackageNotFoundError, version

__all__ = ["init"]

_TraceMLCallback = None


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
    if getattr(model_config, "compile", False):
        unsupported.append("compiled training")
    if getattr(model_config, "segmentation_head", False):
        unsupported.append("segmentation")
    if getattr(model_config, "use_grouppose_keypoints", False):
        unsupported.append("keypoint training")
    if any(
        getattr(config, "cuda_graphs", False)
        for config in (train_config, model_config)
    ):
        unsupported.append("CUDA graphs")

    accelerator = accelerator or getattr(train_config, "accelerator", "auto")
    if str(accelerator).lower() not in {"auto", "cpu", "gpu", "cuda"}:
        unsupported.append(f"accelerator={accelerator!r}")
    strategy = trainer_kwargs.get(
        "strategy", getattr(train_config, "strategy", "auto")
    )
    if isinstance(strategy, str):
        supported_strategy = strategy.lower() in {"auto", "ddp"}
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
            "TraceML's RF-DETR integration supports eager detection on CPU "
            "or CUDA with single-device or ordinary DDP training; "
            f"unsupported: {', '.join(unsupported)}. Use a supported "
            "configuration or remove traceml_rfdetr.init() to train without "
            "this integration. Launch DDP through traceml run."
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
            _validate_mode(
                bound.arguments["train_config"],
                bound.arguments["model_config"],
                bound.arguments["accelerator"],
                bound.arguments.get("trainer_kwargs", {}),
            )
        trainer = original(*args, **kwargs)
        if not enabled:
            return trainer

        device = getattr(trainer.strategy, "root_device", None)
        if getattr(device, "type", str(device).split(":")[0]) not in {
            "cpu",
            "cuda",
        }:
            raise ValueError(
                "TraceML's RF-DETR integration requires a CPU or CUDA "
                f"device; RF-DETR selected {device!r}. Set accelerator "
                "explicitly or remove traceml_rfdetr.init()."
            )
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
                "RF-DETR already has a TraceML callback. Remove the generic "
                "Lightning callback or duplicate callbacks; "
                "traceml_rfdetr.init() installs the RF-DETR callback."
            )
        callbacks.append(callback_class())
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
                "`pip install 'rfdetr[train]==1.10.1'`."
            ) from exc
        raise

    original = getattr(training, "build_trainer", None)
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
        raise RuntimeError(
            "Unsupported RF-DETR training interface. This integration "
            "requires the Lightning build_trainer API from rfdetr==1.10.1."
        )

    from traceml_ai.integrations import lightning

    config = lightning.init()
    if config.disabled or getattr(original, "_traceml_rfdetr_factory", False):
        return config
    try:
        installed_version = version("rfdetr")
    except PackageNotFoundError:
        installed_version = "unknown"
    if installed_version != "1.10.1":
        warnings.warn(
            f"RF-DETR {installed_version} has not been qualified with this "
            "TraceML adapter. Use rfdetr==1.10.1 for the documented example; "
            "development support is limited to the documented source revision.",
            UserWarning,
            stacklevel=2,
        )
    _install_factory(training, original)
    return config
