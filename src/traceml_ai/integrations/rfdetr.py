"""TraceML instrumentation for RF-DETR's ``model.train()`` API."""

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


def _init_timing():
    """Enable the existing Lightning timings and scope input to training."""
    from traceml_ai.integrations import lightning

    config = lightning.init()
    if not config.disabled:
        from traceml_ai.instrumentation.patches.dataloader_patch import (
            require_dataloader_timing_scope,
        )

        # Skipped adapters have no callback to complete DataLoader captures.
        require_dataloader_timing_scope()
    return config


def _auto_ready(
    arguments,
    trainer,
    callback_class,
    *,
    automatic,
    warn_incompatible_config,
):
    """Check whether this Trainer can use the RF-DETR callback."""
    from traceml_ai.integrations import lightning

    if (
        not arguments["include_training_callbacks"]
        or lightning._traceml_disabled()
    ):
        return False, []

    _validate_mode(
        arguments["train_config"],
        arguments["model_config"],
        arguments["accelerator"],
        arguments.get("trainer_kwargs", {}),
    )
    device = getattr(trainer.strategy, "root_device", None)
    if getattr(device, "type", str(device).split(":")[0]) not in {
        "cpu",
        "cuda",
    }:
        raise ValueError(f"unsupported resolved device {device!r}")

    if automatic:
        from traceml_ai.sdk.initial import get_init_config

        config = get_init_config()
        if config is not None and config.disabled:
            return False, []
        if not lightning._auto_config_is_compatible(config):
            warn_incompatible_config()
            return False, []
        if _init_timing().disabled:
            return False, []

    trace_callbacks = [
        callback
        for callback in trainer.callbacks
        if isinstance(callback, lightning.TraceMLCallback)
    ]
    if trace_callbacks and (
        len(trace_callbacks) != 1
        or not isinstance(trace_callbacks[0], callback_class)
    ):
        raise ValueError(
            "existing generic or duplicate TraceML callbacks; "
            "remove them to use the RF-DETR callback"
        )
    return True, trace_callbacks


def _install_factory(training, original, *, automatic=False):
    """Append instrumentation without replacing RF-DETR's callback stack."""
    callback_class = _callback_class()
    factory_signature = inspect.signature(original)
    warned_configuration = False
    warned_version = False

    def warn_incompatible_config():
        nonlocal warned_configuration
        if warned_configuration:
            return
        _warn(
            "automatic instrumentation found an incompatible TraceML init "
            "configuration; keeping the user configuration and callbacks "
            "unchanged."
        )
        warned_configuration = True

    def warn_unvalidated_version(trainer):
        nonlocal warned_version
        if warned_version or not getattr(trainer, "is_global_zero", True):
            return
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
        warned_version = True

    @wraps(original)
    def build_trainer(*args, **kwargs):
        try:
            bound = factory_signature.bind(*args, **kwargs)
        except TypeError:
            # Preserve the native factory's argument errors.
            return original(*args, **kwargs)
        bound.apply_defaults()
        # Native factory errors must propagate, even when tracing is skipped.
        trainer = original(*args, **kwargs)
        # The RF-DETR adapter deliberately owns this Trainer. In particular,
        # an unsupported RF-DETR mode must not fall through to the generic
        # automatic Lightning integration later during ``fit``.
        try:
            trainer._traceml_auto_skip = True
        except Exception:
            pass

        try:
            ready, trace_callbacks = _auto_ready(
                bound.arguments,
                trainer,
                callback_class,
                automatic=build_trainer._traceml_rfdetr_auto,
                warn_incompatible_config=warn_incompatible_config,
            )
            if not ready:
                return trainer
            warn_unvalidated_version(trainer)
            callback_already_present = bool(trace_callbacks)
            if not trace_callbacks:
                from pytorch_lightning.trainer.connectors.callback_connector import (
                    _CallbackConnector,
                )

                trainer.callbacks = _CallbackConnector._reorder_callbacks(
                    [*trainer.callbacks, callback_class()]
                )
            if build_trainer._traceml_rfdetr_auto:
                trainer._traceml_auto_framework = "RF-DETR"
                trainer._traceml_auto_status = (
                    "using existing TraceML callback"
                    if callback_already_present
                    else "TraceML callback added automatically"
                )
        except Exception as exc:
            _warn(f"skipping instrumentation: {exc}")
        return trainer

    build_trainer._traceml_rfdetr_factory = True
    build_trainer._traceml_rfdetr_auto = automatic
    training.build_trainer = build_trainer


def _enable_factory(training, *, automatic):
    """Install one factory wrapper shared by manual and launcher activation."""
    original = getattr(training, "build_trainer", None)
    if getattr(original, "_traceml_rfdetr_factory", False):
        if automatic:
            original._traceml_rfdetr_auto = True
        return
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
    _install_factory(training, original, automatic=automatic)


def _enable_auto_attach(training):
    """Observe RF-DETR without initializing timing until Trainer creation."""
    if os.environ.get("TRACEML_DISABLED") != "1":
        _enable_factory(training, automatic=True)


def init():
    """Enable TraceML for subsequent RF-DETR ``model.train()`` calls.

    ``traceml run`` activates this adapter automatically. For manual setup,
    call once before training in each worker. RF-DETR remains optional. The
    process-wide factory hook is idempotent and leaves standalone evaluation unchanged;
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

    config = _init_timing()
    if config.disabled:
        return config
    try:
        _enable_factory(training, automatic=False)
    except Exception as exc:
        _warn(f"skipping instrumentation: {exc}")
    return config
