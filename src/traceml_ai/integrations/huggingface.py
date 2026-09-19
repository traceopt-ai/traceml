"""TraceML callbacks for the standard Hugging Face Trainer.

The TraceMLTrainer wrapper was intentionally removed. Use init() and register
TraceMLTrainerCallback with transformers.Trainer.
"""

import logging
import os
import sys
from functools import wraps

from traceml_ai.sdk.instrumentation import trace_step

logger = logging.getLogger(__name__)
_WARNED_CAPABILITIES: set[str] = set()


def _traceml_disabled() -> bool:
    """
    Read the TRACEML_DISABLED kill switch dynamically.

    Read per-call rather than captured at import so toggling the env var
    after import (notebooks, tests) is honored, matching ``trace_step``.
    """
    return os.environ.get("TRACEML_DISABLED") == "1"


try:
    from transformers import TrainerCallback

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    TrainerCallback = object  # Fallback for type hinting


def init():
    """
    Initialize TraceML for Hugging Face ``Trainer`` runs.

    Call once before constructing the ``Trainer``, then register
    ``TraceMLTrainerCallback``. ``init()`` makes TraceML's process-wide
    instrumentation explicit: PyTorch ``DataLoader`` fetch timing, the H2D
    ``Tensor.to`` patch, and the forward/backward/optimizer auto-timers that
    ``trace_step`` arms inside each bracketed step.

    The callback is a per-step bracket and cannot install these process-wide
    patches on its own; the auto-timers it arms are no-ops unless the matching
    patch is installed. ``init()`` is the recommended entry point so the
    DataLoader fetch patch in particular is installed deterministically rather
    than relying on import order. It also installs the narrow Trainer lifecycle
    guard that aborts unfinished steps before an automatic batch-size retry,
    plus the collection hook needed to observe Accelerate's pre-callback H2D
    transfers. This mirrors the PyTorch Lightning integration's ``init()``; HF
    uses ``mode="auto"`` because ``trace_step`` drives forward/backward timing
    through the patch-gated auto-timers, whereas Lightning's callback owns that
    timing directly.
    """
    import traceml_ai as traceml

    config = traceml.init(mode="auto")
    try:
        _install_trainer_lifecycle_guard()
    except Exception as exc:
        _log_hf_error("Trainer lifecycle guard installation failed", exc)
    try:
        _install_batch_collection_h2d_timing()
    except Exception as exc:
        _log_hf_error("Batch collection H2D timing installation failed", exc)
    return config


def _log_hf_error(message: str, exc: Exception) -> None:
    """
    Log TraceML HF callback failures without interrupting training.

    Mirrors the Lightning integration's error helper. Uses the shared file
    logger when the launcher has configured it, falling back to stderr so
    direct callback users still see the signal.
    """
    try:
        from traceml_ai.loggers.error_log import get_error_logger

        get_error_logger("HuggingFaceIntegration").exception(
            "[TraceML] %s", message
        )
    except Exception:
        pass

    print(f"[TraceML] {message}: {exc}", file=sys.stderr)


def _warn_hf_once(key: str, message: str, *args) -> None:
    """Emit one actionable integration warning without interrupting training."""
    if key in _WARNED_CAPABILITIES:
        return
    _WARNED_CAPABILITIES.add(key)
    logger.warning("[TraceML] " + message, *args)


class _TraceStepAbort(RuntimeError):
    """Internal signal used to unwind an unfinished ``trace_step``."""


class TraceMLTrainerCallback(TrainerCallback if HAS_TRANSFORMERS else object):
    """
    Hugging Face Trainer integration for TraceML.

    Register with ``Trainer(..., callbacks=[TraceMLTrainerCallback()])``.

    The callback brackets TraceML's ``trace_step`` context manager: it opens
    ``trace_step`` in ``on_step_begin`` and completes it in ``on_step_end``.
    The lifecycle guard installed by :func:`init` aborts an open context when
    the Trainer attempt exits early. ``trace_step`` owns the step memory
    tracker, step counter, auto-timers, and capture publication.

    One completed TraceML step corresponds to one HF accumulation/update
    boundary (``on_step_end``), including when AMP skips the parameter update.
    ``on_substep_end`` does not advance the counter: forward and backward
    events from the actual micro-batches in the group fold into one step,
    including a shorter final group. Optimizer events describe calls that
    actually run; their count does not drive TraceML's step counter.

    TraceML step IDs remain process-local, so their increments match HF's
    ``global_step`` increments during recorded, completed groups; their
    absolute values need not match after checkpoint resume. See the HF
    integration docs for the input-timing limitations and lifecycle behavior.
    """

    def __init__(self) -> None:
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "TraceMLTrainerCallback requires the Hugging Face "
                "integration. Install it with "
                "`pip install 'traceml-ai[hf]'`."
            )
        super().__init__()
        self._step_cm = None
        self._owns_run = True

    def _set_run_owner(self, owns_run: bool) -> None:
        """Select one TraceML callback when duplicate instances are present."""
        if not owns_run:
            self._abort_step_cm_safely()
        self._owns_run = owns_run

    def _complete_step_cm_safely(self) -> None:
        """Complete and publish the currently open TraceML step."""
        cm = self._step_cm
        if cm is None:
            return
        self._step_cm = None
        try:
            cm.__exit__(None, None, None)
        except Exception as exc:
            _log_hf_error("trace_step exit failed", exc)

    def _abort_step_cm_safely(self) -> None:
        """Unwind an unfinished TraceML step without publishing it."""
        cm = self._step_cm
        if cm is None:
            return
        self._step_cm = None
        abort = _TraceStepAbort("Hugging Face training step did not complete")
        try:
            cm.__exit__(type(abort), abort, None)
        except Exception as exc:
            _log_hf_error("trace_step abort failed", exc)

    def on_train_begin(self, args, state, control, **kwargs):
        # A reused callback must never carry an unfinished prior run forward.
        self._abort_step_cm_safely()
        if _traceml_disabled() or not self._owns_run:
            return

        # Check instrumentation once per train() call, after user setup.
        # Missing streams should warn without interrupting training.
        try:
            from traceml_ai.integrations._capability import (
                warn_if_missing_streams,
            )

            warn_if_missing_streams(
                "HuggingFace TraceMLTrainerCallback",
                {"dataloader_fetch", "forward", "backward", "h2d"},
            )
        except Exception:
            pass

    def on_step_begin(self, args, state, control, **kwargs):
        # If the prior boundary did not close, discard it before starting the
        # next group. The lifecycle guard normally handles this on exceptions.
        self._abort_step_cm_safely()
        if _traceml_disabled() or not self._owns_run:
            return

        model = kwargs.get("model")
        if model is None:
            # Defensive: standard Trainer always passes the model, but a None
            # would raise inside StepMemoryTracker, costing a lost step and
            # log noise. Skip bracketing this step instead.
            return

        try:
            self._step_cm = trace_step(model)
            self._step_cm.__enter__()
        except Exception as exc:
            self._step_cm = None
            _log_hf_error("trace_step enter failed", exc)

    def on_step_end(self, args, state, control, **kwargs):
        if _traceml_disabled() or not self._owns_run:
            self._abort_step_cm_safely()
            return
        self._complete_step_cm_safely()

    def on_train_end(self, args, state, control, **kwargs):
        self._abort_step_cm_safely()


def _traceml_callbacks(trainer) -> list[TraceMLTrainerCallback]:
    """Return TraceML callbacks registered on one Trainer instance."""
    handler = getattr(trainer, "callback_handler", None)
    callbacks = getattr(handler, "callbacks", ())
    return [
        callback
        for callback in callbacks
        if isinstance(callback, TraceMLTrainerCallback)
    ]


def _abort_pending_capture_safely() -> None:
    """Discard step events left outside an open callback context."""
    try:
        from traceml_ai.instrumentation.step_events import (
            abort_step_capture,
            begin_step_capture,
        )

        abort_step_capture(begin_step_capture())
    except Exception as exc:
        _log_hf_error("pending step capture abort failed", exc)


def _install_batch_collection_h2d_timing() -> None:
    """Include Trainer-prepared input transfers in the coming TraceML step.

    Accelerate moves each returned training batch to the device inside the
    ``next()`` calls made by ``Trainer.get_batch_samples``. Those transfers
    occur before ``on_step_begin`` opens the main ``trace_step`` envelope.
    Arming the H2D patch here gives each transfer a matching short step-time
    segment while leaving the surrounding DataLoader fetch in Input Wait.

    The events remain in the active pending capture. ``on_step_end`` still
    owns the only counter advance and publication for the accumulation group.
    """
    if not HAS_TRANSFORMERS:
        return

    from transformers import Trainer

    original = getattr(Trainer, "get_batch_samples", None)
    if not callable(original):
        _warn_hf_once(
            "missing-get-batch-samples",
            "Hugging Face pre-step H2D timing requires transformers>=4.46; "
            "training will continue without that signal.",
        )
        return
    if getattr(original, "_traceml_batch_collection_h2d_timing", False):
        return

    @wraps(original)
    def timed_get_batch_samples(trainer, *args, **kwargs):
        if _traceml_disabled() or not _traceml_callbacks(trainer):
            return original(trainer, *args, **kwargs)

        from traceml_ai.instrumentation.patches.h2d_auto_timer_patch import (
            h2d_auto_timer,
        )

        with h2d_auto_timer(include_step_time=True):
            return original(trainer, *args, **kwargs)

    timed_get_batch_samples._traceml_batch_collection_h2d_timing = True
    Trainer.get_batch_samples = timed_get_batch_samples


def _warn_if_batch_collection_h2d_is_bypassed(trainer) -> None:
    """Warn when a Trainer override bypasses the installed collection hook."""
    method = getattr(trainer, "get_batch_samples", None)
    function = getattr(method, "__func__", method)
    if not callable(function) or getattr(
        function,
        "_traceml_batch_collection_h2d_timing",
        False,
    ):
        return

    trainer_type = type(trainer)
    name = f"{trainer_type.__module__}.{trainer_type.__qualname__}"
    _warn_hf_once(
        f"get-batch-samples-override:{name}",
        "%s overrides get_batch_samples, so TraceML cannot guarantee "
        "pre-step H2D timing for this Trainer; training will continue.",
        name,
    )


def _install_trainer_lifecycle_guard() -> None:
    """Install failure cleanup inside HF's per-attempt retry boundary."""
    if not HAS_TRANSFORMERS:
        return

    from transformers import Trainer

    original = Trainer._inner_training_loop
    if getattr(original, "_traceml_lifecycle_guard", False):
        return

    @wraps(original)
    def guarded_inner_training_loop(trainer, *args, **kwargs):
        callbacks = _traceml_callbacks(trainer)
        if not callbacks:
            return original(trainer, *args, **kwargs)

        _warn_if_batch_collection_h2d_is_bypassed(trainer)
        owner = callbacks[0]
        for callback in callbacks:
            callback._set_run_owner(callback is owner)

        # Start every Trainer attempt with a clean capture. This also handles
        # failures that happen while fetching inputs, before on_step_begin.
        _abort_pending_capture_safely()
        try:
            return original(trainer, *args, **kwargs)
        finally:
            # This runs before Accelerate handles an OOM and retries the same
            # inner loop, so a failed attempt cannot leak into the next one.
            for callback in callbacks:
                callback._abort_step_cm_safely()
                # Ownership applies only to this attempt, not later reuse.
                callback._set_run_owner(True)
            _abort_pending_capture_safely()

    guarded_inner_training_loop._traceml_lifecycle_guard = True
    Trainer._inner_training_loop = guarded_inner_training_loop


__all__ = ["TraceMLTrainerCallback", "init"]
