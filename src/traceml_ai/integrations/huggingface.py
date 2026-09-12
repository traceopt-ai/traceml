"""TraceML callbacks for the standard Hugging Face Trainer.

The TraceMLTrainer wrapper was intentionally removed. Use init() and register
TraceMLTrainerCallback with transformers.Trainer.
"""

import os
import sys
from functools import wraps

from traceml_ai.sdk.instrumentation import trace_step


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
    guard that aborts unfinished steps before an automatic batch-size retry.
    This mirrors the PyTorch Lightning integration's ``init()``; HF uses
    ``mode="auto"`` because ``trace_step`` drives forward/backward timing
    through the patch-gated auto-timers, whereas Lightning's callback owns that
    timing directly.
    """
    import traceml_ai as traceml

    config = traceml.init(mode="auto")
    try:
        _install_trainer_lifecycle_guard()
    except Exception as exc:
        _log_hf_error("Trainer lifecycle guard installation failed", exc)
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
        except _TraceStepAbort:
            # Defensive for context-manager implementations that propagate the
            # injected signal instead of returning False from __exit__.
            pass
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
            _abort_pending_capture_safely()

    guarded_inner_training_loop._traceml_lifecycle_guard = True
    Trainer._inner_training_loop = guarded_inner_training_loop


__all__ = ["TraceMLTrainerCallback", "init"]
