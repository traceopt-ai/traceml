"""TraceML callbacks for the standard Hugging Face Trainer.

The TraceMLTrainer wrapper was intentionally removed. Use init() and register
TraceMLTrainerCallback with transformers.Trainer.
"""

import logging
import os
import sys
import threading
from functools import wraps

from traceml_ai.instrumentation.patches.dataloader_patch import (
    dataloader_timing_scope,
)
from traceml_ai.instrumentation.patches.h2d_auto_timer_patch import (
    h2d_auto_timer,
)
from traceml_ai.sdk.instrumentation import trace_step

logger = logging.getLogger(__name__)
_WARNED_CAPABILITIES: set[str] = set()


class _TrainingAttemptState(threading.local):
    """Per-thread state shared by the narrow Trainer integration hooks."""

    def __init__(self) -> None:
        self.active = False
        self.suppress_next_input_group = False
        self.omit_current_group = False


_TRAINING_ATTEMPT_STATE = _TrainingAttemptState()


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
    plus the collection hook needed to isolate training Input Wait and observe
    Accelerate's pre-callback H2D transfers. This mirrors the PyTorch Lightning
    integration's ``init()``; HF uses ``mode="auto"`` because ``trace_step``
    drives forward/backward timing through the patch-gated auto-timers, whereas
    Lightning's callback owns that timing directly.
    """
    import traceml_ai as traceml

    config = traceml.init(mode="auto")
    try:
        _install_trainer_lifecycle_guard()
    except Exception as exc:
        _log_hf_error("Trainer lifecycle guard installation failed", exc)
    try:
        _install_training_batch_timing()
    except Exception as exc:
        _log_hf_error("Training batch timing installation failed", exc)
    try:
        _install_resume_skip_tracking()
    except Exception as exc:
        _log_hf_error("Resume input attribution installation failed", exc)
    try:
        _install_non_training_input_scopes()
    except Exception as exc:
        _log_hf_error("Non-training input scope installation failed", exc)
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
    """Emit one integration warning without affecting Trainer control flow.

    Logging handlers are application-owned and may raise from ``emit``. Treat
    warning delivery as best-effort: use the normal logger first, fall back to
    stderr if it fails, and never let either output path interrupt training.
    """
    if key in _WARNED_CAPABILITIES:
        return
    _WARNED_CAPABILITIES.add(key)
    try:
        logger.warning("[TraceML] " + message, *args)
        return
    except Exception:
        pass

    try:
        rendered = message % args if args else message
        print(f"[TraceML] {rendered}", file=sys.stderr)
    except Exception:
        # Instrumentation diagnostics must never change user control flow.
        pass


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
        if _TRAINING_ATTEMPT_STATE.omit_current_group:
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
        if self._owns_run and _TRAINING_ATTEMPT_STATE.omit_current_group:
            # The lazy resume group has no trace_step or memory capture.
            # Discard any pending events before enabling the next collection.
            _abort_pending_capture_safely()
            _TRAINING_ATTEMPT_STATE.omit_current_group = False
            return
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


class _TrainingBatchIterator:
    """Expose timing only while HF requests a training microbatch.

    ``Trainer.get_batch_samples`` is the standard Trainer's collection seam:
    training batches pass through it, while evaluation and prediction consume
    their dataloaders directly. Accelerate performs device placement inside
    the iterator's ``next`` call, so this proxy scopes the existing DataLoader
    and H2D instrumentation to exactly that call.

    The surrounding Trainer lifecycle keeps DataLoader timing disabled by
    default. The nested scope restores that policy after every fetch, including
    when fetching raises, and leaves collection-side bookkeeping outside both
    the input-wait and H2D regions. ``measure_input=False`` preserves that
    disabled policy for a complete optimizer group whose resume-skip work
    cannot be separated from its first real batch.
    """

    def __init__(self, iterator, *, measure_input: bool = True):
        self._iterator = iterator
        self._measure_input = measure_input

    def __iter__(self):
        return self

    def __next__(self):
        if not self._measure_input:
            return next(self._iterator)
        with dataloader_timing_scope(lambda: True):
            with h2d_auto_timer(include_step_time=True):
                return next(self._iterator)


def _consume_resume_input_suppression() -> bool:
    """Consume the one-shot marker for a lazily skipped resume group."""
    if not _TRAINING_ATTEMPT_STATE.suppress_next_input_group:
        return False
    _TRAINING_ATTEMPT_STATE.suppress_next_input_group = False
    return True


def _install_training_batch_timing() -> None:
    """Measure Trainer training fetches and prepared input transfers.

    Accelerate moves each returned training batch to the device inside the
    ``next()`` calls made by ``Trainer.get_batch_samples``. Those transfers
    occur before ``on_step_begin`` opens the main ``trace_step`` envelope.
    The iterator proxy enables the existing DataLoader timing policy only for
    those training fetches and gives each transfer a matching short step-time
    segment. Evaluation and prediction do not use this collection path and
    remain disabled by the surrounding Trainer lifecycle scope.

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
            "Hugging Face Trainer timing requires transformers>=4.46.1 "
            "and Trainer.get_batch_samples. Training will continue, but "
            "TraceML will omit training Input Wait and pre-step H2D timing. "
            "Upgrade Transformers to enable those measurements.",
        )
        return
    if getattr(original, "_traceml_training_batch_timing", False):
        return

    @wraps(original)
    def timed_get_batch_samples(trainer, epoch_iterator, *args, **kwargs):
        if _traceml_disabled() or not _traceml_callbacks(trainer):
            return original(trainer, epoch_iterator, *args, **kwargs)

        measure_input = not _consume_resume_input_suppression()
        # Standard Trainer collects one accumulation group before its step
        # callbacks. Keep this decision through on_step_end, not just next().
        _TRAINING_ATTEMPT_STATE.omit_current_group = not measure_input
        return original(
            trainer,
            _TrainingBatchIterator(
                epoch_iterator,
                measure_input=measure_input,
            ),
            *args,
            **kwargs,
        )

    timed_get_batch_samples._traceml_training_batch_timing = True
    Trainer.get_batch_samples = timed_get_batch_samples


def _loader_skips_lazily(loader, num_batches: int) -> bool:
    """Return whether Accelerate will consume skipped batches at iteration."""
    if num_batches <= 0:
        return False

    try:
        candidate = loader
        # XLA's MpDeviceLoaderWrapper exposes the underlying loader as .dataloader.
        for _ in range(2):
            if getattr(candidate, "skip_batches", 0) > 0:
                return True
            candidate = getattr(candidate, "dataloader", None)
            if candidate is None:
                break
    except Exception:
        # Compatibility inspection must never interfere with checkpoint
        # restoration. An unknown loader retains the existing measurement.
        return False
    return False


def _install_resume_skip_tracking() -> None:
    """Detect HF resume paths that lazily consume skipped input batches.

    Accelerate uses sampler-level skipping for map-style datasets, so no
    discarded batch is fetched. Iterable datasets instead discard batches
    inside the first iterator call, where their input and H2D work cannot be
    separated from the first real batch. Mark that one optimizer group so the
    collection hook and callback omit the complete group from step telemetry.
    Training still executes every microbatch and the optimizer update.
    """
    if not HAS_TRANSFORMERS:
        return

    import transformers.trainer as trainer_module

    original = getattr(trainer_module, "skip_first_batches", None)
    if not callable(original) or getattr(
        original,
        "_traceml_resume_skip_tracking",
        False,
    ):
        return

    @wraps(original)
    def tracked_skip_first_batches(dataloader, num_batches=0):
        skipped_loader = original(dataloader, num_batches)
        if (
            _TRAINING_ATTEMPT_STATE.active
            and not _traceml_disabled()
            and _loader_skips_lazily(skipped_loader, num_batches)
        ):
            _TRAINING_ATTEMPT_STATE.suppress_next_input_group = True
        return skipped_loader

    tracked_skip_first_batches._traceml_resume_skip_tracking = True
    trainer_module.skip_first_batches = tracked_skip_first_batches


def _scoped_non_training_input(original):
    """Wrap one synchronous Trainer entry point with input timing disabled."""

    @wraps(original)
    def wrapped(trainer, *args, **kwargs):
        if _traceml_disabled() or not _traceml_callbacks(trainer):
            return original(trainer, *args, **kwargs)
        with dataloader_timing_scope(lambda: False):
            return original(trainer, *args, **kwargs)

    wrapped._traceml_non_training_input_scope = True
    return wrapped


def _install_non_training_input_scopes() -> None:
    """Keep standalone evaluation and prediction input out of step captures.

    Evaluation triggered from ``train`` is already covered by the lifecycle's
    disabled-by-default policy. These narrow public-method wrappers apply the
    same policy when users call ``Trainer.evaluate`` or ``Trainer.predict``
    directly, preventing their fetches from remaining in the process-wide
    pending capture.
    """
    if not HAS_TRANSFORMERS:
        return

    from transformers import Trainer

    for method_name in ("evaluate", "predict"):
        original = getattr(Trainer, method_name, None)
        if not callable(original) or getattr(
            original, "_traceml_non_training_input_scope", False
        ):
            continue

        setattr(Trainer, method_name, _scoped_non_training_input(original))


def _warn_if_training_batch_timing_is_bypassed(trainer) -> None:
    """Warn when a Trainer override bypasses the installed collection hook."""
    method = getattr(trainer, "get_batch_samples", None)
    function = getattr(method, "__func__", method)
    if not callable(function) or getattr(
        function,
        "_traceml_training_batch_timing",
        False,
    ):
        return

    trainer_type = type(trainer)
    name = f"{trainer_type.__module__}.{trainer_type.__qualname__}"
    _warn_hf_once(
        f"get-batch-samples-override:{name}",
        "%s overrides get_batch_samples, so TraceML cannot guarantee "
        "training Input Wait or pre-step H2D timing for this Trainer; "
        "training will continue without those signals.",
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

        _warn_if_training_batch_timing_is_bypassed(trainer)
        owner = callbacks[0]
        for callback in callbacks:
            callback._set_run_owner(callback is owner)

        # Start every Trainer attempt with a clean capture. This also handles
        # failures that happen while fetching inputs, before on_step_begin.
        _abort_pending_capture_safely()
        previous_attempt_state = (
            _TRAINING_ATTEMPT_STATE.active,
            _TRAINING_ATTEMPT_STATE.suppress_next_input_group,
            _TRAINING_ATTEMPT_STATE.omit_current_group,
        )
        _TRAINING_ATTEMPT_STATE.active = True
        _TRAINING_ATTEMPT_STATE.suppress_next_input_group = False
        _TRAINING_ATTEMPT_STATE.omit_current_group = False
        try:
            # HF has no reliable public training-loader flag at fetch time:
            # get_batch_samples runs before training_step calls model.train().
            # Keep timing off for the attempt and let the training iterator
            # proxy enable it narrowly around each real training fetch.
            with dataloader_timing_scope(lambda: False):
                return original(trainer, *args, **kwargs)
        finally:
            # This runs before Accelerate handles an OOM and retries the same
            # inner loop, so a failed attempt cannot leak into the next one.
            for callback in callbacks:
                callback._abort_step_cm_safely()
                # Ownership applies only to this attempt, not later reuse.
                callback._set_run_owner(True)
            _abort_pending_capture_safely()
            (
                _TRAINING_ATTEMPT_STATE.active,
                _TRAINING_ATTEMPT_STATE.suppress_next_input_group,
                _TRAINING_ATTEMPT_STATE.omit_current_group,
            ) = previous_attempt_state

    guarded_inner_training_loop._traceml_lifecycle_guard = True
    Trainer._inner_training_loop = guarded_inner_training_loop


__all__ = ["TraceMLTrainerCallback", "init"]
