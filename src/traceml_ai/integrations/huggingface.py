"""TraceML callbacks for the standard Hugging Face Trainer.

The TraceMLTrainer wrapper was intentionally removed. Use init() and register
TraceMLTrainerCallback with transformers.Trainer.
"""

import os
import sys

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
    than relying on import order. This mirrors the PyTorch Lightning
    integration's ``init()``; HF uses ``mode="auto"`` because ``trace_step``
    drives forward/backward timing through the patch-gated auto-timers, whereas
    Lightning's callback owns that timing directly.
    """
    import traceml_ai as traceml

    return traceml.init(mode="auto")


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


class TraceMLTrainerCallback(TrainerCallback if HAS_TRANSFORMERS else object):
    """
    Hugging Face Trainer integration for TraceML.

    Register with ``Trainer(..., callbacks=[TraceMLTrainerCallback()])``.

    The callback is a pure bracket around TraceML's ``trace_step`` context
    manager: it opens ``trace_step`` in ``on_step_begin`` and closes it in
    ``on_step_end``. ``trace_step`` owns the step memory tracker, the step
    counter advance, the auto-timers for forward/backward/h2d, and the
    per-step capture lifecycle. Nothing is duplicated here.

    One completed TraceML step corresponds to one HF accumulation/update
    boundary (``on_step_end``), including when AMP skips the parameter update.
    ``on_substep_end`` does not advance the counter: forward and backward
    events from the actual micro-batches in the group fold into one step,
    including a shorter final group. Optimizer events describe calls that
    actually run; their count does not drive TraceML's step counter.

    TraceML step IDs remain process-local, so their increments match HF's
    ``global_step`` increments during recorded, completed groups; their
    absolute values need not match after checkpoint resume. See the HF
    integration docs for the input-timing and interrupted-step limitations.
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

    def _close_step_cm_safely(self) -> None:
        """Defensively exit any open trace_step context."""
        cm = self._step_cm
        if cm is None:
            return
        self._step_cm = None
        try:
            cm.__exit__(None, None, None)
        except Exception as exc:
            _log_hf_error("trace_step exit failed", exc)

    def on_train_begin(self, args, state, control, **kwargs):
        if _traceml_disabled():
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

        # Self-heal before training starts. If this callback instance is
        # reused and a previous run crashed mid-step, the leaked trace_step is
        # still suspended with its auto-timer flags armed. With
        # eval_on_start=True, HF runs evaluation between here and the first
        # on_step_begin, so those eval forward passes would otherwise be timed
        # into the orphaned step. Closing here covers that window.
        self._close_step_cm_safely()

    def on_step_begin(self, args, state, control, **kwargs):
        if _traceml_disabled():
            return

        # If a previous step raised, HF never fired on_step_end and the
        # trace_step generator is still suspended. Close it before opening
        # a new one so forward/backward auto-timer flags do not stay armed.
        self._close_step_cm_safely()

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
        if _traceml_disabled():
            return
        self._close_step_cm_safely()

    def on_train_end(self, args, state, control, **kwargs):
        # Bounds damage if training aborted mid-step.
        self._close_step_cm_safely()


__all__ = ["TraceMLTrainerCallback", "init"]
