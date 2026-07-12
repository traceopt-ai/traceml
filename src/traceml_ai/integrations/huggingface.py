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
    from transformers import Trainer, TrainerCallback

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    Trainer = object  # Fallback for type hinting
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
    Preferred Hugging Face integration for TraceML.

    Register with ``Trainer(..., callbacks=[TraceMLTrainerCallback()])``.

    The callback is a pure bracket around TraceML's ``trace_step`` context
    manager: it opens ``trace_step`` in ``on_step_begin`` and closes it in
    ``on_step_end``. ``trace_step`` owns the step memory tracker, the step
    counter advance, the auto-timers for forward/backward/h2d, and the
    per-step flush. Nothing is duplicated here.

    One TraceML step equals one optimizer step. With
    ``gradient_accumulation_steps > 1``, forward and backward events from all
    accumulated micro-batches fold into a single TraceML step. See the HF
    integration docs for the full list of limitations vs. ``TraceMLTrainer``.
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


class TraceMLTrainer(Trainer if HAS_TRANSFORMERS else object):
    """
    Thin wrapper around ``transformers.Trainer`` that auto-installs
    ``TraceMLTrainerCallback``.

    Kept for backward compatibility with users on the original TraceML HF
    integration API. New code should prefer
    ``Trainer(..., callbacks=[TraceMLTrainerCallback()])`` directly.
    """

    def __init__(
        self,
        *args,
        traceml_enabled: bool = True,
        **kwargs,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "TraceMLTrainer requires the Hugging Face integration. "
                "Install it with `pip install 'traceml-ai[hf]'`."
            )

        super().__init__(*args, **kwargs)
        self.traceml_enabled = traceml_enabled

        if not traceml_enabled or _traceml_disabled():
            return

        # Dedup guard: a user passing callbacks=[TraceMLTrainerCallback()] to
        # TraceMLTrainer would otherwise double-instrument every step.
        existing = getattr(self.callback_handler, "callbacks", [])
        if any(isinstance(cb, TraceMLTrainerCallback) for cb in existing):
            return

        self.add_callback(TraceMLTrainerCallback())


__all__ = ["TraceMLTrainerCallback", "TraceMLTrainer", "init"]
