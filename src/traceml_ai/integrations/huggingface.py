import logging
import os
import sys
from typing import Any, Dict, Optional

from traceml_ai.sdk.decorators_compat import trace_model_instance, trace_step

logger = logging.getLogger(__name__)

TRACEML_DISABLED = os.environ.get("TRACEML_DISABLED") == "1"

try:
    from transformers import Trainer, TrainerCallback

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    Trainer = object  # Fallback for type hinting
    TrainerCallback = object  # Fallback for type hinting


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

    def __init__(
        self, traceml_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "TraceMLTrainerCallback requires 'transformers' to be "
                "installed. Please run `pip install transformers`."
            )
        super().__init__()
        self._traceml_kwargs = traceml_kwargs
        self._step_cm = None
        self._hooks_attached = False
        self._attached_model_id: Optional[int] = None

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

    def _maybe_attach_model_hooks(self, model) -> None:
        if self._traceml_kwargs is None or model is None:
            return
        if self._hooks_attached and id(model) == self._attached_model_id:
            return
        try:
            trace_model_instance(model, **self._traceml_kwargs)
            self._attached_model_id = id(model)
            self._hooks_attached = True
            logger.info(
                "[TraceML] Optional model tracing initialized (callback)."
            )
        except Exception as exc:
            _log_hf_error("Failed to initialize model tracing", exc)

    def on_train_begin(self, args, state, control, **kwargs):
        if TRACEML_DISABLED:
            return
        self._maybe_attach_model_hooks(kwargs.get("model"))

    def on_step_begin(self, args, state, control, **kwargs):
        if TRACEML_DISABLED:
            return

        # If a previous step raised, HF never fired on_step_end and the
        # trace_step generator is still suspended. Close it before opening
        # a new one so forward/backward auto-timer flags do not stay armed.
        self._close_step_cm_safely()

        model = kwargs.get("model")
        # Late attachment fallback when on_train_begin missed the model.
        self._maybe_attach_model_hooks(model)

        try:
            self._step_cm = trace_step(model)
            self._step_cm.__enter__()
        except Exception as exc:
            self._step_cm = None
            _log_hf_error("trace_step enter failed", exc)

    def on_step_end(self, args, state, control, **kwargs):
        if TRACEML_DISABLED:
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
        traceml_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "TraceMLTrainer requires 'transformers' to be installed. "
                "Please run `pip install transformers`."
            )

        super().__init__(*args, **kwargs)
        self.traceml_enabled = traceml_enabled
        self.traceml_kwargs = traceml_kwargs

        if not traceml_enabled or TRACEML_DISABLED:
            return

        # Dedup guard: a user passing callbacks=[TraceMLTrainerCallback()] to
        # TraceMLTrainer would otherwise double-instrument every step.
        existing = getattr(self.callback_handler, "callbacks", [])
        if any(isinstance(cb, TraceMLTrainerCallback) for cb in existing):
            return

        self.add_callback(
            TraceMLTrainerCallback(traceml_kwargs=traceml_kwargs)
        )
