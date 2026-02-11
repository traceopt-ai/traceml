import logging
from typing import Any, Dict, Optional

from traceml.decorators import trace_model_instance, trace_step

# Setup logging
logger = logging.getLogger(__name__)

try:
    from transformers import Trainer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    Trainer = object  # Fallback for type hinting


class TraceMLTrainer(Trainer if HAS_TRANSFORMERS else object):
    """
    A subclass of Hugging Face's Trainer that automatically integrates TraceML.

    This class wraps the `training_step` with the `trace_step` context manager
    to capture step-level metrics (timing, memory, etc.).
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

        # If model-level tracing (Deep-Dive) is requested, apply it now
        self.traceml_kwargs = traceml_kwargs
        self._traceml_hooks_attached = False

    def training_step(self, model, inputs, *args, **kwargs) -> Any:
        """
        Overridden training step to include TraceML instrumentation.
        """
        if self.traceml_enabled:
            # Lazily attach hooks on the first step to ensure we catch the
            # final wrapped/moved model (e.g. DDP, Accelerator)
            if (
                not self._traceml_hooks_attached
                and self.traceml_kwargs is not None
            ):
                try:
                    # Unwrap model to ensure clean layer names and avoid DDP/FSDP wrapping issues
                    if HAS_TRANSFORMERS:
                        from transformers.modeling_utils import unwrap_model

                        _model = unwrap_model(model)
                    else:
                        _model = model

                    trace_model_instance(_model, **self.traceml_kwargs)
                    self._traceml_hooks_attached = True
                    logger.info(
                        "[TraceML] Deep-Dive model tracing initialized "
                        "(lazy)."
                    )
                except Exception as e:
                    logger.error(
                        f"[TraceML] Failed to initialize model tracing: {e}"
                    )

            with trace_step(model):
                return super().training_step(model, inputs, *args, **kwargs)

        return super().training_step(model, inputs, *args, **kwargs)
