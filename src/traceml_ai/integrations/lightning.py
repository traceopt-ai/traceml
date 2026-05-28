import os
import sys

from traceml_ai.runtime.state import get_trace_session_state
from traceml_ai.utils.flush_buffers import flush_step_events
from traceml_ai.utils.step_memory import StepMemoryTracker
from traceml_ai.utils.timing import (
    TimeEvent,
    TimeScope,
    record_event,
    timed_region,
)

try:
    from lightning.pytorch.callbacks import Callback

    IS_LIGHTNING_AVAILABLE = True
except ImportError:
    Callback = object
    IS_LIGHTNING_AVAILABLE = False

TRACEML_DISABLED = os.environ.get("TRACEML_DISABLED") == "1"


def init():
    """
    Initialize TraceML for PyTorch Lightning runs.

    Lightning owns forward/backward/optimizer timing through TraceMLCallback.
    The integration init only enables DataLoader fetch timing, which happens
    before Lightning callback hooks can observe the batch.
    """
    import traceml_ai as traceml

    return traceml.init(mode="selective", patch_dataloader=True)


def _log_lightning_error(message: str, exc: Exception) -> None:
    """
    Log TraceML callback failures without interrupting Lightning training.

    The callback is best-effort instrumentation. TraceML launcher runs configure
    the shared error logger; direct callback users still get the previous stderr
    fallback if logging has not been configured.
    """
    try:
        from traceml_ai.loggers.error_log import get_error_logger

        get_error_logger("LightningIntegration").exception(
            "[TraceML] %s", message
        )
    except Exception:
        pass

    print(f"[TraceML] {message}: {exc}", file=sys.stderr)


def _device_is_cuda(device) -> bool:
    device_type = getattr(device, "type", None)
    if device_type is not None:
        return str(device_type).lower() == "cuda"
    return str(device).lower().startswith("cuda")


def _lightning_uses_cuda(trainer, pl_module) -> bool:
    strategy = getattr(trainer, "strategy", None)
    root_device = getattr(strategy, "root_device", None)
    if root_device is not None:
        return _device_is_cuda(root_device)

    module_device = getattr(pl_module, "device", None)
    return _device_is_cuda(module_device)


class TraceMLCallback(Callback):
    """
    Official TraceML Callback for PyTorch Lightning.

    Captures full step time (forward + backward + optimizer) as well as
    individual phase timings. Safely handles gradient accumulation by
    treating each micro-batch as a step, providing 0-duration optimizer
    events on accumulating steps to preserve dashboard step alignment.
    """

    def __init__(self):
        if not IS_LIGHTNING_AVAILABLE:
            raise ImportError(
                "Install traceml[lightning] to use Lightning integration"
            )
        super().__init__()
        self._traceml_step_ctx = None
        self._forward_ctx = None
        self._backward_ctx = None
        self._optimizer_ctx = None
        self._h2d_ctx = None

        self._mem_tracker = None
        self._opt_step_occurred = False

    def _close_context(self, ctx_attr: str) -> None:
        ctx = getattr(self, ctx_attr, None)
        if ctx is None:
            return
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass
        setattr(self, ctx_attr, None)

    def on_before_batch_transfer(
        self, trainer, pl_module, batch, dataloader_idx=0
    ):
        if TRACEML_DISABLED or not getattr(trainer, "training", True):
            return
        if not _lightning_uses_cuda(trainer, pl_module):
            return

        self._close_context("_h2d_ctx")
        self._h2d_ctx = timed_region(
            "_traceml_internal:h2d_time", scope="step", use_gpu=True
        )
        self._h2d_ctx.__enter__()

    def on_after_batch_transfer(
        self, trainer, pl_module, batch, dataloader_idx=0
    ):
        if TRACEML_DISABLED:
            return
        self._close_context("_h2d_ctx")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Start overall step timing
        if TRACEML_DISABLED:
            return
        self._traceml_step_ctx = timed_region(
            "_traceml_internal:step_time", scope="step", use_gpu=False
        )
        self._traceml_step_ctx.__enter__()

        # Reset flag for gradient accumulation tracking
        self._opt_step_occurred = False

        # Reset step memory
        try:
            mem_tracker = StepMemoryTracker(pl_module)
            mem_tracker.reset()
            self._mem_tracker = mem_tracker
        except Exception as e:
            _log_lightning_error("memory reset failed", e)
            self._mem_tracker = None

        # Start timing the forward pass (ends in on_before_backward)
        self._forward_ctx = timed_region(
            "_traceml_internal:forward_time", scope="step"
        )
        self._forward_ctx.__enter__()

    def on_before_backward(self, trainer, pl_module, loss):
        if TRACEML_DISABLED:
            return
        # End forward timing
        self._close_context("_forward_ctx")

        # Start backward timing
        self._backward_ctx = timed_region(
            "_traceml_internal:backward_time", scope="step"
        )
        self._backward_ctx.__enter__()

    def on_after_backward(self, trainer, pl_module):
        if TRACEML_DISABLED:
            return
        # End backward timing
        self._close_context("_backward_ctx")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if TRACEML_DISABLED:
            return
        self._opt_step_occurred = True

        # Start optimizer step timing
        self._optimizer_ctx = timed_region(
            "_traceml_internal:optimizer_step", scope="step"
        )
        self._optimizer_ctx.__enter__()

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        if TRACEML_DISABLED:
            return
        # End optimizer step timing (zero_grad happens after step)
        self._close_context("_optimizer_ctx")

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if TRACEML_DISABLED:
            return
        # Safety: end any active context managers (edge cases)
        for ctx_attr in (
            "_h2d_ctx",
            "_forward_ctx",
            "_backward_ctx",
            "_optimizer_ctx",
            "_traceml_step_ctx",
        ):
            self._close_context(ctx_attr)

        # Handle Gradient Accumulation (Micro-batches):
        # If the optimizer didn't run this batch (because of grad accumulation),
        # emit a dummy optimizer event. This ensures the dashboard's step alignment
        # (which requires all metrics to have the exact same steps) doesn't break.
        if not self._opt_step_occurred:
            try:
                record_event(
                    TimeEvent(
                        name="_traceml_internal:optimizer_step",
                        device="cpu",  # Dummy event doesn't matter
                        cpu_start=0.0,
                        cpu_end=0.0,
                        gpu_time_ms=0.0,
                        resolved=True,
                        scope=TimeScope.STEP,
                    )
                )
            except Exception:
                pass

        # Record step memory
        if self._mem_tracker is not None:
            try:
                self._mem_tracker.record()
            except Exception as e:
                _log_lightning_error("record failed", e)

        # Advance step counter and flush (treating every micro-batch as a step
        # to preserve fine-grained forward/backward times)
        trace_state = get_trace_session_state()
        trace_state.advance_step()
        try:
            flush_step_events(pl_module, trace_state.step)
        except Exception as e:
            _log_lightning_error("flush failed", e)


__all__ = ["TraceMLCallback", "init"]
