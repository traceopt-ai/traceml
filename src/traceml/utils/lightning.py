import sys

from lightning.pytorch.callbacks import Callback

from traceml.decorators import TraceState
from traceml.utils.flush_buffers import flush_step_events
from traceml.utils.step_memory import StepMemoryTracker
from traceml.utils.timing import (
    TimeEvent,
    TimeScope,
    record_event,
    timed_region,
)


class TraceMLCallback(Callback):
    """
    Official TraceML Callback for PyTorch Lightning.

    Captures full step time (forward + backward + optimizer) as well as
    individual phase timings. Safely handles gradient accumulation by
    treating each micro-batch as a step, providing 0-duration optimizer
    events on accumulating steps to preserve dashboard step alignment.
    """

    def __init__(self):
        super().__init__()
        self._traceml_step_ctx = None
        self._forward_ctx = None
        self._backward_ctx = None
        self._optimizer_ctx = None

        self._mem_tracker = None
        self._opt_step_occurred = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Start overall step timing
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
            print(f"[TraceML] memory reset failed: {e}", file=sys.stderr)
            self._mem_tracker = None

        # Start timing the forward pass (ends in on_before_backward)
        self._forward_ctx = timed_region(
            "_traceml_internal:forward_time", scope="step"
        )
        self._forward_ctx.__enter__()

    def on_before_backward(self, trainer, pl_module, loss):
        # End forward timing
        if self._forward_ctx is not None:
            try:
                self._forward_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._forward_ctx = None

        # Start backward timing
        self._backward_ctx = timed_region(
            "_traceml_internal:backward_time", scope="step"
        )
        self._backward_ctx.__enter__()

    def on_after_backward(self, trainer, pl_module):
        # End backward timing
        if self._backward_ctx is not None:
            try:
                self._backward_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._backward_ctx = None

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self._opt_step_occurred = True

        # Start optimizer step timing
        self._optimizer_ctx = timed_region(
            "_traceml_internal:optimizer_step", scope="step"
        )
        self._optimizer_ctx.__enter__()

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        # End optimizer step timing (zero_grad happens after step)
        if self._optimizer_ctx is not None:
            try:
                self._optimizer_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._optimizer_ctx = None

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        # Safety: end any active context managers (edge cases)
        for ctx_attr in (
            "_forward_ctx",
            "_backward_ctx",
            "_optimizer_ctx",
            "_traceml_step_ctx",
        ):
            ctx = getattr(self, ctx_attr, None)
            if ctx is not None:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass
                setattr(self, ctx_attr, None)

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
                print(f"[TraceML] record failed: {e}", file=sys.stderr)

        # Advance step counter and flush (treating every micro-batch as a step
        # to preserve fine-grained forward/backward times)
        TraceState.step += 1
        try:
            flush_step_events(pl_module, TraceState.step)
        except Exception as e:
            print(f"[TraceML] flush failed: {e}", file=sys.stderr)
