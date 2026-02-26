import logging
from typing import Optional
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger("traceml.lightning")


class TraceMLCallback(Callback):
    """
    TraceML integration for PyTorch Lightning via a standard Callback.

    What it does:
      - Defines a step boundary aligned to optimizer update steps (trainer.global_step).
        With gradient accumulation, a step spans multiple batches and ends only when
        global_step increments.
      - Times forward/backward per microbatch, and optimizer step when it happens.
      - Resets/records per-step memory (best-effort).
      - Flushes step events exactly once per optimizer step.
      - Dataloader fetch is naturally excluded because hooks run after the batch is produced.

      - For manual optimization, Lightning may not update global_step the same way; we fall back
        to flushing per batch and warn once.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        record_memory: bool = True,
        flush: bool = True,
    ) -> None:
        super().__init__()
        self.use_gpu = use_gpu
        self.record_memory = record_memory
        self.flush = flush

        # Per-step lifecycle
        self._active_step: bool = False
        self._step_ctx = None
        self._mem_tracker = None

        # Per-batch/microbatch timers
        self._forward_ctx = None
        self._backward_ctx = None
        self._optimizer_ctx = None

        # Global-step tracking (optimizer-step boundary)
        self._last_seen_global_step: Optional[int] = None

        # Manual optimization fallback warning
        self._warned_manual_opt: bool = False


    def _rank_zero_warn(self, trainer, msg: str) -> None:
        if getattr(trainer, "is_global_zero", True):
            logger.warning(msg)

    def _rank_zero_debug(self, trainer, msg: str) -> None:
        if getattr(trainer, "is_global_zero", True):
            logger.debug(msg)

    def _safe_exit(self, ctx_attr: str) -> None:
        ctx = getattr(self, ctx_attr, None)
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            setattr(self, ctx_attr, None)

    def _begin_step(self, trainer, pl_module) -> None:
        from traceml.utils.timing import timed_region

        self._active_step = True

        # Start overall step timing (spans accumulation microbatches until optimizer step happens)
        self._step_ctx = timed_region("_traceml_internal:step_time", scope="step", use_gpu=self.use_gpu)
        try:
            self._step_ctx.__enter__()
        except Exception:
            self._step_ctx = None

        # Reset step memory (best-effort)
        self._mem_tracker = None
        if self.record_memory:
            try:
                from traceml.utils.step_memory import StepMemoryTracker

                self._mem_tracker = StepMemoryTracker(pl_module)
                self._mem_tracker.reset()
            except Exception as e:
                self._rank_zero_warn(trainer, f"[TraceML] memory reset failed: {e}")

    def _end_step(self, trainer, pl_module, step_id: int) -> None:
        # Close any optimizer timer still open (best-effort)
        self._safe_exit("_optimizer_ctx")

        # Close overall step timer
        self._safe_exit("_step_ctx")
        self._active_step = False

        # Record memory (best-effort)
        if self._mem_tracker is not None:
            try:
                self._mem_tracker.record()
            except Exception as e:
                self._rank_zero_warn(trainer, f"[TraceML] memory record failed: {e}")
            finally:
                self._mem_tracker = None

        # Flush
        if self.flush:
            try:
                from traceml.decorators import TraceState
                from traceml.utils.flush_buffers import flush_step_events

                TraceState.step = step_id
                flush_step_events(pl_module, step_id)
            except Exception as e:
                self._rank_zero_warn(trainer, f"[TraceML] flush failed: {e}")


    def setup(self, trainer, pl_module, stage: Optional[str] = None) -> None:
        # Initialize our step boundary tracker
        self._last_seen_global_step = int(getattr(trainer, "global_step", 0) or 0)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        # Manual optimization can make global_step behavior different; warn once and fall back later.
        if getattr(pl_module, "automatic_optimization", True) is False and not self._warned_manual_opt:
            self._warned_manual_opt = True
            self._rank_zero_warn(
                trainer,
                "[TraceML] Lightning manual optimization detected; using per-batch flushing fallback "
                "(optimizer-step boundaries may be unavailable).",
            )

        # Start step scope at the beginning of an optimizer-update cycle (first microbatch)
        if not self._active_step:
            self._begin_step(trainer, pl_module)

        # Start forward timing for this microbatch
        from traceml.utils.timing import timed_region

        self._forward_ctx = timed_region(
            "_traceml_internal:forward_time", scope="step", use_gpu=self.use_gpu)
        try:
            self._forward_ctx.__enter__()
        except Exception:
            self._forward_ctx = None

    def on_before_backward(self, trainer, pl_module, loss) -> None:
        from traceml.utils.timing import timed_region

        # End forward, begin backward
        self._safe_exit("_forward_ctx")

        self._backward_ctx = timed_region(
            "_traceml_internal:backward_time", scope="step", use_gpu=self.use_gpu)
        try:
            self._backward_ctx.__enter__()
        except Exception:
            self._backward_ctx = None

    def on_after_backward(self, trainer, pl_module) -> None:
        # End backward timing for this microbatch
        self._safe_exit("_backward_ctx")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        # Start optimizer step timing (only called when an optimizer step actually happens)
        from traceml.utils.timing import timed_region

        self._optimizer_ctx = timed_region(
            "_traceml_internal:optimizer_step", scope="step", use_gpu=self.use_gpu)
        try:
            self._optimizer_ctx.__enter__()
        except Exception:
            self._optimizer_ctx = None

    def on_before_zero_grad(self, trainer, pl_module, optimizer) -> None:
        # End optimizer step timing (this hook is after optimizer.step and before zero_grad)
        self._safe_exit("_optimizer_ctx")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        self._safe_exit("_forward_ctx")
        self._safe_exit("_backward_ctx")

        current_global_step = int(getattr(trainer, "global_step", 0) or 0)

        # Automatic optimization: end a "TraceML step" only when optimizer step happened
        if getattr(pl_module, "automatic_optimization", True) is True:
            if self._last_seen_global_step is None:
                self._last_seen_global_step = current_global_step

            if current_global_step != self._last_seen_global_step and self._active_step:
                self._end_step(trainer, pl_module, step_id=current_global_step)
                self._last_seen_global_step = current_global_step
            return

        # Manual optimization fallback: flush per batch
        if self._active_step:
            # Use batch_idx-derived monotonic id if global_step doesn't advance
            fallback_step_id = (
                current_global_step
                if current_global_step != (self._last_seen_global_step or 0)
                else ((self._last_seen_global_step or 0) + 1)
            )
            self._end_step(trainer, pl_module, step_id=fallback_step_id)
            self._last_seen_global_step = fallback_step_id

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        # Close everything best-effort to avoid leaked contexts on crashes
        self._safe_exit("_forward_ctx")
        self._safe_exit("_backward_ctx")
        self._safe_exit("_optimizer_ctx")
        if self._active_step:
            # Flush what we have under the current global_step
            step_id = int(getattr(trainer, "global_step", 0) or 0)
            self._end_step(trainer, pl_module, step_id=step_id)

    def on_fit_end(self, trainer, pl_module) -> None:
        # Ensure clean shutdown
        self._safe_exit("_forward_ctx")
        self._safe_exit("_backward_ctx")
        self._safe_exit("_optimizer_ctx")
        if self._active_step:
            step_id = int(getattr(trainer, "global_step", 0) or 0)
            self._end_step(trainer, pl_module, step_id=step_id)