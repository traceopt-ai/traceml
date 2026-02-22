import functools


def patch_lightning():
    """
    Automatically injects TraceMLCallback into PyTorch Lightning Trainer.
    Ensures zero-code integration for Lightning users.
    """
    try:
        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import Callback
    except ImportError:
        # Lightning not installed in this environment
        return

    # Guard against double patching
    if getattr(Trainer, "_traceml_patched", False):
        return

    class TraceMLCallback(Callback):
        """
        Internal callback to wrap Lightning training batches with TraceML
        instrumentation. Captures full step time (forward + backward +
        optimizer) as well as individual phase timings.
        """

        def __init__(self):
            super().__init__()
            self._traceml_step_ctx = None
            self._forward_ctx = None
            self._backward_ctx = None
            self._optimizer_ctx = None

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            import sys

            from traceml.utils.step_memory import StepMemoryTracker
            from traceml.utils.timing import timed_region

            # Reset step memory
            try:
                mem_tracker = StepMemoryTracker(pl_module)
                mem_tracker.reset()
                self._mem_tracker = mem_tracker
            except Exception as e:
                print(f"[TraceML] memory reset failed: {e}", file=sys.stderr)
                self._mem_tracker = None

            # Start timing the forward pass (ends in on_before_backward)
            self._forward_ctx = timed_region("forward", scope="step")
            self._forward_ctx.__enter__()

        def on_before_backward(self, trainer, pl_module, loss):
            from traceml.utils.timing import timed_region

            # End forward timing
            if self._forward_ctx is not None:
                try:
                    self._forward_ctx.__exit__(None, None, None)
                except Exception:
                    pass
                self._forward_ctx = None

            # Start backward timing
            self._backward_ctx = timed_region("backward", scope="step")
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
            from traceml.utils.timing import timed_region

            # Start optimizer step timing
            self._optimizer_ctx = timed_region("optimizer_step", scope="step")
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
            import sys

            from traceml.decorators import TraceState
            from traceml.utils.flush_buffers import flush_step_events

            # Safety: end any timers that weren't ended (edge cases)
            for ctx_attr in (
                "_forward_ctx",
                "_backward_ctx",
                "_optimizer_ctx",
            ):
                ctx = getattr(self, ctx_attr, None)
                if ctx is not None:
                    try:
                        ctx.__exit__(None, None, None)
                    except Exception:
                        pass
                    setattr(self, ctx_attr, None)

            # Record step memory
            if self._mem_tracker is not None:
                try:
                    self._mem_tracker.record()
                except Exception as e:
                    print(f"[TraceML] record failed: {e}", file=sys.stderr)

            # Advance step counter and flush
            TraceState.step += 1
            try:
                flush_step_events(pl_module, TraceState.step)
            except Exception as e:
                print(f"[TraceML] flush failed: {e}", file=sys.stderr)

    _orig_init = Trainer.__init__

    @functools.wraps(_orig_init)
    def traceml_init(self, *args, **kwargs):
        # Retrieve existing callbacks or initialize a new list
        cbs = kwargs.get("callbacks", []) or []
        if not isinstance(cbs, list):
            cbs = [cbs]
        else:
            cbs = list(cbs)  # Copy to avoid modifying user's list in place

        # Inject TraceMLCallback if not already present
        if not any(isinstance(c, TraceMLCallback) for c in cbs):
            cbs.append(TraceMLCallback())

        kwargs["callbacks"] = cbs
        _orig_init(self, *args, **kwargs)

    Trainer.__init__ = traceml_init
    Trainer._traceml_patched = True
