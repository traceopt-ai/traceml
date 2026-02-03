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

    # Cache imports at patch time (once) rather than per-callback-call
    from traceml.decorators import begin_trace_step, end_trace_step
    from traceml.utils.steptimer import begin_timed_region, end_timed_region

    class TraceMLCallback(Callback):
        """
        Internal callback to wrap Lightning training batches with TraceML instrumentation.
        Captures full step time (forward + backward + optimizer) as well as individual
        phase timings (forward, backward, optimizer_step).
        """
        
        def __init__(self):
            super().__init__()
            self._traceml_state = None
            self._forward_timer = None
            self._backward_timer = None
            self._optimizer_timer = None
        
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            # Start step boundary + forward timer
            self._traceml_state = begin_trace_step(pl_module)
            self._forward_timer = begin_timed_region("forward")

        def on_before_backward(self, trainer, pl_module, loss):
            # End forward, start backward
            if self._forward_timer:
                end_timed_region(self._forward_timer)
                self._forward_timer = None
            self._backward_timer = begin_timed_region("backward")

        def on_after_backward(self, trainer, pl_module):
            # End backward
            if self._backward_timer:
                end_timed_region(self._backward_timer)
                self._backward_timer = None

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            # Start optimizer step timer
            self._optimizer_timer = begin_timed_region("optimizer_step")

        def on_before_zero_grad(self, trainer, pl_module, optimizer):
            # End optimizer step timer (zero_grad happens after step)
            if self._optimizer_timer:
                end_timed_region(self._optimizer_timer)
                self._optimizer_timer = None

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            # Safety cleanup for edge cases (gradient accumulation, skipped steps, etc.)
            if self._forward_timer:
                end_timed_region(self._forward_timer)
                self._forward_timer = None
            if self._backward_timer:
                end_timed_region(self._backward_timer)
                self._backward_timer = None
            if self._optimizer_timer:
                end_timed_region(self._optimizer_timer)
                self._optimizer_timer = None
            
            # End step boundary and flush buffers
            if self._traceml_state:
                end_trace_step(pl_module, self._traceml_state)
                self._traceml_state = None

    _orig_init = Trainer.__init__

    @functools.wraps(_orig_init)
    def traceml_init(self, *args, **kwargs):
        # Retrieve existing callbacks or initialize a new list
        cbs = kwargs.get("callbacks") or []
        if not isinstance(cbs, list):
            cbs = [cbs]
        else:
            cbs = list(cbs)  # Copy to avoid mutating user's list

        # Inject TraceMLCallback if not already present
        if not any(isinstance(c, TraceMLCallback) for c in cbs):
            cbs.append(TraceMLCallback())
        
        kwargs["callbacks"] = cbs
        _orig_init(self, *args, **kwargs)

    Trainer.__init__ = traceml_init
    Trainer._traceml_patched = True
