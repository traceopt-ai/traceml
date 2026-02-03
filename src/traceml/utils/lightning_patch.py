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
        Internal callback to wrap Lightning training batches with TraceML instrumentation.
        Captures full step time (forward + backward + optimizer) as well as individual
        phase timings (forward, backward, optimizer_step).
        """
        
        def __init__(self):
            super().__init__()
            self._traceml_state = None
            self._forward_timer_state = None
            self._backward_timer_state = None
            self._optimizer_timer_state = None
        
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            from traceml.decorators import begin_trace_step
            from traceml.utils.steptimer import begin_timed_region
            # Store instrumentation state on the callback instance
            self._traceml_state = begin_trace_step(pl_module)
            # Start timing the forward pass (ends in on_before_backward)
            self._forward_timer_state = begin_timed_region("forward")

        def on_before_backward(self, trainer, pl_module, loss):
            from traceml.utils.steptimer import end_timed_region, begin_timed_region
            # End forward timing
            if self._forward_timer_state is not None:
                end_timed_region(self._forward_timer_state)
                self._forward_timer_state = None
            # Start backward timing
            self._backward_timer_state = begin_timed_region("backward")

        def on_after_backward(self, trainer, pl_module):
            from traceml.utils.steptimer import end_timed_region
            # End backward timing
            if self._backward_timer_state is not None:
                end_timed_region(self._backward_timer_state)
                self._backward_timer_state = None

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            from traceml.utils.steptimer import begin_timed_region
            # Start optimizer step timing
            self._optimizer_timer_state = begin_timed_region("optimizer_step")

        def on_before_zero_grad(self, trainer, pl_module, optimizer):
            from traceml.utils.steptimer import end_timed_region
            # End optimizer step timing (zero_grad happens after step)
            if self._optimizer_timer_state is not None:
                end_timed_region(self._optimizer_timer_state)
                self._optimizer_timer_state = None

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            from traceml.decorators import end_trace_step
            from traceml.utils.steptimer import end_timed_region
            
            # Safety: end any timers that weren't ended (edge cases)
            if self._forward_timer_state is not None:
                end_timed_region(self._forward_timer_state)
                self._forward_timer_state = None
            if self._backward_timer_state is not None:
                end_timed_region(self._backward_timer_state)
                self._backward_timer_state = None
            if self._optimizer_timer_state is not None:
                end_timed_region(self._optimizer_timer_state)
                self._optimizer_timer_state = None
                
            if self._traceml_state is not None:
                end_trace_step(pl_module, self._traceml_state)
                self._traceml_state = None

    _orig_init = Trainer.__init__

    @functools.wraps(_orig_init)
    def traceml_init(self, *args, **kwargs):
        # Retrieve existing callbacks or initialize a new list
        cbs = kwargs.get("callbacks", []) or []
        if not isinstance(cbs, list):
            cbs = [cbs]
        else:
            cbs = list(cbs) # Copy to avoid modifying user's list in place

        # Inject TraceMLCallback if not already present
        if not any(isinstance(c, TraceMLCallback) for c in cbs):
            cbs.append(TraceMLCallback())
        
        kwargs["callbacks"] = cbs
        _orig_init(self, *args, **kwargs)
    
   
    Trainer.__init__ = traceml_init
    Trainer._traceml_patched = True