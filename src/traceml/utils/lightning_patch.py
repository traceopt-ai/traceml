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
        Captures full step time (forward + backward + optimizer).
        """
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            from traceml.decorators import begin_trace_step
            # Store instrumentation state on the callback instance
            self._traceml_state = begin_trace_step(pl_module)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            from traceml.decorators import end_trace_step
            if hasattr(self, "_traceml_state"):
                end_trace_step(pl_module, self._traceml_state)

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