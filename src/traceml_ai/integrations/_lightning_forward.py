"""Forward observation owned by the Lightning callback."""

import functools

from traceml_ai.instrumentation.step_events import TimeScope

_MISSING = object()


class ForwardObserver:
    """Observe common Lightning module calls during ``training_step``."""

    def __init__(self, owner, *, disabled, on_error, region):
        self.owner = owner
        self.disabled = disabled
        self.on_error = on_error
        self.region = region
        self.handles = []
        self.active_calls = {}
        self.depth = 0
        self.training_step_active = False
        self.training_module = None
        self.original_training_step_attr = _MISSING
        self.wrapped_training_step = None
        self.calls = 0

    def install(self, trainer, pl_module, target) -> None:
        if self.wrapped_training_step is not None:
            return
        original = pl_module.training_step
        self.training_module = pl_module
        self.original_training_step_attr = pl_module.__dict__.get(
            "training_step", _MISSING
        )

        def before(module, args):
            enabled = (
                self.training_step_active
                and not self.disabled()
                and getattr(trainer, "training", False)
                and self.owner._step_capture is not None
                and self.owner._backward_ctx is None
            )
            ctx = None
            if enabled:
                self.depth += 1
                if self.depth == 1:
                    self.owner._close_context("_optimizer_ctx")
                    try:
                        ctx = self.region(
                            "_traceml_internal:forward_time",
                            scope=TimeScope.STEP,
                            record_gpu_events=True,
                        )
                        ctx.__enter__()
                        self.calls += 1
                    except Exception as exc:
                        self.on_error("forward timing unavailable", exc)
            self.active_calls.setdefault(module, []).append((enabled, ctx))

        def after(module, args, output):
            calls = self.active_calls.get(module)
            if not calls:
                return
            enabled, ctx = calls.pop()
            try:
                if ctx is not None:
                    ctx.__exit__(None, None, None)
            except Exception as exc:
                self.on_error("forward timing cleanup failed", exc)
            finally:
                if enabled:
                    self.depth -= 1

        @functools.wraps(original)
        def training_step(*args, **kwargs):
            previous = self.training_step_active
            self.training_step_active = True
            try:
                return original(*args, **kwargs)
            finally:
                self.training_step_active = previous

        try:
            for module in dict.fromkeys((target, *target.children())):
                self.handles.append(module.register_forward_pre_hook(before))
                self.handles.append(
                    module.register_forward_hook(after, always_call=True)
                )
            self.wrapped_training_step = training_step
            pl_module.training_step = training_step
        except Exception:
            self.restore()
            raise

    def restore(self) -> None:
        for handle in self.handles:
            try:
                handle.remove()
            except Exception as exc:
                self.on_error("forward hook removal failed", exc)
        self.handles.clear()
        for calls in self.active_calls.values():
            for _, ctx in reversed(calls):
                if ctx is not None:
                    try:
                        ctx.__exit__(None, None, None)
                    except Exception as exc:
                        self.on_error("forward timing cleanup failed", exc)
        self.active_calls.clear()

        module = self.training_module
        try:
            if (
                module is not None
                and module.__dict__.get("training_step")
                is self.wrapped_training_step
            ):
                if self.original_training_step_attr is _MISSING:
                    delattr(module, "training_step")
                else:
                    module.training_step = self.original_training_step_attr
        except Exception as exc:
            self.on_error("training_step restore failed", exc)
        finally:
            self.training_module = None
            self.original_training_step_attr = _MISSING
            self.wrapped_training_step = None
            self.training_step_active = False
            self.depth = 0

    def reset_calls(self) -> int:
        calls, self.calls = self.calls, 0
        return calls
