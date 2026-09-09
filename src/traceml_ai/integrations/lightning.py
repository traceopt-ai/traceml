import functools
import os
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from traceml_ai.instrumentation.patches.dataloader_patch import (
    suppress_dataloader_timing,
)
from traceml_ai.instrumentation.patches.h2d_auto_timer_patch import (
    h2d_auto_timer,
)
from traceml_ai.instrumentation.step_events import (
    TimeScope,
    abort_step_capture,
    begin_step_capture,
    complete_step_capture,
)
from traceml_ai.runtime.state import (
    get_trace_session_state,
    mark_trace_step_flushed,
)
from traceml_ai.utils.step_memory import StepMemoryTracker
from traceml_ai.utils.timing import timed_region

if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback as _CallbackBase
else:
    _CallbackBase = object


def _dedupe_callback_bases(bases: tuple[type, ...]) -> tuple[type, ...]:
    """Return callback bases without duplicates while preserving order."""
    out: list[type] = []
    for base in bases:
        if any(base is existing for existing in out):
            continue
        out.append(base)
    return tuple(out)


def _build_callback_base(bases: tuple[type, ...]) -> Any:
    """Build one runtime callback base from available namespace bases."""
    unique_bases = _dedupe_callback_bases(bases)
    if not unique_bases:
        return SimpleNamespace(base=object, available=False)

    if len(unique_bases) == 1:
        return SimpleNamespace(base=unique_bases[0], available=True)

    try:
        base = type(
            "_TraceMLLightningCallbackBase",
            unique_bases,
            {},
        )
    except TypeError:
        # If the namespaces expose incompatible callback classes, prefer the
        # legacy package because that is the namespace most likely to hit the
        # mixed-import error this compatibility path fixes.
        base = unique_bases[-1]

    return SimpleNamespace(base=base, available=True)


def _resolve_callback_base() -> Any:
    """
    Build a callback base accepted by either Lightning namespace.

    ``lightning.pytorch`` and ``pytorch_lightning`` can be installed side by
    side, but their ``Callback`` classes are not always identical. Inheriting
    from every available callback base lets the same TraceML callback work with
    whichever Trainer namespace the user already has.
    """
    bases: list[type] = []

    try:
        from lightning.pytorch.callbacks import Callback as LightningCallback

        bases.append(LightningCallback)
    except ImportError:
        pass

    try:
        from pytorch_lightning.callbacks import (
            Callback as PyTorchLightningCallback,
        )

        bases.append(PyTorchLightningCallback)
    except ImportError:
        pass

    return _build_callback_base(tuple(bases))


if not TYPE_CHECKING:
    _callback_resolution = _resolve_callback_base()
    _CallbackBase = _callback_resolution.base
    IS_LIGHTNING_AVAILABLE = bool(_callback_resolution.available)
else:
    IS_LIGHTNING_AVAILABLE = True


_MISSING = object()


def _traceml_disabled() -> bool:
    """Read the TraceML kill switch dynamically."""
    return os.environ.get("TRACEML_DISABLED") == "1"


def init():
    """
    Initialize TraceML for PyTorch Lightning runs.

    Lightning owns the training loop, so TraceMLCallback owns step boundaries,
    capture completion, and framework hook integration. The integration init
    enables DataLoader fetch timing plus the H2D Tensor.to patch. The callback
    opens the traced step around Lightning's batch transfer (so H2D is inside
    it), turns H2D timing on only there, and wraps LightningModule.forward
    directly for model-forward timing. Fetches of non-training loaders are
    excluded.
    """
    import traceml_ai as traceml

    return traceml.init(
        mode="selective",
        patch_dataloader=True,
        patch_h2d=True,
    )


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


class TraceMLCallback(_CallbackBase):
    """
    Official TraceML Callback for PyTorch Lightning.

    One TraceML step is one Lightning training batch. The traced step opens
    when Lightning moves the batch to the device (so the H2D transfer is
    inside Traced Step Time) and closes at ``on_train_batch_end``. Forward,
    backward and optimizer phases are timed individually. Under gradient
    accumulation every micro-batch is still a step; the optimizer phase is
    recorded only on batches where the optimizer ran, and absent on the
    others. Fetches of validation, sanity-check, test and predict loaders are
    kept out of Input Wait. A batch that raises is discarded, not published.
    """

    def __init__(self):
        if not IS_LIGHTNING_AVAILABLE:
            raise ImportError(
                "Install either 'lightning' or 'pytorch-lightning' to use "
                "TraceML's Lightning integration."
            )
        super().__init__()
        self._traceml_step_ctx = None
        self._backward_ctx = None
        self._optimizer_ctx = None
        self._batch_to_device_strategy = None
        self._original_batch_to_device = None
        self._forward_module = None
        self._original_forward = None
        self._original_forward_attr = _MISSING
        self._wrapped_forward = None
        self._step_capture = None
        self._suppress_cm = None
        self._suppress_depth = 0

        self._mem_tracker = None

    def _close_context(self, ctx_attr: str) -> None:
        ctx = getattr(self, ctx_attr, None)
        if ctx is None:
            return
        try:
            ctx.__exit__(None, None, None)
        except Exception as e:
            _log_lightning_error(f"{ctx_attr} cleanup failed", e)
        setattr(self, ctx_attr, None)

    def _close_all_contexts(self) -> None:
        for ctx_attr in (
            "_backward_ctx",
            "_optimizer_ctx",
            "_traceml_step_ctx",
        ):
            self._close_context(ctx_attr)

    def _open_step_region(self) -> None:
        """
        Open the traced step once; later callers find it already open.

        Opening also takes the capture this batch completes. ``begin`` adopts
        the active capture, so the fetch and H2D events recorded before this
        point in the batch stay with it.
        """
        if self._traceml_step_ctx is not None:
            return
        self._step_capture = begin_step_capture()
        try:
            ctx = timed_region(
                "_traceml_internal:step_time",
                scope=TimeScope.STEP,
                record_gpu_events=True,
            )
            ctx.__enter__()
        except Exception as e:
            _log_lightning_error("step region open failed", e)
            return
        self._traceml_step_ctx = ctx

    def _enter_suppression(self) -> None:
        """
        Keep a non-training loader's fetches out of Input Wait.

        Refcounted: the sanity check runs the validation loop inside it, so
        the hooks nest (sanity start, validation start, validation end,
        sanity end) and suppression must hold until the outermost end.
        """
        self._suppress_depth += 1
        if self._suppress_cm is not None:
            return
        try:
            cm = suppress_dataloader_timing()
            cm.__enter__()
        except Exception as e:
            _log_lightning_error("fetch suppression enter failed", e)
            return
        self._suppress_cm = cm

    def _exit_suppression(self, force: bool = False) -> None:
        self._suppress_depth = 0 if force else max(0, self._suppress_depth - 1)
        if self._suppress_depth > 0:
            return
        cm = self._suppress_cm
        if cm is None:
            return
        self._suppress_cm = None
        try:
            cm.__exit__(None, None, None)
        except Exception as e:
            _log_lightning_error("fetch suppression exit failed", e)

    def setup(self, trainer, pl_module, stage=None):
        if _traceml_disabled():
            return
        self._wrap_forward(trainer, pl_module)
        self._wrap_batch_to_device(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        if _traceml_disabled():
            return
        # Fail loud (never raise) when the init config will not capture the
        # patch-gated streams this callback owes. Forward, backward, optimizer
        # and the step envelope are timed by the callback itself.
        try:
            from traceml_ai.integrations._capability import (
                warn_if_missing_streams,
            )

            warn_if_missing_streams(
                "Lightning TraceMLCallback",
                requires={"dataloader_fetch", "h2d"},
            )
        except Exception:
            pass
        if self._original_batch_to_device is None:
            # The config check above cannot see this: without the wrapper the
            # H2D transfer is not timed and the traced step opens late.
            print(
                "[TraceML] Lightning strategy.batch_to_device is not wrapped; "
                "H2D will not be timed and the traced step starts at "
                "on_train_batch_start.",
                file=sys.stderr,
            )

    def _wrap_forward(self, trainer, pl_module) -> None:
        if self._original_forward is not None:
            return

        original_forward = getattr(pl_module, "forward", None)
        if not callable(original_forward):
            return
        original_forward_attr = getattr(pl_module, "__dict__", {}).get(
            "forward", _MISSING
        )

        @functools.wraps(original_forward)
        def wrapped_forward(*args, **kwargs):
            if _traceml_disabled() or not getattr(trainer, "training", False):
                return original_forward(*args, **kwargs)

            # A forward after an optimizer step (manual optimization with
            # several steps per batch) ends that step's region; it must not
            # absorb the next forward.
            self._close_context("_optimizer_ctx")
            with timed_region(
                "_traceml_internal:forward_time",
                scope=TimeScope.STEP,
                record_gpu_events=True,
            ):
                return original_forward(*args, **kwargs)

        try:
            pl_module.forward = wrapped_forward
        except Exception as e:
            _log_lightning_error("forward wrap failed", e)
            return

        self._forward_module = pl_module
        self._original_forward = original_forward
        self._original_forward_attr = original_forward_attr
        self._wrapped_forward = wrapped_forward

    def _wrap_batch_to_device(self, trainer, pl_module) -> None:
        if self._original_batch_to_device is not None:
            return

        strategy = getattr(trainer, "strategy", None)
        if strategy is None:
            return

        original = strategy.batch_to_device

        def wrapped_batch_to_device(batch, *args, **kwargs):
            if _traceml_disabled() or not getattr(trainer, "training", True):
                return original(batch, *args, **kwargs)
            # Lightning moves the batch before on_train_batch_start fires, so
            # the traced step opens here to keep the transfer inside it.
            self._open_step_region()
            if not _lightning_uses_cuda(trainer, pl_module):
                return original(batch, *args, **kwargs)
            with h2d_auto_timer():
                return original(batch, *args, **kwargs)

        try:
            strategy.batch_to_device = wrapped_batch_to_device
        except Exception as e:
            _log_lightning_error("H2D batch transfer wrap failed", e)
            return

        self._batch_to_device_strategy = strategy
        self._original_batch_to_device = original

    def teardown(self, trainer, pl_module, stage=None):
        # Events recorded after the last completed step (the fetch that raised
        # StopIteration at the end of an epoch, or a batch cut short by the
        # kill switch) must not leak into the next fit in this process.
        self._abandon_pending(pl_module)
        self._restore_forward()
        self._restore_batch_to_device()

    def _abandon_pending(self, pl_module) -> None:
        """
        Close open regions and drop everything recorded since the last step.

        Nothing is published and the step counter is left where it was. The
        active capture is process-wide, so this also drops any other
        step-scoped events recorded since the last completed step.
        """
        try:
            self._close_all_contexts()
            self._exit_suppression(force=True)
            capture = self._step_capture
            self._step_capture = None
            abort_step_capture(
                capture if capture is not None else begin_step_capture()
            )
            self._mem_tracker = None
        except Exception as e:
            _log_lightning_error("pending-step cleanup failed", e)

    def on_exception(self, trainer, pl_module, exception):
        """
        Abandon the current batch without publishing it.

        Lightning calls this hook, not ``teardown``, when training raises.
        The open timing regions are closed, the pending timing events and
        memory snapshot are discarded, the step counter is left where it was,
        and the wrappers are restored. The user's exception propagates
        untouched.
        """
        self._abandon_pending(pl_module)
        self._restore_forward()
        self._restore_batch_to_device()

    # Non-training loops: their loader fetches are not training input wait.
    # Deliberately not gated on TRACEML_DISABLED: suppression is harmless when
    # disabled and must still pair correctly if the switch flips back on.
    def on_sanity_check_start(self, trainer, pl_module):
        self._enter_suppression()

    def on_sanity_check_end(self, trainer, pl_module):
        self._exit_suppression()

    def on_validation_start(self, trainer, pl_module):
        self._enter_suppression()

    def on_validation_end(self, trainer, pl_module):
        self._exit_suppression()

    def on_test_start(self, trainer, pl_module):
        self._enter_suppression()

    def on_test_end(self, trainer, pl_module):
        self._exit_suppression()

    def on_predict_start(self, trainer, pl_module):
        self._enter_suppression()

    def on_predict_end(self, trainer, pl_module):
        self._exit_suppression()

    def _restore_forward(self) -> None:
        module = self._forward_module
        original = self._original_forward
        if module is None or original is None:
            return

        try:
            current_forward_attr = getattr(module, "__dict__", {}).get(
                "forward", _MISSING
            )
            if current_forward_attr is self._wrapped_forward:
                if self._original_forward_attr is _MISSING:
                    delattr(module, "forward")
                else:
                    module.forward = self._original_forward_attr
        except Exception as e:
            _log_lightning_error("forward restore failed", e)
        finally:
            self._forward_module = None
            self._original_forward = None
            self._original_forward_attr = _MISSING
            self._wrapped_forward = None

    def _restore_batch_to_device(self) -> None:
        strategy = self._batch_to_device_strategy
        original = self._original_batch_to_device
        if strategy is None or original is None:
            return

        try:
            strategy.batch_to_device = original
        except Exception as e:
            _log_lightning_error("H2D batch transfer restore failed", e)
        finally:
            self._batch_to_device_strategy = None
            self._original_batch_to_device = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if _traceml_disabled():
            return
        # Normally already open from the batch_to_device wrapper; this is the
        # fallback for loops that hand the iterator to training_step and
        # never call batch_to_device.
        self._open_step_region()

        # Reset step memory
        try:
            mem_tracker = StepMemoryTracker(pl_module)
            mem_tracker.reset()
            self._mem_tracker = mem_tracker
        except Exception as e:
            _log_lightning_error("memory reset failed", e)
            self._mem_tracker = None

    def on_before_backward(self, trainer, pl_module, loss):
        if _traceml_disabled():
            return
        # A backward after an optimizer step (manual optimization, several
        # steps per batch) ends that step's region; it must not absorb the
        # next backward.
        self._close_context("_optimizer_ctx")
        self._close_context("_backward_ctx")
        self._backward_ctx = timed_region(
            "_traceml_internal:backward_time", scope=TimeScope.STEP
        )
        self._backward_ctx.__enter__()

    def on_after_backward(self, trainer, pl_module):
        if _traceml_disabled():
            return
        # End backward timing
        self._close_context("_backward_ctx")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if _traceml_disabled():
            return

        # Lightning fires this after the closure (training_step, zero_grad,
        # backward) and before optimizer.step(). The region stays open until
        # on_train_batch_end, so on Lightning it also covers the step-interval
        # LR scheduler update. Manual optimization may step more than once per
        # batch: close the previous region first so each step is one event.
        self._close_context("_optimizer_ctx")
        self._optimizer_ctx = timed_region(
            "_traceml_internal:optimizer_step", scope=TimeScope.STEP
        )
        self._optimizer_ctx.__enter__()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if _traceml_disabled():
            # The kill switch was flipped during this batch. Close what was
            # opened and drop what was buffered so the next batch starts on a
            # fresh envelope instead of merging into this one.
            self._abandon_pending(pl_module)
            return
        # Close the optimizer region, any backward region left open, and the
        # step envelope. On an accumulating micro-batch no optimizer region
        # was opened, so no optimizer event exists for this step: the
        # reporting side treats that absence as "did not occur", never as a
        # measured zero.
        self._close_all_contexts()

        # Record step memory
        if self._mem_tracker is not None:
            try:
                self._mem_tracker.record()
            except Exception as e:
                _log_lightning_error("record failed", e)

        # Advance and complete the capture (treating every micro-batch as a step
        # to preserve fine-grained forward/backward times)
        trace_state = get_trace_session_state()
        trace_state.advance_step()
        capture, self._step_capture = self._step_capture, None
        try:
            # The envelope always opens before this hook, so the capture is
            # normally the one this batch began. Falling back to the active
            # capture keeps a batch that somehow started without one from
            # holding its events back into the next step.
            if capture is None:
                capture = begin_step_capture()
            complete_step_capture(capture, trace_state.step)
        except Exception as e:
            _log_lightning_error("step capture completion failed", e)

        try:
            mark_trace_step_flushed(trace_state.step)
        except Exception as e:
            _log_lightning_error("recording state update failed", e)


__all__ = ["TraceMLCallback", "init"]
