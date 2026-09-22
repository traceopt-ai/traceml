"""TraceML handler for MONAI's ``SupervisedTrainer``.

MONAI's engines are Ignite engines. ``TraceMLHandler`` is a MONAI handler:
pass it in ``train_handlers=[...]`` and it times each training step from the
engine's own events. One TraceML step is one optimizer update, as with the
Lightning callback, so under ``accumulation_steps`` a step spans several
iterations.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

from traceml_ai.instrumentation.patches.h2d_auto_timer_patch import (
    h2d_auto_timer,
)
from traceml_ai.instrumentation.step_events import (
    TimeEvent,
    TimeScope,
    abort_step_capture,
    begin_step_capture,
    complete_step_capture,
)
from traceml_ai.runtime.state import (
    get_trace_session_state,
    mark_trace_step_flushed,
    should_record_trace_events,
)
from traceml_ai.sdk.initial import get_init_config
from traceml_ai.step_time.model import STEP_TIME_EVENT_NAMES
from traceml_ai.utils.cuda_event_pool import get_cuda_event
from traceml_ai.utils.step_memory import StepMemoryTracker
from traceml_ai.utils.timing import record_event, timed_region

try:
    from ignite.engine import Events
    from monai.engines import SupervisedTrainer
    from monai.engines.utils import IterationEvents
except Exception:  # a broken MONAI install must not break this import
    IS_MONAI_AVAILABLE = False
else:
    IS_MONAI_AVAILABLE = True

__all__ = ["TraceMLHandler", "init"]

_STEP = STEP_TIME_EVENT_NAMES["traced_step_time"]
_FETCH = STEP_TIME_EVENT_NAMES["input_wait"]
_FORWARD = STEP_TIME_EVENT_NAMES["forward"]
_BACKWARD = STEP_TIME_EVENT_NAMES["backward"]
_OPTIMIZER = STEP_TIME_EVENT_NAMES["optimizer_step"]
_RUN_SCOPED_REPORTS = frozenset(
    {
        "run start",
        "run end",
        "step open",
        "step close",
        "step memory",
        "fetch",
        "h2d timer",
        "prepare_batch",
        "detached capture",
        "unfinished group",
        "inferer wrap",
        "inferer",
        "restore",
        "forward open",
        "forward close",
        "backward open",
        "backward close",
        "optimizer open",
        "optimizer close",
        "optimizer hooks",
    }
)


def _traceml_disabled() -> bool:
    """Read the TraceML kill switch dynamically."""
    return os.environ.get("TRACEML_DISABLED") == "1"


def _warn(message: str) -> None:
    try:
        print(f"[TraceML] MONAI: {message}", file=sys.stderr)
    except Exception:
        pass


def _log_monai_error(message: str, exc: Exception) -> None:
    """Report a TraceML failure without interrupting MONAI training."""
    try:
        from traceml_ai.loggers.error_log import get_error_logger

        get_error_logger("MonaiIntegration").exception("[TraceML] %s", message)
    except Exception:
        pass
    _warn(f"{message}: {exc}")


def init():
    """
    Initialize TraceML for MONAI ``SupervisedTrainer`` runs.

    Call once before building the trainer, then pass ``TraceMLHandler()`` in
    ``train_handlers``. H2D timing is armed; the DataLoader fetch patch is
    not. The handler times Input Wait from the engine's own fetch events on
    the training thread, so a loader that fetches on a background thread,
    such as ``ThreadDataLoader``, is timed by what the loop waited.
    """
    import traceml_ai as traceml

    return traceml.init(mode="selective", patch_h2d=True)


def _unsupported_reason(engine) -> Optional[str]:
    """Return why the handler cannot read ``engine``, or None if it can."""
    name = type(engine).__name__
    if not isinstance(engine, SupervisedTrainer):
        return f"{name} is not a SupervisedTrainer"
    # Compare the function, not bound methods: ``engine._iteration`` resolves
    # to a subclass override, and every access builds a new bound method.
    process = getattr(engine, "_process_function", None)
    function = getattr(process, "__func__", None)
    if getattr(process, "__self__", None) is not engine or (
        function is not SupervisedTrainer._iteration
    ):
        return (
            f"{name} replaces SupervisedTrainer's iteration "
            "(iteration_update= or an overridden _iteration)"
        )
    return None


def _ends_update_group(engine) -> bool:
    """
    Return whether the iteration that just ran stepped the optimizer.

    MONAI decides this inline in ``SupervisedTrainer._iteration`` and does
    not expose it, so the rule is repeated here as MONAI 1.6.0 states it in
    ``monai/engines/trainer.py`` (the ``should_step`` block), including the
    forced step at the end of an epoch of known length. The tests pin it
    against real ``optimizer.step()`` calls, and the CI job pins the MONAI
    minor, so a changed rule shows up as a failure rather than as drift.
    """
    accumulation = engine.accumulation_steps or 1
    if accumulation <= 1:
        return True
    iteration = engine.state.iteration
    epoch_length = engine.state.epoch_length
    if epoch_length is None:
        return iteration % accumulation == 0
    position = (iteration - 1) % epoch_length + 1
    return position % accumulation == 0 or position == epoch_length


def _zeroes_gradients(engine, iteration: int) -> bool:
    """
    Return whether MONAI is about to call ``optimizer.zero_grad()`` for the
    given upcoming iteration.

    Mirrors MONAI 1.6.0's ``should_zero_grad``, from the same ``should_step``
    block ``_ends_update_group`` reads. In steady state this is only True at
    the start of a fresh window, right after the previous one stepped, which
    is also exactly when the handler's capture is already ``None``. It
    diverges only for a loader with no length: ``epoch_length`` is unknown
    for the whole first epoch, so the position basis is the raw iteration
    count, and it changes under a window that is still open once Ignite
    learns the length at that epoch's end. MONAI zeroes that window's
    gradients before ``optimizer.step()`` ever sees them, so its
    measurements must be dropped rather than merged into whatever group
    steps next.

    ``iteration`` is passed explicitly rather than read from
    ``engine.state.iteration`` because the caller is ``GET_BATCH_STARTED``,
    which fires before Ignite increments it for the iteration that is
    starting; at that point ``engine.state.iteration`` still names the
    iteration that just finished.
    """
    accumulation = engine.accumulation_steps or 1
    if accumulation <= 1:
        return True
    epoch_length = engine.state.epoch_length
    local_iteration = (
        (iteration - 1) % epoch_length
        if epoch_length is not None
        else iteration - 1
    )
    return local_iteration % accumulation == 0


def _fetch_patch_installed() -> bool:
    """Whether the torch DataLoader fetch patch is installed right now."""
    try:
        from torch.utils.data import DataLoader

        return bool(getattr(DataLoader, "_traceml_patched", False))
    except Exception:
        return False


def _is_cuda(device) -> bool:
    import torch

    try:
        return torch.device(device).type == "cuda"
    except Exception:
        return False


def _enter_h2d():
    """Open the H2D timing window and return the context that closes it."""
    timer = h2d_auto_timer()
    timer.__enter__()
    return timer


def _start_fetch_clock():
    """Stamp a fetch start the way ``timed_region`` stamps a region start."""
    import torch

    if _traceml_disabled() or not should_record_trace_events():
        # Past the step budget nothing is published, so take no CUDA events.
        return None
    cpu_start = time.time()
    if not torch.cuda.is_available():
        return cpu_start, None, None, "cpu"
    start, end = get_cuda_event(), get_cuda_event()
    start.record()
    return cpu_start, start, end, f"cuda:{torch.cuda.current_device()}"


def _record_fetch(clock) -> None:
    cpu_start, start, end, device = clock
    if start is not None:
        end.record()
    record_event(
        TimeEvent(
            name=_FETCH,
            device=device,
            cpu_start=cpu_start,
            cpu_end=time.time(),
            gpu_start=start,
            gpu_end=end,
            scope=TimeScope.STEP,
        )
    )


class _TimedInferer:
    """Time the model call MONAI makes, and pass everything else through."""

    def __init__(self, handler, inferer):
        self._handler, self._inferer = handler, inferer

    def __call__(self, *args, **kwargs):
        handler = self._handler
        if not handler._inside_iteration():
            # A call outside prepare_batch..MODEL_COMPLETED belongs to no
            # step. Timing it would attribute its forward time to whatever
            # step opens next, the same class of bug the optimizer hooks
            # guard against with this same check.
            return self._inferer(*args, **kwargs)
        handler._saw_forward = True
        handler._best_effort("forward open", handler._open, _FORWARD)
        try:
            return self._inferer(*args, **kwargs)
        finally:
            handler._best_effort("forward close", handler._close, _FORWARD)

    def __getattr__(self, name):
        # __dict__ directly: a copied or unpickled proxy has no _inferer yet,
        # and self._inferer would recurse through this method forever.
        inferer = self.__dict__.get("_inferer")
        if inferer is None:
            raise AttributeError(name)
        return getattr(inferer, name)

    def __setattr__(self, name, value):
        # Writes go to the real inferer, not to this wrapper. Otherwise a
        # handler that stores state on engine.inferer during a run would
        # lose it when the original is restored, which is tracing changing
        # what training leaves behind.
        inferer = self.__dict__.get("_inferer")
        if inferer is None or name in ("_handler", "_inferer"):
            object.__setattr__(self, name, value)
        else:
            setattr(inferer, name, value)


class TraceMLHandler:
    """
    MONAI handler that times ``SupervisedTrainer`` steps for TraceML.

    It traces one trainer that runs ``SupervisedTrainer._iteration`` itself,
    including a subclass that inherits it. Any other engine gets one warning
    and nothing attached. Each run first drops step events recorded earlier
    in the process. Every exit closes open regions, drops an unfinished step
    and restores ``prepare_batch``. Nothing is registered on
    ``Events.EXCEPTION_RAISED``, so the user's exception always propagates.
    """

    def __init__(self):
        if not IS_MONAI_AVAILABLE:
            raise ImportError(
                "Install 'monai' and 'pytorch-ignite' to use TraceML's MONAI "
                "integration: pip install 'traceml-ai[monai]'."
            )
        self._engine = None
        self._active = False
        self._cuda = False
        self._regions = {}
        self._capture = None
        self._mem_tracker = None
        self._fetch_clock = None
        # attribute name -> (original, ours) while a run is traced
        self._wrapped = {}
        self._opt_handles = []
        self._opt_depth = 0
        self._saw_forward = False
        self._reported = set()

    def attach(self, engine) -> None:
        """Called by MONAI for each engine the handler is listed on."""
        self._best_effort("attach", self._attach, engine)

    def _attach(self, engine) -> None:
        reason = _unsupported_reason(engine)
        if reason is not None:
            _warn(f"{reason}; this engine is not traced.")
            return
        if getattr(engine, "_traceml_handler", None) is not None:
            _warn(
                "this trainer is already traced by a TraceMLHandler; use one "
                "handler per trainer. This attach is ignored."
            )
            return
        if self._engine is not None:
            _warn(
                "this TraceMLHandler already traces a trainer; use one "
                "handler per trainer. This attach is ignored."
            )
            return
        self._engine = engine
        engine._traceml_handler = self
        original_run = engine.run

        def run(*args, **kwargs):
            self._best_effort("run start", self._start_run, engine)
            try:
                return original_run(*args, **kwargs)
            finally:
                self._best_effort("run end", self._end_run, engine)

        engine.run = run
        engine.add_event_handler(Events.GET_BATCH_STARTED, self._fetch_start)
        engine.add_event_handler(Events.GET_BATCH_COMPLETED, self._fetch_end)
        engine.add_event_handler(
            IterationEvents.LOSS_COMPLETED, self._loss_completed
        )
        engine.add_event_handler(
            IterationEvents.BACKWARD_COMPLETED, self._backward_completed
        )
        engine.add_event_handler(
            IterationEvents.MODEL_COMPLETED, self._model_completed
        )

    def _best_effort(self, where: str, fn, *args):
        try:
            return fn(*args)
        except Exception as exc:
            if where not in self._reported:
                self._reported.add(where)
                _log_monai_error(f"{where} failed", exc)
            return None

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._reported:
            self._reported.add(key)
            _warn(message)

    def _start_run(self, engine) -> None:
        self._active = False
        if _traceml_disabled():
            return
        config = get_init_config()
        armed = config is not None and config.patch_dataloader
        if armed or _fetch_patch_installed():
            self._warn_once(
                "fetch patch",
                "the torch DataLoader fetch patch is installed, for "
                "example by traceml.init(), and it would count every fetch "
                "a second time. Use traceml_ai.integrations.monai.init() "
                "as the only TraceML init in this process. This trainer is "
                "not traced.",
            )
            return
        if not hasattr(engine, "accumulation_steps"):
            # MONAI sets it after the handlers attach, so this is read here.
            # Without it the step boundary would be guessed, not read.
            self._warn_once(
                "accumulation",
                "this trainer does not expose accumulation_steps, so the "
                "step boundary cannot be read. It is not traced.",
            )
            return
        self._cuda = _is_cuda(getattr(engine.state, "device", None))
        if self._cuda and (config is None or not config.patch_h2d):
            self._warn_once(
                "h2d",
                "H2D transfers are not timed because H2D timing is off. "
                "Call traceml_ai.integrations.monai.init() before building "
                "the trainer.",
            )
        # Events recorded before this run do not belong to its first step.
        abort_step_capture(begin_step_capture())
        self._regions, self._capture = {}, None
        self._mem_tracker, self._fetch_clock = None, None
        self._opt_depth = 0
        # Run-scoped failures report again next run; the config warnings
        # above stay once per handler.
        self._reported -= _RUN_SCOPED_REPORTS
        self._wrap_prepare_batch(engine)
        # Each phase installs behind its own boundary: one that cannot be
        # installed costs its own stream, never the whole run.
        self._best_effort("inferer wrap", self._wrap_inferer, engine)
        self._best_effort("optimizer hooks", self._hook_optimizer, engine)
        self._active = True

    def _end_run(self, engine) -> None:
        was_active, self._active = self._active, False
        self._fetch_clock = None
        # If optimizer.step() raised between the pre- and post-hook, torch
        # skips the post-hook and depth would stick above 0. Nothing today
        # depends on that (the run is already ending), but "depth is 0
        # outside a run" should not depend on _start_run always following.
        self._opt_depth = 0
        try:
            if was_active:
                # A group the run did not finish is dropped, never published.
                self._abandon()
        finally:
            # The wrappers come off even when dropping the group failed.
            for handle in self._opt_handles:
                self._best_effort("optimizer hooks", handle.remove)
            self._opt_handles = []
            wrapped, self._wrapped = self._wrapped, {}
            for name, (original, ours) in wrapped.items():
                self._best_effort(
                    "restore", self._restore, engine, name, original, ours
                )

    @staticmethod
    def _restore(engine, name, original, ours) -> None:
        # A replacement made during the run is left alone.
        if getattr(engine, name, None) is ours:
            setattr(engine, name, original)

    def _wrap_prepare_batch(self, engine) -> None:
        original = engine.prepare_batch

        def prepare_batch(*args, **kwargs):
            if not self._active or _traceml_disabled():
                return original(*args, **kwargs)
            # The iteration starts here, so the transfer is inside the step.
            self._best_effort("step open", self._open_step, engine)
            timer = (
                self._best_effort("h2d timer", _enter_h2d)
                if self._cuda
                else None
            )
            try:
                return original(*args, **kwargs)
            finally:
                if timer is not None:
                    self._best_effort(
                        "h2d timer", timer.__exit__, None, None, None
                    )

        engine.prepare_batch = prepare_batch
        self._wrapped["prepare_batch"] = (original, prepare_batch)

    def _wrap_inferer(self, engine) -> None:
        """MONAI calls the inferer to run the model (``trainer.py:253``)."""
        original = engine.inferer
        timed = _TimedInferer(self, original)
        engine.inferer = timed
        self._wrapped["inferer"] = (original, timed)

    def _hook_optimizer(self, engine) -> None:
        """Time the real ``step()``, including the one AMP's scaler makes."""
        optimizer = engine.optimizer
        if getattr(optimizer, "_traceml_step_instance_wrapped", False):
            # traceml.wrap_optimizer() already times this one.
            return
        if not hasattr(optimizer, "register_step_pre_hook"):
            # MONAI accepts any object with step() and zero_grad().
            self._warn_once(
                "optimizer hooks",
                "this optimizer does not support torch step hooks, so "
                "optimizer time is not measured. Every other phase is.",
            )
            return
        # Appended one at a time: a handle that is registered but lost
        # cannot be removed at the end of the run.
        handles = []
        self._opt_handles = handles
        handles.append(
            optimizer.register_step_pre_hook(self._optimizer_started)
        )
        handles.append(
            optimizer.register_step_post_hook(self._optimizer_finished)
        )

    def _optimizer_started(self, optimizer, args, kwargs) -> None:
        # torch patches step() per class, so a subclass that calls
        # super().step() fires this pair twice. Only the outer one counts.
        if not self._inside_iteration():
            return
        self._opt_depth += 1
        if self._opt_depth == 1:
            self._best_effort("optimizer open", self._open, _OPTIMIZER)

    def _optimizer_finished(self, optimizer, args, kwargs) -> None:
        if self._opt_depth == 0:
            return
        self._opt_depth -= 1
        if self._opt_depth == 0 and self._inside_iteration():
            self._best_effort("optimizer close", self._close, _OPTIMIZER)

    def _inside_iteration(self) -> bool:
        """True while MONAI is between prepare_batch and MODEL_COMPLETED."""
        return (
            self._active and _STEP in self._regions and not _traceml_disabled()
        )

    def _loss_completed(self, engine) -> None:
        if self._active and not _traceml_disabled():
            self._best_effort("backward open", self._open, _BACKWARD)

    def _backward_completed(self, engine) -> None:
        if self._active and not _traceml_disabled():
            self._best_effort("backward close", self._close, _BACKWARD)

    def _open_step(self, engine) -> None:
        self._saw_forward = False
        if self._capture is not None and self._capture is not (
            begin_step_capture()
        ):
            # Another producer detached the process-wide capture while this
            # group was open, so its events are gone. Start a new group
            # rather than finish one that can no longer publish.
            self._warn_once(
                "detached capture",
                "another TraceML step producer took the active capture "
                "mid-step, so this optimizer-update group restarts and "
                "covers fewer iterations than the trainer ran.",
            )
            self._abandon()
        if self._capture is None:
            # First iteration of an optimizer-update group.
            self._capture = begin_step_capture()
            self._mem_tracker = self._best_effort(
                "step memory", self._new_memory_tracker, engine
            )
        self._open(_STEP)

    @staticmethod
    def _new_memory_tracker(engine):
        tracker = StepMemoryTracker(engine.network)
        tracker.reset()
        return tracker

    def _fetch_start(self, engine) -> None:
        if self._active and not _traceml_disabled():
            self._best_effort(
                "unfinished group", self._discard_if_zeroed, engine
            )
            # Replacing an earlier start drops it unrecorded. Ignite ends the
            # first epoch of a loader with no length with a start that has
            # no matching end.
            self._fetch_clock = self._best_effort("fetch", _start_fetch_clock)

    def _discard_if_zeroed(self, engine) -> None:
        if self._capture is not None and _zeroes_gradients(
            engine, engine.state.iteration + 1
        ):
            # MONAI is about to zero the gradients this window
            # accumulated without ever stepping them (an epoch boundary
            # landed mid-accumulation on a loader with no length). Its
            # measurements never reached a real update, so drop them
            # before this iteration's own fetch is recorded into the
            # same capture. This is the earliest hook that fires for a
            # new iteration, ahead of ``prepare_batch``.
            self._warn_once(
                "unfinished group",
                "an unfinished optimizer-update group was discarded "
                "because MONAI zeroed its gradients before stepping. "
                "Its measurements are dropped, not carried into the "
                "next step.",
            )
            self._abandon()

    def _fetch_end(self, engine) -> None:
        clock, self._fetch_clock = self._fetch_clock, None
        if clock is not None and self._active:
            self._best_effort("fetch", _record_fetch, clock)

    def _model_completed(self, engine) -> None:
        if self._active:
            self._best_effort("step close", self._close_step, engine)

    def _close_step(self, engine) -> None:
        opened = _STEP in self._regions
        if not opened and not _traceml_disabled():
            wrapper = self._wrapped.get("prepare_batch", (None, None))[1]
            if wrapper is not None and engine.prepare_batch is not wrapper:
                self._warn_once(
                    "prepare_batch",
                    "prepare_batch was replaced after the run started, so "
                    "this trainer is no longer traced.",
                )
        if opened and not self._saw_forward:
            proxy = self._wrapped.get("inferer", (None, None))[1]
            if proxy is not None and engine.inferer is not proxy:
                self._warn_once(
                    "inferer",
                    "the inferer was replaced after the run started, so "
                    "forward time is no longer measured.",
                )
        if _traceml_disabled() or not opened:
            # The kill switch flipped mid-run, or no step opened in this
            # iteration: drop the group rather than leave events pending.
            self._abandon()
            return
        self._close_all()
        if not _ends_update_group(engine):
            return
        tracker, self._mem_tracker = self._mem_tracker, None
        if tracker is not None:
            self._best_effort("step memory", tracker.record)
        state = get_trace_session_state()
        state.advance_step()
        capture, self._capture = self._capture, None
        complete_step_capture(capture, state.step)
        mark_trace_step_flushed(state.step)

    def _abandon(self) -> None:
        self._close_all()
        capture, self._capture = self._capture, None
        self._mem_tracker = None
        abort_step_capture(
            capture if capture is not None else begin_step_capture()
        )

    def _open(self, name: str) -> None:
        self._close(name)
        region = timed_region(name, scope=TimeScope.STEP)
        region.__enter__()
        self._regions[name] = region

    def _close(self, name: str) -> None:
        region = self._regions.pop(name, None)
        if region is not None:
            region.__exit__(None, None, None)

    def _close_all(self) -> None:
        for name in list(self._regions):
            self._close(name)
