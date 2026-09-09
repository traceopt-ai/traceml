from contextlib import contextmanager
from queue import Queue
from types import SimpleNamespace

import torch.nn as nn

from traceml_ai.instrumentation import step_events
from traceml_ai.instrumentation.step_events import (
    StepCapture,
    TimeEvent,
    TimeScope,
)
from traceml_ai.integrations import lightning as lightning_integration


def _enable_callback_without_lightning(monkeypatch):
    monkeypatch.setattr(
        lightning_integration,
        "IS_LIGHTNING_AVAILABLE",
        True,
    )


def test_lightning_callback_base_combines_distinct_namespaces() -> None:
    class NewNamespaceCallback:
        pass

    class LegacyNamespaceCallback:
        pass

    resolved = lightning_integration._build_callback_base(
        (NewNamespaceCallback, LegacyNamespaceCallback)
    )

    assert resolved.available is True
    assert issubclass(resolved.base, NewNamespaceCallback)
    assert issubclass(resolved.base, LegacyNamespaceCallback)


def test_lightning_callback_base_dedupes_matching_namespaces() -> None:
    class SharedCallback:
        pass

    resolved = lightning_integration._build_callback_base(
        (SharedCallback, SharedCallback)
    )

    assert resolved.available is True
    assert resolved.base is SharedCallback


def test_lightning_forward_wrapper_times_only_forward(monkeypatch):
    _enable_callback_without_lightning(monkeypatch)
    calls = []

    @contextmanager
    def fake_timed_region(name, scope, record_gpu_events=True):
        calls.append((name, scope, record_gpu_events))
        yield

    monkeypatch.setattr(
        lightning_integration,
        "timed_region",
        fake_timed_region,
    )

    class FakeModule(nn.Module):
        def forward(self, value):
            return value + 1

    trainer = SimpleNamespace(training=True, strategy=None)
    module = FakeModule()
    callback = lightning_integration.TraceMLCallback()

    assert "forward" not in module.__dict__

    callback.setup(trainer, module)
    assert "forward" in module.__dict__

    assert module(1) == 2
    trainer.training = False
    assert module(2) == 3

    assert calls == [("_traceml_internal:forward_time", TimeScope.STEP, True)]

    callback.teardown(trainer, module)
    assert "forward" not in module.__dict__

    assert module(3) == 4
    assert calls == [("_traceml_internal:forward_time", TimeScope.STEP, True)]


def test_lightning_disabled_after_import_does_not_wrap_or_record(monkeypatch):
    _enable_callback_without_lightning(monkeypatch)
    monkeypatch.setenv("TRACEML_DISABLED", "1")
    calls = []

    @contextmanager
    def fake_timed_region(name, scope, record_gpu_events=True):
        calls.append((name, scope, record_gpu_events))
        yield

    class FakeStrategy:
        root_device = "cuda:0"

        def batch_to_device(self, batch, *args, **kwargs):
            return batch

    class FakeModule(nn.Module):
        device = "cuda:0"

        def forward(self, value):
            return value + 1

    monkeypatch.setattr(
        lightning_integration,
        "timed_region",
        fake_timed_region,
    )
    monkeypatch.setattr(
        lightning_integration,
        "StepMemoryTracker",
        lambda module: (_ for _ in ()).throw(
            AssertionError("disabled callback must not create memory tracker")
        ),
    )

    strategy = FakeStrategy()
    original_batch_to_device = strategy.batch_to_device
    trainer = SimpleNamespace(training=True, strategy=strategy)
    module = FakeModule()
    callback = lightning_integration.TraceMLCallback()

    callback.setup(trainer, module)
    assert "forward" not in module.__dict__
    assert (
        strategy.batch_to_device.__func__ is original_batch_to_device.__func__
    )

    callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
    callback.on_before_backward(trainer, module, loss=None)
    callback.on_after_backward(trainer, module)
    callback.on_before_optimizer_step(trainer, module, optimizer=None)
    callback.on_train_batch_end(
        trainer,
        module,
        outputs=None,
        batch=None,
        batch_idx=0,
    )

    assert module(1) == 2
    assert calls == []


def test_lightning_batch_start_does_not_open_forward_region(monkeypatch):
    _enable_callback_without_lightning(monkeypatch)
    calls = []

    @contextmanager
    def fake_timed_region(name, scope, record_gpu_events=True):
        calls.append((name, scope, record_gpu_events))
        yield

    class FakeMemoryTracker:
        def __init__(self, module):
            self.module = module

        def reset(self):
            return None

    monkeypatch.setattr(
        lightning_integration,
        "timed_region",
        fake_timed_region,
    )
    monkeypatch.setattr(
        lightning_integration,
        "StepMemoryTracker",
        FakeMemoryTracker,
    )

    callback = lightning_integration.TraceMLCallback()

    callback.on_train_batch_start(
        SimpleNamespace(training=True),
        object(),
        batch=None,
        batch_idx=0,
    )
    callback._close_context("_traceml_step_ctx")

    assert calls == [("_traceml_internal:step_time", "step", True)]


def test_lightning_teardown_discards_an_incomplete_capture(monkeypatch):
    _enable_callback_without_lightning(monkeypatch)
    monkeypatch.setattr(step_events, "_STEP_TIME_QUEUE", Queue(maxsize=2048))
    monkeypatch.setattr(step_events, "_ACTIVE_STEP_CAPTURE", StepCapture())

    @contextmanager
    def fake_timed_region(name, scope, record_gpu_events=True):
        yield

    class FakeMemoryTracker:
        def __init__(self, module):
            pass

        def reset(self):
            pass

    monkeypatch.setattr(
        lightning_integration, "timed_region", fake_timed_region
    )
    monkeypatch.setattr(
        lightning_integration, "StepMemoryTracker", FakeMemoryTracker
    )

    trainer = SimpleNamespace(training=True, strategy=None)
    module = nn.Linear(1, 1)
    callback = lightning_integration.TraceMLCallback()
    callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
    capture = callback._step_capture
    step_events.record_step_time_event(
        TimeEvent("failed_lightning_batch", "cpu", 1.0, 2.0)
    )

    callback.teardown(trainer, module)

    assert capture is not None
    assert not step_events.complete_step_capture(capture, 1)
    assert step_events.drain_step_time_batches() == []


class _FakeMemoryTracker:
    def __init__(self, module):
        self.module = module

    def reset(self):
        return None

    def record(self):
        return None


def _ordered_fake_timed_region(calls):
    @contextmanager
    def fake_timed_region(name, scope, record_gpu_events=True):
        calls.append(f"enter:{name}")
        try:
            yield
        finally:
            calls.append(f"exit:{name}")

    return fake_timed_region


def test_lightning_batch_to_device_opens_the_step_envelope_first(monkeypatch):
    _enable_callback_without_lightning(monkeypatch)
    calls = []
    monkeypatch.setattr(
        lightning_integration,
        "timed_region",
        _ordered_fake_timed_region(calls),
    )
    monkeypatch.setattr(
        lightning_integration, "StepMemoryTracker", _FakeMemoryTracker
    )
    monkeypatch.setattr(
        lightning_integration, "complete_step_capture", lambda *a: None
    )
    monkeypatch.setattr(
        lightning_integration, "mark_trace_step_flushed", lambda *a: None
    )

    class FakeStrategy:
        root_device = "cpu"

        def batch_to_device(self, batch, *args, **kwargs):
            calls.append("transfer")
            return batch

    strategy = FakeStrategy()
    trainer = SimpleNamespace(training=True, strategy=strategy)
    module = nn.Linear(2, 2)
    callback = lightning_integration.TraceMLCallback()
    callback.setup(trainer, module)

    strategy.batch_to_device(object())
    callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
    callback.on_train_batch_end(
        trainer, module, outputs=None, batch=None, batch_idx=0
    )

    assert calls == [
        "enter:_traceml_internal:step_time",
        "transfer",
        "exit:_traceml_internal:step_time",
    ]

    # Outside training (validation, test, predict) the transfer opens nothing.
    calls.clear()
    trainer.training = False
    strategy.batch_to_device(object())
    assert calls == ["transfer"]
    callback.teardown(trainer, module)


def test_lightning_accumulating_batch_records_no_optimizer_event(monkeypatch):
    _enable_callback_without_lightning(monkeypatch)
    monkeypatch.setattr(step_events, "_ACTIVE_STEP_CAPTURE", StepCapture())
    # The fake regions record nothing, so anything left in the capture
    # after the batch would be a fabricated event.
    monkeypatch.setattr(
        lightning_integration, "timed_region", _ordered_fake_timed_region([])
    )
    monkeypatch.setattr(
        lightning_integration, "StepMemoryTracker", _FakeMemoryTracker
    )
    monkeypatch.setattr(
        lightning_integration, "complete_step_capture", lambda *a: None
    )
    monkeypatch.setattr(
        lightning_integration, "mark_trace_step_flushed", lambda *a: None
    )
    trainer = SimpleNamespace(training=True, strategy=None)
    module = nn.Linear(2, 2)
    callback = lightning_integration.TraceMLCallback()

    callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
    callback.on_before_backward(trainer, module, loss=None)
    callback.on_after_backward(trainer, module)
    # No on_before_optimizer_step: this micro-batch only accumulated.
    callback.on_train_batch_end(
        trainer, module, outputs=None, batch=None, batch_idx=0
    )

    assert (
        step_events.begin_step_capture().timing_events == []
    ), "an accumulating batch must not fabricate events"


def test_lightning_second_optimizer_step_closes_the_previous_region(
    monkeypatch,
):
    _enable_callback_without_lightning(monkeypatch)
    calls = []
    monkeypatch.setattr(
        lightning_integration,
        "timed_region",
        _ordered_fake_timed_region(calls),
    )
    trainer = SimpleNamespace(training=True, strategy=None)
    module = nn.Linear(2, 2)
    callback = lightning_integration.TraceMLCallback()

    callback.on_before_optimizer_step(trainer, module, optimizer=None)
    callback.on_before_optimizer_step(trainer, module, optimizer=None)
    callback._close_context("_optimizer_ctx")

    assert calls == [
        "enter:_traceml_internal:optimizer_step",
        "exit:_traceml_internal:optimizer_step",
        "enter:_traceml_internal:optimizer_step",
        "exit:_traceml_internal:optimizer_step",
    ]


def test_lightning_next_forward_closes_an_open_optimizer_region(monkeypatch):
    _enable_callback_without_lightning(monkeypatch)
    calls = []
    monkeypatch.setattr(
        lightning_integration,
        "timed_region",
        _ordered_fake_timed_region(calls),
    )

    class FakeModule(nn.Module):
        def forward(self, value):
            return value

    trainer = SimpleNamespace(training=True, strategy=None)
    module = FakeModule()
    callback = lightning_integration.TraceMLCallback()
    callback.setup(trainer, module)

    callback.on_before_optimizer_step(trainer, module, optimizer=None)
    module(1)  # the next forward (manual optimization, second step)

    assert calls == [
        "enter:_traceml_internal:optimizer_step",
        "exit:_traceml_internal:optimizer_step",
        "enter:_traceml_internal:forward_time",
        "exit:_traceml_internal:forward_time",
    ]
    callback.teardown(trainer, module)


def test_lightning_on_exception_discards_and_restores(monkeypatch):
    _enable_callback_without_lightning(monkeypatch)
    calls = []
    discarded = []
    monkeypatch.setattr(
        lightning_integration,
        "timed_region",
        _ordered_fake_timed_region(calls),
    )
    monkeypatch.setattr(
        lightning_integration,
        "abort_step_capture",
        lambda capture: discarded.append("discard") or True,
    )
    monkeypatch.setattr(
        lightning_integration, "StepMemoryTracker", _FakeMemoryTracker
    )

    class FakeStrategy:
        root_device = "cpu"

        def batch_to_device(self, batch, *args, **kwargs):
            return batch

    strategy = FakeStrategy()
    original_transfer = strategy.batch_to_device
    trainer = SimpleNamespace(training=True, strategy=strategy)

    class FakeModule(nn.Module):
        def forward(self, value):
            return value

    module = FakeModule()
    callback = lightning_integration.TraceMLCallback()
    callback.setup(trainer, module)
    callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
    callback.on_before_backward(trainer, module, loss=None)
    assert callback._traceml_step_ctx is not None
    assert callback._backward_ctx is not None

    callback.on_exception(trainer, module, RuntimeError("boom"))

    assert calls[-2:] == [
        "exit:_traceml_internal:backward_time",
        "exit:_traceml_internal:step_time",
    ]
    assert discarded == ["discard"]
    assert callback._traceml_step_ctx is None
    assert callback._backward_ctx is None
    assert callback._optimizer_ctx is None
    assert "forward" not in module.__dict__
    assert strategy.batch_to_device.__func__ is original_transfer.__func__

    # A repeated call or a later teardown raises nothing and leaves the
    # callback in the same clean state (aborting an empty capture is a no-op).
    callback.on_exception(trainer, module, RuntimeError("again"))
    callback.teardown(trainer, module)
    assert len(discarded) == 3
    assert callback._traceml_step_ctx is None
    assert "forward" not in module.__dict__


def test_lightning_non_training_loops_suppress_fetch_timing(monkeypatch):
    _enable_callback_without_lightning(monkeypatch)
    events = []

    class FakeSuppress:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *exc):
            events.append("exit")
            return False

    monkeypatch.setattr(
        lightning_integration, "suppress_dataloader_timing", FakeSuppress
    )
    trainer = SimpleNamespace(training=False, strategy=None)
    module = nn.Linear(2, 2)
    callback = lightning_integration.TraceMLCallback()

    callback.on_sanity_check_start(trainer, module)
    callback.on_sanity_check_end(trainer, module)
    callback.on_validation_start(trainer, module)
    callback.on_validation_end(trainer, module)
    callback.on_test_start(trainer, module)
    callback.on_test_end(trainer, module)
    callback.on_predict_start(trainer, module)
    callback.on_predict_end(trainer, module)
    assert events == ["enter", "exit"] * 4

    # The sanity check runs the validation loop inside it: the nested
    # start/end pair must not release suppression before the outer end.
    events.clear()
    callback.on_sanity_check_start(trainer, module)
    callback.on_validation_start(trainer, module)
    callback.on_validation_end(trainer, module)
    assert events == ["enter"]
    callback.on_sanity_check_end(trainer, module)
    assert events == ["enter", "exit"]

    # An exception while suppressed releases the suppression.
    events.clear()
    callback.on_validation_start(trainer, module)
    callback.on_exception(trainer, module, RuntimeError("boom"))
    callback.on_validation_end(trainer, module)
    assert events == ["enter", "exit"]


def test_lightning_backward_closes_an_open_optimizer_region(monkeypatch):
    # Manual optimization whose second path never calls pl_module.forward:
    # the backward after opt_g.step() must end the optimizer region.
    _enable_callback_without_lightning(monkeypatch)
    calls = []
    monkeypatch.setattr(
        lightning_integration,
        "timed_region",
        _ordered_fake_timed_region(calls),
    )
    trainer = SimpleNamespace(training=True, strategy=None)
    module = nn.Linear(2, 2)
    callback = lightning_integration.TraceMLCallback()

    callback.on_before_optimizer_step(trainer, module, optimizer=None)
    callback.on_before_backward(trainer, module, loss=None)
    callback.on_after_backward(trainer, module)

    assert calls == [
        "enter:_traceml_internal:optimizer_step",
        "exit:_traceml_internal:optimizer_step",
        "enter:_traceml_internal:backward_time",
        "exit:_traceml_internal:backward_time",
    ]


def test_lightning_kill_switch_flipped_mid_batch_abandons_the_batch(
    monkeypatch,
):
    _enable_callback_without_lightning(monkeypatch)
    monkeypatch.delenv("TRACEML_DISABLED", raising=False)
    monkeypatch.setattr(step_events, "_ACTIVE_STEP_CAPTURE", StepCapture())
    discarded = []
    monkeypatch.setattr(
        lightning_integration,
        "abort_step_capture",
        lambda capture: discarded.append("discard") or True,
    )
    calls = []
    monkeypatch.setattr(
        lightning_integration,
        "timed_region",
        _ordered_fake_timed_region(calls),
    )
    monkeypatch.setattr(
        lightning_integration, "StepMemoryTracker", _FakeMemoryTracker
    )
    trainer = SimpleNamespace(training=True, strategy=None)
    module = nn.Linear(2, 2)
    callback = lightning_integration.TraceMLCallback()

    callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
    assert callback._traceml_step_ctx is not None
    monkeypatch.setenv("TRACEML_DISABLED", "1")
    callback.on_train_batch_end(
        trainer, module, outputs=None, batch=None, batch_idx=0
    )

    # The envelope was closed and the capture dropped, nothing advanced.
    assert callback._traceml_step_ctx is None
    assert discarded == ["discard"]
    assert calls == [
        "enter:_traceml_internal:step_time",
        "exit:_traceml_internal:step_time",
    ]

    # Switched back on: the next batch opens its own envelope.
    monkeypatch.delenv("TRACEML_DISABLED")
    callback.on_train_batch_start(trainer, module, batch=None, batch_idx=1)
    assert calls[-1] == "enter:_traceml_internal:step_time"
    callback._close_context("_traceml_step_ctx")


def test_lightning_teardown_discards_whatever_is_still_pending(monkeypatch):
    _enable_callback_without_lightning(monkeypatch)
    discarded = []
    monkeypatch.setattr(
        lightning_integration,
        "abort_step_capture",
        lambda capture: discarded.append("discard") or True,
    )
    trainer = SimpleNamespace(training=True, strategy=None)
    module = nn.Linear(2, 2)
    callback = lightning_integration.TraceMLCallback()
    callback.setup(trainer, module)

    callback.teardown(trainer, module)

    assert discarded == ["discard"]
    assert "forward" not in module.__dict__
