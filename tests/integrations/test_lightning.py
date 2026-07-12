from contextlib import contextmanager
from types import SimpleNamespace

import torch.nn as nn

from traceml_ai.integrations import lightning as lightning_integration
from traceml_ai.utils.timing import TimeScope


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
    def fake_timed_region(name, scope, use_gpu=True):
        calls.append((name, scope, use_gpu))
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


def test_lightning_batch_start_does_not_open_forward_region(monkeypatch):
    _enable_callback_without_lightning(monkeypatch)
    calls = []

    @contextmanager
    def fake_timed_region(name, scope, use_gpu=True):
        calls.append((name, scope, use_gpu))
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

    assert calls == [("_traceml_internal:step_time", "step", False)]
