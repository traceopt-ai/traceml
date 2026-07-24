"""Tests for runtime startup driven by `traceml.init(...)` (issue #84)."""

import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _FakeHandle:
    """Minimal RuntimeHandle stand-in that records stop() calls."""

    def __init__(self) -> None:
        self.stops = 0

    def stop(self) -> None:
        self.stops += 1


@pytest.fixture
def initialization():
    """Reload sdk.initial and reset process-global runtime state per test.

    ``init(disabled=...)`` writes ``TRACEML_DISABLED`` straight to os.environ,
    so this fixture snapshots and restores it to avoid leaking into other
    tests (monkeypatch.delenv does not register a restore for an absent var).
    """
    import traceml_ai.runtime.lifecycle as lifecycle

    original_disabled = os.environ.get("TRACEML_DISABLED")
    os.environ.pop("TRACEML_DISABLED", None)
    lifecycle._ACTIVE_RUNTIME_HANDLE = None
    module = importlib.reload(
        importlib.import_module("traceml_ai.sdk.initial")
    )
    try:
        yield module
    finally:
        lifecycle._ACTIVE_RUNTIME_HANDLE = None
        if original_disabled is None:
            os.environ.pop("TRACEML_DISABLED", None)
        else:
            os.environ["TRACEML_DISABLED"] = original_disabled


def test_init_disabled_is_noop(initialization, monkeypatch):
    monkeypatch.delenv("TRACEML_DISABLED", raising=False)

    started = []
    patched = []
    monkeypatch.setattr(
        initialization,
        "_start_runtime_for_init",
        lambda **kwargs: started.append(kwargs),
    )
    monkeypatch.setattr(
        initialization,
        "_apply_requested_patches",
        lambda cfg: patched.append(cfg),
    )

    cfg = initialization.init(disabled=True)

    assert cfg.disabled is True
    assert started == []  # runtime never started
    assert patched == []  # no instrumentation patches installed
    assert os.environ["TRACEML_DISABLED"] == "1"


def test_init_disabled_via_env(initialization, monkeypatch):
    import traceml_ai.runtime.lifecycle as lifecycle

    monkeypatch.setenv("TRACEML_DISABLED", "1")
    started = []
    monkeypatch.setattr(lifecycle, "get_active_runtime_handle", lambda: None)
    monkeypatch.setattr(
        lifecycle, "start_runtime", lambda *a, **k: started.append(1)
    )

    cfg = initialization.init()  # mode defaults to 'auto'; disabled env wins

    assert cfg.disabled is True
    assert started == []


def test_disabled_env_dynamically_silences_low_level_utilities(monkeypatch):
    import torch.nn as nn

    from traceml_ai.utils.step_memory import (
        StepMemoryTracker,
        flush_step_memory_buffer,
        step_memory_queue,
    )
    from traceml_ai.utils.timing import (
        TimeEvent,
        TimeScope,
        flush_step_time_buffer,
        get_step_time_queue,
        record_event,
        timed_region,
    )

    step_time_queue = get_step_time_queue()
    while not step_time_queue.empty():
        step_time_queue.get_nowait()
    while not step_memory_queue.empty():
        step_memory_queue.get_nowait()

    monkeypatch.setenv("TRACEML_DISABLED", "1")

    record_event(
        TimeEvent(
            name="disabled_step",
            device="cpu",
            cpu_start=0.0,
            cpu_end=1.0,
            scope=TimeScope.STEP,
        )
    )
    flush_step_time_buffer(1)

    model = nn.Linear(1, 1)
    tracker = StepMemoryTracker(model)
    tracker.reset()
    tracker.record()
    flush_step_memory_buffer(model, 1)

    with timed_region("disabled_region"):
        pass
    flush_step_time_buffer(2)

    assert step_time_queue.empty()
    assert step_memory_queue.empty()


def test_init_starts_runtime_when_aggregator_reachable(
    initialization, monkeypatch
):
    import traceml_ai.runtime.lifecycle as lifecycle

    monkeypatch.delenv("TRACEML_DISABLED", raising=False)
    monkeypatch.delenv("TRACEML_AGGREGATOR_HOST", raising=False)
    monkeypatch.delenv("TRACEML_AGGREGATOR_PORT", raising=False)

    waited = {}

    def fake_wait(host, port, **kwargs):
        waited["host"] = host
        waited["port"] = port
        return True

    captured = {}
    fake_handle = _FakeHandle()

    def fake_start(settings, **kwargs):
        captured["settings"] = settings
        return fake_handle

    monkeypatch.setattr(lifecycle, "get_active_runtime_handle", lambda: None)
    monkeypatch.setattr(lifecycle, "wait_for_aggregator", fake_wait)
    monkeypatch.setattr(lifecycle, "start_runtime", fake_start)

    cfg = initialization.init(
        mode="manual",
        aggregator_host="10.0.0.5",
        aggregator_port=40000,
    )

    assert cfg.disabled is False
    assert waited == {"host": "10.0.0.5", "port": 40000}
    assert captured["settings"].aggregator.connect_host == "10.0.0.5"
    assert captured["settings"].aggregator.port == 40000
    assert initialization._RUNTIME_HANDLE is fake_handle


def test_patch_install_failure_warns_noops_and_stops_runtime(
    initialization, monkeypatch, capsys
):
    import traceml_ai.instrumentation.patches.forward_auto_timer_patch as forward_patch
    import traceml_ai.runtime.lifecycle as lifecycle

    monkeypatch.delenv("TRACEML_DISABLED", raising=False)
    monkeypatch.delenv("TRACEML_ON_MISSING_AGGREGATOR", raising=False)
    fake_handle = _FakeHandle()

    monkeypatch.setattr(lifecycle, "get_active_runtime_handle", lambda: None)
    monkeypatch.setattr(lifecycle, "wait_for_aggregator", lambda *a, **k: True)
    monkeypatch.setattr(
        lifecycle, "start_runtime", lambda *a, **k: fake_handle
    )
    monkeypatch.setattr(
        forward_patch,
        "patch_forward",
        lambda: (_ for _ in ()).throw(RuntimeError("patch boom")),
    )

    cfg = initialization.init(
        mode="selective",
        patch_forward=True,
        aggregator_host="127.0.0.1",
        aggregator_port=40000,
    )

    assert cfg.disabled is True
    assert cfg.patch_forward is False
    assert fake_handle.stops == 1
    assert initialization.get_init_config() is cfg
    assert initialization.is_tracing_armed() is False
    assert os.environ["TRACEML_DISABLED"] == "1"
    captured = capsys.readouterr()
    assert "[TraceML]" in captured.err
    assert "patch boom" in captured.err


def test_patch_install_failure_strict_raises_and_cleans_up(
    initialization, monkeypatch
):
    import traceml_ai.instrumentation.patches.forward_auto_timer_patch as forward_patch
    import traceml_ai.runtime.lifecycle as lifecycle

    monkeypatch.delenv("TRACEML_DISABLED", raising=False)
    fake_handle = _FakeHandle()

    monkeypatch.setattr(lifecycle, "get_active_runtime_handle", lambda: None)
    monkeypatch.setattr(lifecycle, "wait_for_aggregator", lambda *a, **k: True)
    monkeypatch.setattr(
        lifecycle, "start_runtime", lambda *a, **k: fake_handle
    )
    monkeypatch.setattr(
        forward_patch,
        "patch_forward",
        lambda: (_ for _ in ()).throw(RuntimeError("patch boom")),
    )

    with pytest.raises(RuntimeError, match="patch boom"):
        initialization.init(
            mode="selective",
            patch_forward=True,
            aggregator_host="127.0.0.1",
            aggregator_port=40000,
            on_missing_aggregator="raise",
        )

    assert fake_handle.stops == 1
    assert initialization.get_init_config() is None
    assert initialization.is_tracing_armed() is False


def test_init_warns_and_noops_when_aggregator_unreachable(
    initialization, monkeypatch, capsys
):
    import traceml_ai.runtime.lifecycle as lifecycle

    monkeypatch.delenv("TRACEML_DISABLED", raising=False)
    monkeypatch.delenv("TRACEML_ON_MISSING_AGGREGATOR", raising=False)
    started = []
    monkeypatch.setattr(lifecycle, "get_active_runtime_handle", lambda: None)
    monkeypatch.setattr(
        lifecycle, "wait_for_aggregator", lambda *a, **k: False
    )
    monkeypatch.setattr(
        lifecycle, "start_runtime", lambda *a, **k: started.append(1)
    )

    # Default is fail-open: a library call inside user code must never crash
    # the training process for a transport reason. Detach to a no-op + warn.
    cfg = initialization.init(mode="manual", connect_timeout_sec=1.0)

    assert cfg.disabled is True
    assert started == []  # never started after a failed preflight
    assert "[TraceML]" in capsys.readouterr().err  # detached loudly
    assert initialization.get_init_config() is cfg  # no-op config stored


def test_init_raises_when_aggregator_unreachable_and_strict(
    initialization, monkeypatch
):
    import traceml_ai.runtime.lifecycle as lifecycle

    monkeypatch.delenv("TRACEML_DISABLED", raising=False)
    monkeypatch.delenv("TRACEML_ON_MISSING_AGGREGATOR", raising=False)
    started = []
    monkeypatch.setattr(lifecycle, "get_active_runtime_handle", lambda: None)
    monkeypatch.setattr(
        lifecycle, "wait_for_aggregator", lambda *a, **k: False
    )
    monkeypatch.setattr(
        lifecycle, "start_runtime", lambda *a, **k: started.append(1)
    )

    with pytest.raises(RuntimeError, match="could not reach the aggregator"):
        initialization.init(
            mode="manual",
            connect_timeout_sec=1.0,
            on_missing_aggregator="raise",
        )

    assert started == []  # never started after a failed preflight
    assert initialization.get_init_config() is None  # config not stored


def test_init_skips_runtime_when_already_active(initialization, monkeypatch):
    import traceml_ai.runtime.lifecycle as lifecycle

    monkeypatch.delenv("TRACEML_DISABLED", raising=False)
    monkeypatch.setattr(
        lifecycle, "get_active_runtime_handle", lambda: object()
    )

    waited = []
    started = []
    monkeypatch.setattr(
        lifecycle,
        "wait_for_aggregator",
        lambda *a, **k: waited.append(1) or True,
    )
    monkeypatch.setattr(
        lifecycle, "start_runtime", lambda *a, **k: started.append(1)
    )

    cfg = initialization.init(mode="manual")

    assert cfg.disabled is False
    assert waited == []  # no preflight when a runtime already runs here
    assert started == []  # no second runtime started
    assert initialization._RUNTIME_HANDLE is None


def test_stop_runtime_for_init_is_idempotent(initialization):
    handle = _FakeHandle()
    initialization._RUNTIME_HANDLE = handle

    initialization._stop_runtime_for_init()
    initialization._stop_runtime_for_init()

    assert handle.stops == 1
    assert initialization._RUNTIME_HANDLE is None
