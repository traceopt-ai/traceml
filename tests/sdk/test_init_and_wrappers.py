import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _reload_initialization_module():
    import traceml_ai.sdk.initial as initialization

    importlib.reload(initialization)
    # These tests exercise instrumentation patch policy, not runtime startup.
    # Stub the runtime bootstrap so init() does not try to reach an aggregator.
    initialization._start_runtime_for_init = lambda **kwargs: None
    return initialization


def _reload_instrumentation_module():
    import traceml_ai.sdk.instrumentation as instrumentation

    return importlib.reload(instrumentation)


def _reload_wrappers_module():
    import traceml_ai.sdk.wrappers as wrappers

    return importlib.reload(wrappers)


def test_init_auto_enables_all_supported_patches(monkeypatch):
    initialization = _reload_initialization_module()

    calls = []

    import traceml_ai.instrumentation.patches.backward_auto_timer_patch as backward_patch
    import traceml_ai.instrumentation.patches.dataloader_patch as dataloader_patch
    import traceml_ai.instrumentation.patches.forward_auto_timer_patch as forward_patch

    monkeypatch.setattr(
        dataloader_patch,
        "patch_dataloader",
        lambda: calls.append("dataloader"),
    )
    monkeypatch.setattr(
        forward_patch,
        "patch_forward",
        lambda: calls.append("forward"),
    )
    monkeypatch.setattr(
        backward_patch,
        "patch_backward",
        lambda: calls.append("backward"),
    )

    cfg = initialization.init(mode="auto")

    assert cfg.mode == "auto"
    assert cfg.patch_dataloader is True
    assert cfg.patch_forward is True
    assert cfg.patch_backward is True
    assert calls == ["dataloader", "forward", "backward"]


def test_init_manual_installs_no_patches():
    initialization = _reload_initialization_module()

    cfg = initialization.init(mode="manual")

    assert cfg.mode == "manual"
    assert cfg.patch_dataloader is False
    assert cfg.patch_forward is False
    assert cfg.patch_backward is False


def test_init_selective_only_installs_requested_patches(monkeypatch):
    initialization = _reload_initialization_module()

    calls = []

    import traceml_ai.instrumentation.patches.backward_auto_timer_patch as backward_patch
    import traceml_ai.instrumentation.patches.dataloader_patch as dataloader_patch
    import traceml_ai.instrumentation.patches.forward_auto_timer_patch as forward_patch

    monkeypatch.setattr(
        dataloader_patch,
        "patch_dataloader",
        lambda: calls.append("dataloader"),
    )
    monkeypatch.setattr(
        forward_patch,
        "patch_forward",
        lambda: calls.append("forward"),
    )
    monkeypatch.setattr(
        backward_patch,
        "patch_backward",
        lambda: calls.append("backward"),
    )

    cfg = initialization.init(
        mode="selective",
        patch_dataloader=True,
        patch_forward=False,
        patch_backward=True,
    )

    assert cfg.mode == "selective"
    assert cfg.patch_dataloader is True
    assert cfg.patch_forward is False
    assert cfg.patch_backward is True
    assert calls == ["dataloader", "backward"]


def test_init_auto_and_manual_reject_patch_overrides():
    initialization = _reload_initialization_module()

    with pytest.raises(ValueError, match="may only be provided"):
        initialization.init(mode="auto", patch_forward=True)

    initialization = _reload_initialization_module()

    with pytest.raises(ValueError, match="may only be provided"):
        initialization.init(mode="manual", patch_backward=True)


def test_init_selective_requires_meaningful_overrides():
    initialization = _reload_initialization_module()

    with pytest.raises(ValueError, match="requires at least one explicit"):
        initialization.init(mode="selective")

    initialization = _reload_initialization_module()

    with pytest.raises(ValueError, match="must enable at least one"):
        initialization.init(
            mode="selective",
            patch_dataloader=False,
            patch_forward=False,
            patch_backward=False,
        )


def test_init_custom_alias_maps_to_selective(monkeypatch):
    initialization = _reload_initialization_module()

    calls = []

    import traceml_ai.instrumentation.patches.forward_auto_timer_patch as forward_patch

    monkeypatch.setattr(
        forward_patch,
        "patch_forward",
        lambda: calls.append("forward"),
    )

    cfg = initialization.init(mode="custom", patch_forward=True)

    assert cfg.mode == "selective"
    assert cfg.patch_dataloader is False
    assert cfg.patch_forward is True
    assert cfg.patch_backward is False
    assert calls == ["forward"]


def test_init_is_idempotent_for_same_config_and_rejects_conflicts():
    initialization = _reload_initialization_module()

    cfg_a = initialization.init(mode="manual")
    cfg_b = initialization.init(mode="manual")

    assert cfg_a == cfg_b

    with pytest.raises(RuntimeError, match="already been initialized"):
        initialization.init(mode="auto")


def test_start_is_alias_for_init():
    initialization = _reload_initialization_module()

    cfg = initialization.start(mode="manual")

    assert cfg.mode == "manual"
    assert cfg.patch_dataloader is False
    assert cfg.patch_forward is False
    assert cfg.patch_backward is False


def test_api_import_does_not_initialize_implicitly():
    initialization = _reload_initialization_module()

    import traceml_ai.api as api

    importlib.reload(api)

    assert initialization.get_init_config() is None


@contextmanager
def _noop_context_manager(*args, **kwargs):
    yield


class _NoopStepMemoryTracker:
    def __init__(self, model):
        self.model = model

    def reset(self):
        return None

    def record(self):
        return None


def _run_trace_step_once(*, mode, monkeypatch):
    initialization = _reload_initialization_module()
    instrumentation = _reload_instrumentation_module()

    initialization.init(
        mode=mode,
        patch_forward=(False if mode == "selective" else None),
        patch_backward=(False if mode == "selective" else None),
        patch_dataloader=(True if mode == "selective" else None),
    )

    calls = []

    monkeypatch.setattr(
        instrumentation,
        "timed_region",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "forward_auto_timer",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "backward_auto_timer",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "StepMemoryTracker",
        _NoopStepMemoryTracker,
    )
    monkeypatch.setattr(
        instrumentation,
        "flush_step_events",
        lambda model, step: None,
    )
    monkeypatch.setattr(
        instrumentation,
        "ensure_optimizer_timing_installed",
        lambda: calls.append("optimizer"),
    )
    monkeypatch.setattr(
        instrumentation,
        "_publish_runtime_environment",
        lambda model: None,
    )

    model = nn.Linear(2, 2)
    with instrumentation.trace_step(model):
        pass

    return calls


def test_trace_step_only_auto_installs_optimizer_timing(monkeypatch):
    assert _run_trace_step_once(mode="auto", monkeypatch=monkeypatch) == [
        "optimizer"
    ]
    assert _run_trace_step_once(mode="manual", monkeypatch=monkeypatch) == []
    assert (
        _run_trace_step_once(mode="selective", monkeypatch=monkeypatch) == []
    )


def test_trace_step_publishes_runtime_environment_once(monkeypatch):
    _reload_initialization_module()
    instrumentation = _reload_instrumentation_module()
    published = []

    monkeypatch.setattr(
        instrumentation,
        "timed_region",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "forward_auto_timer",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "backward_auto_timer",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "h2d_auto_timer",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "StepMemoryTracker",
        _NoopStepMemoryTracker,
    )
    monkeypatch.setattr(
        instrumentation,
        "flush_step_events",
        lambda model, step: None,
    )
    monkeypatch.setattr(
        instrumentation,
        "ensure_optimizer_timing_installed",
        lambda: None,
    )
    monkeypatch.setattr(
        instrumentation,
        "has_runtime_environment_info",
        lambda: bool(published),
    )
    monkeypatch.setattr(
        instrumentation,
        "detect_runtime_environment",
        lambda model: ("runtime-info", id(model)),
    )
    monkeypatch.setattr(
        instrumentation,
        "publish_runtime_environment_once",
        lambda info: published.append(info) or True,
    )

    model = nn.Linear(2, 2)
    with instrumentation.trace_step(model):
        pass
    with instrumentation.trace_step(model):
        pass

    assert published == [("runtime-info", id(model))]


def test_trace_step_without_init_does_not_auto_install_optimizer_timing(
    monkeypatch,
):
    _reload_initialization_module()
    instrumentation = _reload_instrumentation_module()
    calls = []

    monkeypatch.setattr(
        instrumentation,
        "timed_region",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "forward_auto_timer",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "backward_auto_timer",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "h2d_auto_timer",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "StepMemoryTracker",
        _NoopStepMemoryTracker,
    )
    monkeypatch.setattr(
        instrumentation,
        "flush_step_events",
        lambda model, step: None,
    )
    monkeypatch.setattr(
        instrumentation,
        "ensure_optimizer_timing_installed",
        lambda: calls.append("optimizer"),
    )

    with instrumentation.trace_step(nn.Linear(2, 2)):
        pass

    assert calls == []


def test_trace_step_records_gpu_events_for_step_envelope(monkeypatch):
    _reload_initialization_module()
    instrumentation = _reload_instrumentation_module()
    calls = []

    @contextmanager
    def fake_timed_region(name, scope, record_gpu_events):
        calls.append((name, scope, record_gpu_events))
        yield

    monkeypatch.setattr(
        instrumentation,
        "timed_region",
        fake_timed_region,
    )
    monkeypatch.setattr(
        instrumentation,
        "forward_auto_timer",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "backward_auto_timer",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "h2d_auto_timer",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "StepMemoryTracker",
        _NoopStepMemoryTracker,
    )
    monkeypatch.setattr(
        instrumentation,
        "flush_step_events",
        lambda model, step: None,
    )
    monkeypatch.setattr(
        instrumentation,
        "ensure_optimizer_timing_installed",
        lambda: None,
    )

    with instrumentation.trace_step(nn.Linear(2, 2)):
        pass

    assert calls == [("_traceml_internal:step_time", "step", True)]


def test_trace_step_marks_recording_draining_after_configured_step(
    monkeypatch,
):
    from traceml_ai.runtime.state import (
        TraceMLRecordingStatus,
        configure_trace_recording,
        get_trace_recording_state,
        reset_trace_session_state,
    )

    instrumentation = _reload_instrumentation_module()
    reset_trace_session_state()
    configure_trace_recording(max_steps=1)

    monkeypatch.setattr(
        instrumentation,
        "timed_region",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "forward_auto_timer",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "backward_auto_timer",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "h2d_auto_timer",
        _noop_context_manager,
    )
    monkeypatch.setattr(
        instrumentation,
        "StepMemoryTracker",
        _NoopStepMemoryTracker,
    )
    monkeypatch.setattr(
        instrumentation,
        "flush_step_events",
        lambda model, step: None,
    )
    monkeypatch.setattr(
        instrumentation,
        "ensure_optimizer_timing_installed",
        lambda: None,
    )

    with instrumentation.trace_step(nn.Linear(2, 2)):
        pass

    assert (
        get_trace_recording_state().status == TraceMLRecordingStatus.DRAINING
    )

    configure_trace_recording()
    reset_trace_session_state()


def test_wrap_dataloader_fetch_allows_custom_iterator_when_torch_patch_active(
    monkeypatch,
):
    wrappers = _reload_wrappers_module()

    from torch.utils.data import DataLoader

    monkeypatch.setattr(DataLoader, "_traceml_patched", True, raising=False)

    wrapped = wrappers.wrap_dataloader_fetch(iter([1, 2, 3]))

    assert list(wrapped) == [1, 2, 3]


def test_wrap_dataloader_fetch_rejects_torch_dataloader_when_auto_patch_active(
    monkeypatch,
):
    wrappers = _reload_wrappers_module()

    from torch.utils.data import DataLoader

    monkeypatch.setattr(DataLoader, "_traceml_patched", True, raising=False)
    loader = DataLoader([1, 2, 3], batch_size=1)

    with pytest.raises(RuntimeError, match="already active"):
        wrappers.wrap_dataloader_fetch(loader)


def test_auto_dataloader_patch_records_gpu_events(monkeypatch):
    import traceml_ai.instrumentation.patches.dataloader_patch as dataloader_patch

    calls = []

    @contextmanager
    def fake_timed_region(name, scope, record_gpu_events):
        calls.append((name, scope, record_gpu_events))
        yield

    monkeypatch.setattr(
        dataloader_patch,
        "timed_region",
        fake_timed_region,
    )
    monkeypatch.setattr(
        dataloader_patch,
        "is_tracing_armed",
        lambda: True,
    )

    from torch.utils.data import DataLoader

    loader = DataLoader([1, 2, 3], batch_size=1)

    assert len(list(dataloader_patch._traceml_dataloader_iter(loader))) == 3
    assert calls == [
        ("_traceml_internal:dataloader_next", "step", True),
        ("_traceml_internal:dataloader_next", "step", True),
        ("_traceml_internal:dataloader_next", "step", True),
        ("_traceml_internal:dataloader_next", "step", True),
    ]


def test_auto_dataloader_patch_passes_through_when_unarmed(monkeypatch):
    import traceml_ai.instrumentation.patches.dataloader_patch as dataloader_patch

    calls = []

    @contextmanager
    def fake_timed_region(name, scope, record_gpu_events):
        calls.append((name, scope, record_gpu_events))
        yield

    monkeypatch.setattr(
        dataloader_patch,
        "timed_region",
        fake_timed_region,
    )
    monkeypatch.setattr(
        dataloader_patch,
        "is_tracing_armed",
        lambda: False,
    )

    from torch.utils.data import DataLoader

    loader = DataLoader([1, 2, 3], batch_size=1)

    assert len(list(dataloader_patch._traceml_dataloader_iter(loader))) == 3
    assert calls == []


def test_wrap_dataloader_fetch_records_gpu_events(monkeypatch):
    wrappers = _reload_wrappers_module()

    calls = []

    @contextmanager
    def fake_timed_region(name, scope, record_gpu_events):
        calls.append((name, scope, record_gpu_events))
        yield

    monkeypatch.setattr(wrappers, "timed_region", fake_timed_region)

    from torch.utils.data import DataLoader

    monkeypatch.setattr(
        DataLoader,
        "_traceml_patched",
        False,
        raising=False,
    )

    wrapped = wrappers.wrap_dataloader_fetch(iter([1, 2]))

    assert list(wrapped) == [1, 2]
    assert calls == [
        ("_traceml_internal:dataloader_next", "step", True),
        ("_traceml_internal:dataloader_next", "step", True),
        ("_traceml_internal:dataloader_next", "step", True),
    ]


def test_wrap_forward_times_model_instance(monkeypatch):
    wrappers = _reload_wrappers_module()

    calls = []

    @contextmanager
    def fake_timed_region(name, scope, record_gpu_events):
        calls.append((name, scope, record_gpu_events))
        yield

    monkeypatch.setattr(wrappers, "timed_region", fake_timed_region)
    monkeypatch.setattr(
        nn.Module,
        "_traceml_forward_patched",
        False,
        raising=False,
    )

    model = nn.Linear(4, 2)
    wrapped = wrappers.wrap_forward(model)

    x = torch.randn(3, 4)
    _ = wrapped(x)

    assert wrapped is model
    assert calls == [("_traceml_internal:forward_time", "step", True)]


def test_wrap_backward_times_backward(monkeypatch):
    wrappers = _reload_wrappers_module()

    calls = []

    @contextmanager
    def fake_timed_region(name, scope, record_gpu_events):
        calls.append((name, scope, record_gpu_events))
        yield

    monkeypatch.setattr(wrappers, "timed_region", fake_timed_region)
    monkeypatch.setattr(
        torch,
        "_traceml_backward_patched",
        False,
        raising=False,
    )

    class DummyLoss:
        def __init__(self):
            self.called = False

        def backward(self):
            self.called = True

    loss = DummyLoss()
    wrapped = wrappers.wrap_backward(loss)
    wrapped.backward()

    assert loss.called is True
    assert calls == [("_traceml_internal:backward_time", "step", True)]


def test_wrap_optimizer_preserves_identity_and_times_step(monkeypatch):
    wrappers = _reload_wrappers_module()

    calls = []

    @contextmanager
    def fake_timed_region(name, scope, record_gpu_events):
        calls.append((name, scope, record_gpu_events))
        yield

    monkeypatch.setattr(wrappers, "timed_region", fake_timed_region)
    monkeypatch.setattr(
        torch.optim.Optimizer,
        "_traceml_opt_hooks_installed",
        False,
        raising=False,
    )

    class DummyOptimizer:
        def __init__(self):
            self.called = False

        def step(self):
            self.called = True
            return "ok"

    optimizer = DummyOptimizer()
    wrapped = wrappers.wrap_optimizer(optimizer)
    result = optimizer.step()

    assert wrapped is optimizer
    assert result == "ok"
    assert optimizer.called is True
    assert calls == [("_traceml_internal:optimizer_step", "step", True)]
