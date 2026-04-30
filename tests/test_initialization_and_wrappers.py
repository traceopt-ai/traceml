import importlib
import sys
from contextlib import contextmanager

import pytest


def _reload_initialization_module():
    import traceml.sdk.initial as initialization

    return importlib.reload(initialization)


def _reload_wrappers_module():
    import traceml.sdk.wrappers as wrappers

    return importlib.reload(wrappers)


def _reload_instrumentation_module():
    import traceml.sdk.instrumentation as instrumentation

    return importlib.reload(instrumentation)


def test_auto_mode_enables_all_supported_patches(monkeypatch):
    initialization = _reload_initialization_module()

    calls = []

    import traceml.instrumentation.patches.backward_auto_timer_patch as backward_patch
    import traceml.instrumentation.patches.dataloader_patch as dataloader_patch
    import traceml.instrumentation.patches.forward_auto_timer_patch as forward_patch

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


def test_manual_mode_installs_no_patches():
    initialization = _reload_initialization_module()

    cfg = initialization.init(mode="manual")

    assert cfg.mode == "manual"
    assert cfg.patch_dataloader is False
    assert cfg.patch_forward is False
    assert cfg.patch_backward is False


def test_auto_mode_rejects_patch_overrides():
    initialization = _reload_initialization_module()

    with pytest.raises(ValueError, match="may only be provided"):
        initialization.init(mode="auto", patch_forward=True)


def test_manual_mode_rejects_patch_overrides():
    initialization = _reload_initialization_module()

    with pytest.raises(ValueError, match="may only be provided"):
        initialization.init(mode="manual", patch_backward=True)


def test_selective_mode_requires_explicit_overrides():
    initialization = _reload_initialization_module()

    with pytest.raises(ValueError, match="requires at least one explicit"):
        initialization.init(mode="selective")


def test_selective_mode_requires_at_least_one_enabled_patch():
    initialization = _reload_initialization_module()

    with pytest.raises(ValueError, match="must enable at least one"):
        initialization.init(
            mode="selective",
            patch_dataloader=False,
            patch_forward=False,
            patch_backward=False,
        )


def test_custom_alias_maps_to_selective(monkeypatch):
    initialization = _reload_initialization_module()

    calls = []

    import traceml.instrumentation.patches.forward_auto_timer_patch as forward_patch

    monkeypatch.setattr(
        forward_patch,
        "patch_forward",
        lambda: calls.append("forward"),
    )

    cfg = initialization.init(mode="custom", patch_forward=True)

    assert cfg.mode == "selective"
    assert cfg.patch_forward is True
    assert cfg.patch_dataloader is False
    assert cfg.patch_backward is False
    assert calls == ["forward"]


def test_same_effective_init_is_idempotent():
    initialization = _reload_initialization_module()

    cfg_a = initialization.init(mode="manual")
    cfg_b = initialization.init(mode="manual")

    assert cfg_a == cfg_b


def test_conflicting_reinit_raises():
    initialization = _reload_initialization_module()

    initialization.init(mode="manual")

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

    import traceml.api as api

    importlib.reload(api)

    assert initialization.get_init_config() is None


def test_legacy_decorators_import_triggers_auto_init(monkeypatch):
    initialization = _reload_initialization_module()

    calls = []

    monkeypatch.setattr(
        initialization,
        "enable_legacy_decorator_auto_init",
        lambda: calls.append("legacy"),
    )

    # Import-time compatibility behavior lives in the module body, so clear
    # both public and SDK compatibility paths to force a fresh import.
    sys.modules.pop("traceml.decorators", None)
    sys.modules.pop("traceml.sdk.decorators_compat", None)
    import traceml.sdk.decorators_compat  # noqa: F401

    assert calls == ["legacy"]


def test_wrap_optimizer_wraps_real_instance_step(monkeypatch):
    wrappers = _reload_wrappers_module()

    calls = []

    @contextmanager
    def fake_timed_region(name, scope, use_gpu):
        calls.append((name, scope, use_gpu))
        yield

    monkeypatch.setattr(wrappers, "timed_region", fake_timed_region)

    class DummyOptimizer:
        def __init__(self):
            self.called = False

        def step(self):
            self.called = True
            return "ok"

    optimizer = DummyOptimizer()
    wrapped = wrappers.wrap_optimizer(optimizer)

    assert wrapped is optimizer
    result = optimizer.step()

    assert result == "ok"
    assert optimizer.called is True
    assert calls == [("_traceml_internal:optimizer_step", "step", True)]


def test_wrap_dataloader_fetch_allows_custom_iterator_even_when_torch_dataloader_is_patched(
    monkeypatch,
):
    wrappers = _reload_wrappers_module()

    monkeypatch.setattr(
        wrappers.DataLoader,
        "_traceml_patched",
        True,
        raising=False,
    )

    class CustomIterator:
        def __init__(self):
            self._values = iter([1, 2, 3])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._values)

    wrapped = wrappers.wrap_dataloader_fetch(CustomIterator())

    assert list(wrapped) == [1, 2, 3]


def test_wrap_dataloader_fetch_rejects_torch_dataloader_when_auto_patch_active(
    monkeypatch,
):
    wrappers = _reload_wrappers_module()

    monkeypatch.setattr(
        wrappers.DataLoader,
        "_traceml_patched",
        True,
        raising=False,
    )

    class DummyTorchDataLoader(wrappers.DataLoader):
        def __iter__(self):
            return iter([1, 2, 3])

    loader = object.__new__(DummyTorchDataLoader)

    with pytest.raises(RuntimeError, match="already active"):
        wrappers.wrap_dataloader_fetch(loader)


def test_trace_step_installs_optimizer_hooks_only_in_auto(monkeypatch):
    initialization = _reload_initialization_module()
    instrumentation = _reload_instrumentation_module()

    install_calls = []

    monkeypatch.setattr(
        instrumentation,
        "ensure_optimizer_timing_installed",
        lambda: install_calls.append("install"),
    )

    @contextmanager
    def fake_timed_region(*args, **kwargs):
        yield

    class FakeAutoTimer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeMemTracker:
        def __init__(self, model):
            self.model = model

        def reset(self):
            return None

        def record(self):
            return None

    monkeypatch.setattr(instrumentation, "timed_region", fake_timed_region)
    monkeypatch.setattr(instrumentation, "forward_auto_timer", FakeAutoTimer)
    monkeypatch.setattr(instrumentation, "backward_auto_timer", FakeAutoTimer)
    monkeypatch.setattr(instrumentation, "StepMemoryTracker", FakeMemTracker)
    monkeypatch.setattr(
        instrumentation,
        "flush_step_events",
        lambda model, step: None,
    )

    class DummyModel:
        pass

    initialization.init(mode="auto")
    with instrumentation.trace_step(DummyModel()):
        pass

    assert install_calls == ["install"]


def test_trace_step_does_not_install_optimizer_hooks_in_manual(monkeypatch):
    initialization = _reload_initialization_module()
    instrumentation = _reload_instrumentation_module()

    install_calls = []

    monkeypatch.setattr(
        instrumentation,
        "ensure_optimizer_timing_installed",
        lambda: install_calls.append("install"),
    )

    @contextmanager
    def fake_timed_region(*args, **kwargs):
        yield

    class FakeAutoTimer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeMemTracker:
        def __init__(self, model):
            self.model = model

        def reset(self):
            return None

        def record(self):
            return None

    monkeypatch.setattr(instrumentation, "timed_region", fake_timed_region)
    monkeypatch.setattr(instrumentation, "forward_auto_timer", FakeAutoTimer)
    monkeypatch.setattr(instrumentation, "backward_auto_timer", FakeAutoTimer)
    monkeypatch.setattr(instrumentation, "StepMemoryTracker", FakeMemTracker)
    monkeypatch.setattr(
        instrumentation,
        "flush_step_events",
        lambda model, step: None,
    )

    class DummyModel:
        pass

    initialization.init(mode="manual")
    with instrumentation.trace_step(DummyModel()):
        pass

    assert install_calls == []
