"""
Tests for H2D (host-to-device) transfer timing instrumentation.

All tests run without a GPU.  The auto-patch and wrap_h2d() paths both fall
back to CPU timing when CUDA is unavailable, so the event is still emitted and
stored in the step buffer.

Coverage
--------
- Auto-patch: torch.Tensor.to() is patched and disabled outside trace_step.
- Auto-patch: event is recorded inside trace_step only.
- Auto-patch: non-CUDA targets are NOT timed.
- wrap_h2d(): wraps any object with a .to() method and records the event.
- wrap_h2d(): raises when automatic H2D patch is already active.
- wrap_h2d(): raises for objects without .to().
- init(mode="auto"): patch_h2d is True.
- init(mode="manual"): patch_h2d is False.
- init(mode="selective", patch_h2d=True): patch_h2d is True.
- init(mode="selective", patch_h2d=False but another enabled): h2d is False.
"""

from __future__ import annotations

import importlib
import sys
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import traceml.instrumentation.patches.forward_auto_timer_patch as fwd_patch
import traceml.instrumentation.patches.h2d_auto_timer_patch as h2d_patch
import traceml.utils.timing as timing_module
from traceml.instrumentation.patches.h2d_auto_timer_patch import (
    _H2D_TLS,
    _ORIG_TENSOR_TO,
    _is_cuda_target,
    _traceml_tensor_to,
    h2d_auto_timer,
)
from traceml.sdk.wrappers import wrap_h2d


# _reload_initialization and _reload_h2d_patch use local imports intentionally.
# importlib.reload() requires a live reference to the already-imported module
# object.  Moving these imports to module level would capture the original
# binding and make reload() operate on a stale reference.


def _reload_initialization():
    import traceml.sdk.initial as m

    return importlib.reload(m)


def _reload_h2d_patch():
    import traceml.instrumentation.patches.h2d_auto_timer_patch as m

    return importlib.reload(m)


@contextmanager
def _fresh_step_buffer():
    """
    Temporarily replace _STEP_BUFFER so tests can inspect recorded events
    without touching the shared global queue.
    """
    original = timing_module._STEP_BUFFER
    timing_module._STEP_BUFFER = deque()
    try:
        yield timing_module._STEP_BUFFER
    finally:
        timing_module._STEP_BUFFER = original


def _recorded_h2d_events(buf: deque) -> list:
    return [e for e in buf if e.name == "_traceml_internal:h2d_time"]


# Auto-patch: _is_cuda_target detection


class TestIsCudaTarget:
    """Unit-test _is_cuda_target without needing a real GPU."""

    def _fn(self, args, kwargs):
        return _is_cuda_target(args, kwargs)

    def test_string_cuda_device(self):
        assert self._fn(("cuda:0",), {}) is True

    def test_string_plain_cuda(self):
        assert self._fn(("cuda",), {}) is True

    def test_torch_device_cpu_is_false(self):
        assert self._fn((torch.device("cpu"),), {}) is False

    def test_torch_device_cuda_is_true(self):
        assert self._fn((torch.device("cuda", 0),), {}) is True

    def test_keyword_device_string(self):
        assert self._fn((), {"device": "cuda:1"}) is True

    def test_keyword_device_torch_device(self):
        assert self._fn((), {"device": torch.device("cuda", 0)}) is True

    def test_cpu_string_is_false(self):
        assert self._fn(("cpu",), {}) is False

    def test_dtype_only_is_false(self):
        # tensor.to(torch.float16) — first arg is a dtype, not a device
        assert self._fn((torch.float16,), {}) is False

    def test_no_args_is_false(self):
        assert self._fn((), {}) is False

    def test_string_containing_cuda_but_not_device_is_false(self):
        # I2 regression: "/path/to/cudacheckpoint" contains "cuda" as substring
        # but is not a valid CUDA device — must not be classified as a CUDA target.
        assert self._fn(("/path/to/cudacheckpoint",), {}) is False

    def test_invalid_device_string_is_false(self):
        # I2 regression: arbitrary strings that aren't valid device specs must
        # return False, not raise.
        assert self._fn(("not_a_device",), {}) is False


# Auto-patch: TLS enable-flag gating


class TestH2DAutoTimerPatch:
    def setup_method(self):
        # Reload so each test starts with a clean TLS state.
        self._mod = _reload_h2d_patch()
        # Reset the TLS flag directly.
        self._mod._H2D_TLS._traceml_h2d_enabled = False

    def test_disabled_outside_context_manager(self):
        assert self._mod._enabled() is False

    def test_enabled_inside_context_manager(self):
        with self._mod.h2d_auto_timer():
            assert self._mod._enabled() is True
        assert self._mod._enabled() is False

    def test_patched_to_skips_timing_when_disabled(self):
        """
        When the TLS flag is False, the patched .to() must call the original
        without going through timed_region.
        """
        self._mod._H2D_TLS._traceml_h2d_enabled = False

        tensor = torch.ones(4)
        recorded = []

        def fake_timed_region(name, scope, use_gpu):
            # Should never be reached when disabled
            recorded.append(name)

            @contextmanager
            def _ctx():
                yield

            return _ctx()

        with patch(
            "traceml.instrumentation.patches.h2d_auto_timer_patch.timed_region",
            side_effect=fake_timed_region,
        ):
            self._mod._traceml_tensor_to(tensor, "cpu")

        assert recorded == [], "timed_region must NOT be called when disabled"

    def test_patched_to_skips_timing_for_cpu_target_even_when_enabled(self):
        self._mod._H2D_TLS._traceml_h2d_enabled = True

        tensor = torch.ones(4)
        recorded = []

        def fake_timed_region(name, scope, use_gpu):
            recorded.append(name)

            @contextmanager
            def _ctx():
                yield

            return _ctx()

        with patch(
            "traceml.instrumentation.patches.h2d_auto_timer_patch.timed_region",
            side_effect=fake_timed_region,
        ):
            # CPU target → _is_cuda_target returns False → no timing
            self._mod._traceml_tensor_to(tensor, "cpu")

        assert recorded == [], "CPU target must NOT be timed"

    def test_patched_to_records_event_when_enabled_and_cuda_target(self):
        """
        Simulate an H2D transfer: enabled=True + cuda target → timed_region
        must be entered.  We mock the original .to() so no real GPU is needed.
        """
        self._mod._H2D_TLS._traceml_h2d_enabled = True

        tensor = torch.ones(4)
        recorded = []

        @contextmanager
        def fake_timed_region(name, scope, use_gpu):
            recorded.append(name)
            yield

        # Patch the original .to() so it doesn't try to move to a real GPU.
        with patch.object(torch.Tensor, "to", return_value=tensor):
            with patch(
                "traceml.instrumentation.patches.h2d_auto_timer_patch.timed_region",
                side_effect=fake_timed_region,
            ):
                self._mod._traceml_tensor_to(tensor, "cuda:0")

        assert recorded == ["_traceml_internal:h2d_time"]

    def test_c1_d2d_skips_timing_when_source_is_cuda(self):
        """
        C1: cuda_tensor.to('cuda:1') is a D2D transfer and must NOT be
        recorded as h2d_time regardless of the destination.
        """
        self._mod._H2D_TLS._traceml_h2d_enabled = True

        # Simulate a tensor whose .is_cuda is True without a real GPU.
        cuda_tensor = torch.ones(4)
        recorded = []

        @contextmanager
        def fake_timed_region(name, scope, use_gpu):
            recorded.append(name)
            yield

        with patch.object(
            type(cuda_tensor),
            "is_cuda",
            new_callable=lambda: property(lambda self: True),
        ):
            with patch(
                "traceml.instrumentation.patches.h2d_auto_timer_patch.timed_region",
                side_effect=fake_timed_region,
            ):
                # Even though destination is CUDA, source is already CUDA → D2D.
                self._mod._traceml_tensor_to(cuda_tensor, "cuda:0")

        assert recorded == [], "D2D transfers must NOT be timed as h2d_time"

    def test_c2_parameter_skips_timing_to_prevent_model_to_device_overcount(
        self,
    ):
        """
        C2: nn.Parameter.to(device) must NOT be recorded as h2d_time.
        model.to(device) calls .to() once per parameter, which would inflate
        n_calls by the parameter count rather than the batch transfer count.
        """
        self._mod._H2D_TLS._traceml_h2d_enabled = True

        param = torch.nn.Parameter(torch.ones(4))
        recorded = []

        @contextmanager
        def fake_timed_region(name, scope, use_gpu):
            recorded.append(name)
            yield

        with patch(
            "traceml.instrumentation.patches.h2d_auto_timer_patch.timed_region",
            side_effect=fake_timed_region,
        ):
            self._mod._traceml_tensor_to(param, "cuda:0")

        assert recorded == [], "Parameter.to() must NOT be timed (C2 gate)"

    def test_patch_h2d_is_idempotent(self):
        """patch_h2d() must be safe to call multiple times."""
        mod = self._mod
        # Reset patched flag so we can test the install path.
        torch.Tensor._traceml_h2d_patched = False  # type: ignore[attr-defined]
        try:
            mod.patch_h2d()
            mod.patch_h2d()  # second call must be a no-op
        finally:
            # Restore original to avoid contaminating other tests.
            torch.Tensor.to = mod._ORIG_TENSOR_TO  # type: ignore[assignment]
            torch.Tensor._traceml_h2d_patched = False  # type: ignore[attr-defined]


# Step-buffer integration: event recorded inside trace_step only


class TestH2DStepScoping:
    """
    Verify that .to() timing appears in the step buffer only when inside
    trace_step, using the h2d_auto_timer context manager directly (no full
    init/patch machinery required).
    """

    def test_event_buffered_inside_h2d_auto_timer(self):
        tensor = torch.ones(4)

        with _fresh_step_buffer() as buf:
            with h2d_auto_timer():
                # Patch .to() at the original pointer level so it returns
                # the same tensor without needing a GPU.
                with patch.object(torch.Tensor, "to", return_value=tensor):
                    _traceml_tensor_to(tensor, "cuda:0")

            events = _recorded_h2d_events(buf)

        assert len(events) == 1
        assert events[0].name == "_traceml_internal:h2d_time"

    def test_no_event_buffered_outside_h2d_auto_timer(self):
        _H2D_TLS._traceml_h2d_enabled = False
        tensor = torch.ones(4)

        with _fresh_step_buffer() as buf:
            with patch.object(torch.Tensor, "to", return_value=tensor):
                _traceml_tensor_to(tensor, "cuda:0")

            events = _recorded_h2d_events(buf)

        assert events == []


# wrap_h2d()


class TestWrapH2D:
    def setup_method(self):
        # Ensure no auto H2D patch is installed for manual-wrapper tests.
        torch.Tensor._traceml_h2d_patched = False  # type: ignore[attr-defined]

    def teardown_method(self):
        torch.Tensor._traceml_h2d_patched = False  # type: ignore[attr-defined]

    def test_wrap_h2d_returns_proxy(self):
        tensor = torch.ones(4)
        wrapped = wrap_h2d(tensor)
        assert hasattr(wrapped, "to")

    def test_wrap_h2d_to_records_event(self):
        tensor = torch.ones(4)
        wrapped = wrap_h2d(tensor)

        with _fresh_step_buffer() as buf:
            with patch.object(torch.Tensor, "to", return_value=tensor):
                wrapped.to("cpu")  # device doesn't matter for wrap_h2d timing

            events = _recorded_h2d_events(buf)

        assert len(events) == 1
        assert events[0].name == "_traceml_internal:h2d_time"

    def test_wrap_h2d_forwards_attributes(self):
        tensor = torch.ones(4, 4)
        wrapped = wrap_h2d(tensor)
        assert wrapped.shape == tensor.shape
        assert wrapped.dtype == tensor.dtype

    def test_wrap_h2d_raises_for_object_without_to(self):
        with pytest.raises(TypeError, match="callable .to\\(\\) method"):
            wrap_h2d(object())

    def test_wrap_h2d_defers_when_auto_patch_active(self):
        """C3: wrap-then-init race — wrapper must defer to the auto-patch,
        not add a second timed_region, producing exactly 0 extra events."""
        tensor = torch.ones(4)
        wrapped = wrap_h2d(tensor)  # sentinel False at construction time

        recorded = []

        @contextmanager
        def fake_timed_region(name, scope, use_gpu):
            recorded.append(name)
            yield

        # Simulate init(mode="auto") running after wrap_h2d().
        torch.Tensor._traceml_h2d_patched = True  # type: ignore[attr-defined]
        try:
            with patch(
                "traceml.sdk.wrappers.timed_region",
                side_effect=fake_timed_region,
            ):
                with patch.object(torch.Tensor, "to", return_value=tensor):
                    wrapped.to("cuda:0")
        finally:
            torch.Tensor._traceml_h2d_patched = False  # type: ignore[attr-defined]

        assert (
            recorded == []
        ), "_WrappedH2D.to() must not enter timed_region when auto-patch is active (C3)"

    def test_wrap_h2d_accepts_custom_batch_object(self):
        """wrap_h2d() works on any object with a .to() method, not only tensors."""

        class FakeBatch:
            def to(self, device):
                return self

        batch = FakeBatch()
        wrapped = wrap_h2d(batch)

        with _fresh_step_buffer() as buf:
            wrapped.to("cpu")
            events = _recorded_h2d_events(buf)

        assert len(events) == 1

    def test_wrap_h2d_dict_batch_dunder_forwarding(self):
        """I1: Python bypasses __getattr__ for special methods, so _WrappedH2D
        must define __getitem__, __len__, __iter__, __contains__ explicitly.
        A dict-like batch must be fully usable through the wrapper."""

        class DictBatch:
            def __init__(self, data):
                self._data = data

            def to(self, device):
                return self

            def __getitem__(self, key):
                return self._data[key]

            def __len__(self):
                return len(self._data)

            def __iter__(self):
                return iter(self._data)

            def __contains__(self, item):
                return item in self._data

        batch = DictBatch({"x": 1, "y": 2})
        wrapped = wrap_h2d(batch)

        # All four special methods must work without TypeError.
        assert wrapped["x"] == 1
        assert len(wrapped) == 2
        assert "x" in wrapped
        assert set(wrapped) == {"x", "y"}


# init() config correctness


class TestInitConfigH2D:
    def test_auto_mode_enables_h2d(self, monkeypatch):
        initialization = _reload_initialization()
        monkeypatch.setattr(
            initialization, "_apply_requested_patches", lambda cfg: None
        )
        cfg = initialization.init(mode="auto")
        assert cfg.patch_h2d is True

    def test_manual_mode_disables_h2d(self, monkeypatch):
        initialization = _reload_initialization()
        monkeypatch.setattr(
            initialization, "_apply_requested_patches", lambda cfg: None
        )
        cfg = initialization.init(mode="manual")
        assert cfg.patch_h2d is False

    def test_selective_mode_with_h2d_true(self, monkeypatch):
        initialization = _reload_initialization()
        monkeypatch.setattr(
            initialization, "_apply_requested_patches", lambda cfg: None
        )

        cfg = initialization.init(mode="selective", patch_h2d=True)
        assert cfg.patch_h2d is True

    def test_selective_mode_with_h2d_false_other_patch_enabled(
        self, monkeypatch
    ):
        initialization = _reload_initialization()
        monkeypatch.setattr(
            initialization, "_apply_requested_patches", lambda cfg: None
        )

        cfg = initialization.init(
            mode="selective", patch_forward=True, patch_h2d=False
        )
        assert cfg.patch_h2d is False
        assert cfg.patch_forward is True

    def test_selective_mode_h2d_only(self, monkeypatch):
        initialization = _reload_initialization()
        monkeypatch.setattr(
            initialization, "_apply_requested_patches", lambda cfg: None
        )

        cfg = initialization.init(mode="selective", patch_h2d=True)
        assert cfg.mode == "selective"
        assert cfg.patch_h2d is True
        assert cfg.patch_forward is False
        assert cfg.patch_backward is False
        assert cfg.patch_dataloader is False

    def test_auto_mode_rejects_patch_h2d_override(self):
        initialization = _reload_initialization()

        with pytest.raises(ValueError, match="may only be provided"):
            initialization.init(mode="auto", patch_h2d=True)

    def test_manual_mode_rejects_patch_h2d_override(self):
        initialization = _reload_initialization()

        with pytest.raises(ValueError, match="may only be provided"):
            initialization.init(mode="manual", patch_h2d=False)
