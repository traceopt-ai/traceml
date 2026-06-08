"""
Tests for DDP gradient-sync timing via register_comm_hook.

Covers:
- install_ddp_comm_hook: sentinel, idempotency, type checks, fail-open
- Per-bucket event emission and step aggregation
- base_hook composition and CUDA event pool cleanup
- Auto-install via trace_step + DDP unwrap
- 2-rank gloo smoke test (gated)
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from traceml_ai.instrumentation.hooks.ddp_comm_hook import (
    _WIRE_NAME,
    _traceml_ddp_comm_hook_factory,
    install_ddp_comm_hook,
)
from traceml_ai.utils.timing import (
    TimeScope,
    _STEP_BUFFER,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_ddp(*, hook_raises: bool = False) -> MagicMock:
    """
    Return a MagicMock that quacks like a DistributedDataParallel instance.

    ``isinstance(mock, DistributedDataParallel)`` returns True via
    ``spec=DistributedDataParallel``.
    """
    mock = MagicMock(spec=DistributedDataParallel)
    mock._traceml_ddp_comm_hook_installed = False

    if hook_raises:
        mock.register_comm_hook.side_effect = RuntimeError(
            "user hook already registered"
        )

    return mock


def _make_fake_bucket(index: int = 0, is_last: bool = False) -> MagicMock:
    """Return a MagicMock that quacks like a dist.GradBucket."""
    bucket = MagicMock(spec=dist.GradBucket)
    bucket.index.return_value = index
    bucket.is_last.return_value = is_last
    bucket.buffer.return_value = torch.zeros(10)
    return bucket


def _make_fake_future(
    tensor: torch.Tensor | None = None,
) -> MagicMock:
    """
    Return a MagicMock Future whose .then() fires the callback.

    Mirrors real PyTorch .then() semantics: callback receives the
    Future itself and must call .value() to get the result.
    """
    if tensor is None:
        tensor = torch.zeros(10)

    fut = MagicMock()
    fut.value.return_value = [tensor]

    def then_impl(callback):
        result_fut = MagicMock()
        result = callback(fut)
        result_fut.wait.return_value = result
        result_fut.value.return_value = [result]
        return result_fut

    fut.then.side_effect = then_impl
    return fut


# ---------------------------------------------------------------------------
# Commit 5: install + sentinel + error paths
# ---------------------------------------------------------------------------


class TestInstallDDPCommHook:
    """install_ddp_comm_hook: sentinel, idempotency, type checks."""

    def test_rejects_non_ddp_model(self):
        with pytest.raises(TypeError, match="DistributedDataParallel"):
            install_ddp_comm_hook(nn.Linear(10, 10))

    def test_rejects_non_callable_base_hook(self):
        mock = _make_mock_ddp()
        with pytest.raises(TypeError, match="callable"):
            install_ddp_comm_hook(mock, base_hook="not_callable")

    def test_installs_hook_and_sets_sentinel(self):
        mock = _make_mock_ddp()

        result = install_ddp_comm_hook(mock)

        assert result is mock
        mock.register_comm_hook.assert_called_once()
        assert mock._traceml_ddp_comm_hook_installed is True

    def test_idempotent_second_call_is_noop(self):
        mock = _make_mock_ddp()
        install_ddp_comm_hook(mock)
        mock.register_comm_hook.reset_mock()

        install_ddp_comm_hook(mock)

        mock.register_comm_hook.assert_not_called()

    def test_graceful_failopen_when_hook_already_registered(self, capsys):
        mock = _make_mock_ddp(hook_raises=True)

        result = install_ddp_comm_hook(mock)

        assert result is mock
        assert mock._traceml_ddp_comm_hook_installed is False
        captured = capsys.readouterr()
        assert "[TraceML]" in captured.err
        assert "cannot register" in captured.err

    def test_default_base_hook_is_allreduce(self):
        mock = _make_mock_ddp()
        install_ddp_comm_hook(mock, base_hook=None)

        call_kwargs = mock.register_comm_hook.call_args
        assert call_kwargs is not None
        hook_fn = call_kwargs[1].get(
            "hook", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None
        )
        assert callable(hook_fn)


# ---------------------------------------------------------------------------
# Commit 6: per-bucket emission + per-step aggregation
# ---------------------------------------------------------------------------


class TestPerBucketEmission:
    """Hook emits one TimeEvent per bucket; events land in _STEP_BUFFER."""

    @pytest.fixture(autouse=True)
    def _clear_step_buffer(self):
        _STEP_BUFFER.clear()
        yield
        _STEP_BUFFER.clear()

    def _drive_hook(self, hook, n_buckets: int = 3):
        """Simulate n_buckets through the hook."""
        for i in range(n_buckets):
            bucket = _make_fake_bucket(index=i, is_last=(i == n_buckets - 1))
            hook(None, bucket)

    def test_emits_one_event_per_bucket(self):
        def base_hook(state, bucket):
            return _make_fake_future()

        hook = _traceml_ddp_comm_hook_factory(base_hook)
        self._drive_hook(hook, n_buckets=3)

        events = [e for e in _STEP_BUFFER if e.name == _WIRE_NAME]
        assert len(events) == 3

    def test_events_have_correct_scope(self):
        def base_hook(state, bucket):
            return _make_fake_future()

        hook = _traceml_ddp_comm_hook_factory(base_hook)
        self._drive_hook(hook, n_buckets=2)

        for evt in _STEP_BUFFER:
            assert evt.scope == TimeScope.STEP

    def test_events_have_nonzero_cpu_times(self):
        def base_hook(state, bucket):
            return _make_fake_future()

        hook = _traceml_ddp_comm_hook_factory(base_hook)
        self._drive_hook(hook, n_buckets=1)

        evt = list(_STEP_BUFFER)[0]
        assert evt.cpu_start > 0
        assert evt.cpu_end >= evt.cpu_start

    def test_events_from_separate_steps_stay_separate(self):
        """Two rounds of driving produce independent events."""

        def base_hook(state, bucket):
            return _make_fake_future()

        hook = _traceml_ddp_comm_hook_factory(base_hook)

        self._drive_hook(hook, n_buckets=2)
        step_1_events = list(_STEP_BUFFER)
        _STEP_BUFFER.clear()

        self._drive_hook(hook, n_buckets=2)
        step_2_events = list(_STEP_BUFFER)

        assert len(step_1_events) == 2
        assert len(step_2_events) == 2


# ---------------------------------------------------------------------------
# Commit 7: base_hook composition + pool cleanup
# ---------------------------------------------------------------------------


class TestBaseHookComposition:
    """Verify user-supplied base_hook is called and TraceML still emits."""

    @pytest.fixture(autouse=True)
    def _clear_step_buffer(self):
        _STEP_BUFFER.clear()
        yield
        _STEP_BUFFER.clear()

    def test_counting_base_hook_called_per_bucket(self):
        call_count = 0

        def counting_hook(state, bucket):
            nonlocal call_count
            call_count += 1
            return _make_fake_future()

        hook = _traceml_ddp_comm_hook_factory(counting_hook)

        for i in range(3):
            bucket = _make_fake_bucket(index=i)
            hook(None, bucket)

        assert call_count == 3
        events = [e for e in _STEP_BUFFER if e.name == _WIRE_NAME]
        assert len(events) == 3

    def test_base_hook_exception_propagates(self):
        """If base_hook raises before returning a Future, it propagates."""

        def exploding_hook(state, bucket):
            raise ValueError("boom from base_hook")

        hook = _traceml_ddp_comm_hook_factory(exploding_hook)
        bucket = _make_fake_bucket()

        with pytest.raises(ValueError, match="boom from base_hook"):
            hook(None, bucket)

    def test_base_hook_exception_returns_cuda_events_to_pool(self):
        """If base_hook raises synchronously on a CUDA device, the
        gpu_start/gpu_end events acquired before the call must be
        returned to the pool (the ``.then()`` callback never runs, so
        the hook body itself is responsible for cleanup)."""
        outstanding = {"n": 0}

        def fake_get():
            outstanding["n"] += 1
            return MagicMock()  # quacks like torch.cuda.Event

        def fake_return(evt):
            if evt is not None:
                outstanding["n"] -= 1

        def exploding_hook(state, bucket):
            raise ValueError("boom from base_hook")

        mod = "traceml_ai.instrumentation.hooks.ddp_comm_hook"
        with (
            patch(f"{mod}.torch.cuda.is_available", return_value=True),
            patch(f"{mod}.torch.cuda.current_device", return_value=0),
            patch(f"{mod}.get_cuda_event", side_effect=fake_get),
            patch(f"{mod}.return_cuda_event", side_effect=fake_return),
        ):
            hook = _traceml_ddp_comm_hook_factory(exploding_hook)
            bucket = _make_fake_bucket()

            with pytest.raises(ValueError, match="boom from base_hook"):
                hook(None, bucket)

        assert outstanding["n"] == 0, "CUDA events leaked from the pool"

    def test_then_callback_exception_does_not_crash(self):
        """If .then() callback has internal error, result still returns."""
        tensor = torch.zeros(10)

        def base_hook(state, bucket):
            return _make_fake_future(tensor)

        hook = _traceml_ddp_comm_hook_factory(base_hook)

        with patch(
            "traceml_ai.instrumentation.hooks.ddp_comm_hook.record_event",
            side_effect=RuntimeError("record_event exploded"),
        ):
            bucket = _make_fake_bucket()
            hook(None, bucket)


# ---------------------------------------------------------------------------
# Commit 7.5: auto-install + DDP unwrap via trace_step
# ---------------------------------------------------------------------------


class TestAutoInstallViaTraceStep:
    """trace_step auto-installs DDP comm hook when model is DDP wrapper."""

    def _trace_step_patches(self):
        """Context manager that mocks all trace_step internals."""
        from contextlib import ExitStack

        stack = ExitStack()
        patches = {
            "ensure_ddp": stack.enter_context(
                patch(
                    "traceml_ai.sdk.instrumentation."
                    "ensure_ddp_comm_hook_installed"
                )
            ),
            "unwrap": stack.enter_context(
                patch(
                    "traceml_ai.sdk.instrumentation._maybe_unwrap_ddp",
                    return_value=MagicMock(spec=nn.Module),
                )
            ),
            "mem_tracker": stack.enter_context(
                patch("traceml_ai.sdk.instrumentation.StepMemoryTracker")
            ),
            "flush": stack.enter_context(
                patch("traceml_ai.sdk.instrumentation.flush_step_events")
            ),
            "timed": stack.enter_context(
                patch("traceml_ai.sdk.instrumentation.timed_region")
            ),
            "fwd": stack.enter_context(
                patch("traceml_ai.sdk.instrumentation.forward_auto_timer")
            ),
            "bwd": stack.enter_context(
                patch("traceml_ai.sdk.instrumentation.backward_auto_timer")
            ),
            "h2d": stack.enter_context(
                patch("traceml_ai.sdk.instrumentation.h2d_auto_timer")
            ),
            "opt": stack.enter_context(
                patch(
                    "traceml_ai.sdk.instrumentation."
                    "ensure_optimizer_timing_installed"
                )
            ),
            "gate": stack.enter_context(
                patch(
                    "traceml_ai.sdk.instrumentation."
                    "_should_auto_install_ddp_comm_timing",
                    return_value=True,
                )
            ),
        }
        return stack, patches

    def test_auto_install_on_ddp_wrapper(self):
        """trace_step(ddp_model) calls ensure_ddp_comm_hook_installed."""
        mock_ddp = _make_mock_ddp()
        stack, patches = self._trace_step_patches()

        with stack:
            from traceml_ai.sdk.instrumentation import trace_step

            with trace_step(mock_ddp):
                pass

            patches["ensure_ddp"].assert_called_once_with(mock_ddp)

    def test_no_auto_install_on_plain_module(self):
        """trace_step(nn.Module) does NOT call ensure_ddp_comm_hook."""
        model = nn.Linear(10, 10)
        stack, patches = self._trace_step_patches()

        with stack:
            from traceml_ai.sdk.instrumentation import trace_step

            with trace_step(model):
                pass

            patches["ensure_ddp"].assert_not_called()

    def test_no_auto_install_when_gate_closed(self):
        """When the init-mode gate is closed, no auto-install.

        Mirrors init(mode='manual'/'selective') or no init config, where
        `_should_auto_install_ddp_comm_timing()` returns False.
        """
        mock_ddp = _make_mock_ddp()
        stack, patches = self._trace_step_patches()

        with stack:
            # _trace_step_patches opens the gate by default; close it here.
            patches["gate"].return_value = False

            from traceml_ai.sdk.instrumentation import trace_step

            with trace_step(mock_ddp):
                pass

            patches["ensure_ddp"].assert_not_called()

    def test_explicit_wrap_ddp_then_auto_install_is_idempotent(self):
        """wrap_ddp + trace_step on same model doesn't double-install."""
        mock_ddp = _make_mock_ddp()
        install_ddp_comm_hook(mock_ddp)
        assert mock_ddp._traceml_ddp_comm_hook_installed is True

        stack, patches = self._trace_step_patches()

        with stack:
            from traceml_ai.sdk.instrumentation import trace_step

            with trace_step(mock_ddp):
                pass

            assert mock_ddp.register_comm_hook.call_count == 1


class TestDDPUnwrap:
    """trace_step unwraps DDP for downstream id()-keyed operations."""

    def test_maybe_unwrap_ddp_returns_module(self):
        from traceml_ai.sdk.instrumentation import _maybe_unwrap_ddp

        inner = nn.Linear(10, 10)
        mock_ddp = MagicMock(spec=DistributedDataParallel)
        mock_ddp.module = inner

        result = _maybe_unwrap_ddp(mock_ddp)
        assert result is inner

    def test_maybe_unwrap_ddp_passthrough_for_plain(self):
        from traceml_ai.sdk.instrumentation import _maybe_unwrap_ddp

        model = nn.Linear(10, 10)
        result = _maybe_unwrap_ddp(model)
        assert result is model


# ---------------------------------------------------------------------------
# Commit 8: 2-rank gloo smoke test (gated)
# ---------------------------------------------------------------------------

_RUN_DIST_TESTS = os.environ.get("TRACEML_RUN_DIST_TESTS", "0") == "1"

_GLOO_WORKER_SCRIPT = """\
import os, sys, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.environ["TRACEML_SRC"])
from traceml_ai.instrumentation.hooks.ddp_comm_hook import install_ddp_comm_hook
from traceml_ai.utils.timing import _STEP_BUFFER

dist.init_process_group(backend="gloo")
rank = dist.get_rank()

model = DDP(nn.Linear(10, 10))
install_ddp_comm_hook(model)

x = torch.randn(4, 10)
loss = model(x).sum()
loss.backward()

events = [e for e in _STEP_BUFFER if e.name == "_traceml_comm:ddp_grad_sync"]
print(f"RANK={rank} EVENTS={len(events)}", flush=True)

assert len(events) >= 1, f"rank {rank}: expected >=1 event, got {len(events)}"

for evt in events:
    assert evt.cpu_end >= evt.cpu_start, "cpu_end < cpu_start"

dist.destroy_process_group()
print(f"RANK={rank} OK", flush=True)
"""


@pytest.mark.skipif(
    not _RUN_DIST_TESTS,
    reason="Set TRACEML_RUN_DIST_TESTS=1 to run distributed tests",
)
def test_two_rank_gloo_ddp_grad_sync_fires(tmp_path):
    """
    Spawn 2 gloo ranks, run one backward step, verify each rank
    produces at least one _traceml_comm:ddp_grad_sync event.
    """
    import subprocess

    script_path = tmp_path / "gloo_worker.py"
    script_path.write_text(_GLOO_WORKER_SCRIPT)

    src_path = os.path.join(os.path.dirname(__file__), "..", "src")

    env = {
        **os.environ,
        "TRACEML_SRC": os.path.abspath(src_path),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }

    procs = []
    for rank in range(2):
        rank_env = {
            **env,
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": "2",
        }
        p = subprocess.Popen(
            [sys.executable, str(script_path)],
            env=rank_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append(p)

    for p in procs:
        stdout, stderr = p.communicate(timeout=60)
        output = stdout.decode()
        assert p.returncode == 0, (
            f"Worker failed (rc={p.returncode}):\n"
            f"stdout: {output}\n"
            f"stderr: {stderr.decode()}"
        )
        assert "OK" in output
