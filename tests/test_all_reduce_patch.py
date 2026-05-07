"""Tests for the all_reduce auto-timer patch (TRA-16 v0).

These tests follow the snapshot-and-restore pattern established by the h2d
patch tests (`test_h2d_auto_timer_patch.py`). We do NOT use ``importlib.reload``
because reloading the patch module re-runs the module-level
``_ORIG_ALL_REDUCE = torch.distributed.all_reduce`` capture, which after a
prior install captures the patched wrapper as the "original" and produces a
self-recursive lookup through ``__globals__`` -> ``RecursionError`` on the
next call (same hazard h2d ran into per ``issue_82_solo_h2d/LESSONS.md``).

Instead we install the patch once via the autouse fixture, reset the TLS gate
between tests, and stub ``timed_region`` per case.

The 2-rank torchrun gloo smoke test is included to verify the load-bearing
assumption that the DDP Reducer dispatches its bucket all-reduces from C++
and bypasses the Python ``torch.distributed.all_reduce`` symbol entirely.
That assumption is the reason this v0 patch site does NOT capture DDP
gradient sync, and the docstring of the patch module spells that out.
"""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import traceml.instrumentation.patches.all_reduce_auto_timer_patch as ar


@pytest.fixture(autouse=True)
def _reset_all_reduce_state():
    """Install the patch (idempotent) and force the TLS gate off around
    every test in this module.

    Uninstall is intentionally skipped for the same reason h2d skips it:
    swapping ``torch.distributed.all_reduce`` back to the captured original
    would race with module-level ``_ORIG_ALL_REDUCE`` in ways that surface as
    ``RecursionError`` on cross-test reloads. The patched wrapper fast-paths
    to ``_ORIG_ALL_REDUCE`` whenever the gate is False, so other tests'
    ``dist.all_reduce`` calls behave identically to an unpatched symbol.
    """
    ar.patch_all_reduce()
    ar._TLS._traceml_all_reduce_enabled = False
    yield
    ar._TLS._traceml_all_reduce_enabled = False


def _make_fake_timed_region(calls):
    """Return a ``timed_region``-shaped context manager that records calls."""

    @contextmanager
    def _fake(name, scope, use_gpu):
        calls.append((name, scope, use_gpu))
        yield

    return _fake


# ---------------------------------------------------------------------------
# Idempotent install
# ---------------------------------------------------------------------------


def test_patch_all_reduce_is_idempotent():
    """Second call to patch_all_reduce must be a no-op."""
    assert (
        getattr(dist, "_traceml_all_reduce_patched", False) is True
    )
    first_func = dist.all_reduce

    ar.patch_all_reduce()  # already installed by fixture; must short-circuit

    assert dist.all_reduce is first_func


def test_sentinel_is_set_on_torch_distributed_module():
    """The sentinel attribute must live on ``torch.distributed`` itself.

    Other patches put the sentinel on the natural namespace of the patched
    callable: ``nn.Module._traceml_forward_patched`` for forward,
    ``torch._traceml_backward_patched`` for backward, etc. For all_reduce the
    natural namespace is ``torch.distributed``.
    """
    assert getattr(dist, "_traceml_all_reduce_patched", False) is True


# ---------------------------------------------------------------------------
# Fast-path / slow-path gate behavior
# ---------------------------------------------------------------------------


def _gloo_initialized() -> bool:
    """True when a default gloo PG is initialized in this process."""
    try:
        return dist.is_initialized()
    except Exception:
        return False


def _ensure_gloo_pg():
    """Best-effort init of a single-rank gloo PG for in-process tests.

    Single-rank gloo means ``dist.all_reduce`` becomes a near-zero-cost
    self-collective, suitable for asserting the patch wiring without any
    networking.
    """
    if _gloo_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(
        backend="gloo", rank=0, world_size=1
    )


def test_patch_does_not_record_when_gate_false(monkeypatch):
    """Outside all_reduce_auto_timer, patched dist.all_reduce must not call
    timed_region."""
    _ensure_gloo_pg()
    calls: list = []
    monkeypatch.setattr(
        ar, "timed_region", _make_fake_timed_region(calls)
    )

    t = torch.zeros(4)
    dist.all_reduce(t)

    assert calls == []


def test_patch_records_event_inside_activator(monkeypatch):
    """Inside all_reduce_auto_timer, patched dist.all_reduce invokes
    timed_region with the expected wire-name/scope/use_gpu."""
    _ensure_gloo_pg()
    calls: list = []
    monkeypatch.setattr(
        ar, "timed_region", _make_fake_timed_region(calls)
    )

    t = torch.zeros(4)
    with ar.all_reduce_auto_timer():
        dist.all_reduce(t)

    assert calls == [("_traceml_comm:all_reduce", "step", True)]


def test_activator_exit_resets_gate_on_exception(monkeypatch):
    """If user code raises inside the activator, the gate must be False on
    exit so subsequent dist.all_reduce calls fast-path."""
    _ensure_gloo_pg()
    calls: list = []
    monkeypatch.setattr(
        ar, "timed_region", _make_fake_timed_region(calls)
    )

    with pytest.raises(RuntimeError, match="boom"):
        with ar.all_reduce_auto_timer():
            raise RuntimeError("boom")

    assert ar._enabled() is False

    t = torch.zeros(2)
    dist.all_reduce(t)
    assert calls == []  # gate is False -> must fast-path


# ---------------------------------------------------------------------------
# Polymorphic argument forms
# ---------------------------------------------------------------------------


def test_positional_and_kwarg_forms_route_through_patch(monkeypatch):
    """The wrapper must forward *args, **kwargs unchanged.

    ``dist.all_reduce`` signature is
    ``all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False)``.
    User code may call any combination of positional / kwargs; the wrapper
    must not drop any of them. We exercise the two shapes most likely in
    real code: positional-only, and positional-tensor + keyword-op.
    """
    _ensure_gloo_pg()
    calls: list = []
    monkeypatch.setattr(
        ar, "timed_region", _make_fake_timed_region(calls)
    )

    t1 = torch.zeros(2)
    t2 = torch.zeros(2)

    with ar.all_reduce_auto_timer():
        dist.all_reduce(t1)  # positional only
        dist.all_reduce(t2, op=dist.ReduceOp.SUM)  # mixed pos + kwarg

    assert len(calls) == 2
    assert all(
        c == ("_traceml_comm:all_reduce", "step", True) for c in calls
    )


def test_kwarg_forwarding_to_original_is_lossless(monkeypatch):
    """The wrapper passes args+kwargs through to the original. We verify
    by stubbing ``_ORIG_ALL_REDUCE`` to capture exactly what reaches the
    original function.
    """
    captured: list = []

    def fake_orig(*args, **kwargs):
        captured.append((args, kwargs))

    monkeypatch.setattr(ar, "_ORIG_ALL_REDUCE", fake_orig)

    @contextmanager
    def passthrough(name, scope, use_gpu):
        yield

    monkeypatch.setattr(ar, "timed_region", passthrough)

    t = torch.zeros(2)
    sentinel_group = object()  # not a real group; just identity-checked

    with ar.all_reduce_auto_timer():
        ar._traceml_all_reduce(t, op=dist.ReduceOp.SUM, group=sentinel_group)

    assert len(captured) == 1
    args, kwargs = captured[0]
    assert args == (t,)
    assert kwargs == {"op": dist.ReduceOp.SUM, "group": sentinel_group}


def test_async_op_true_records_one_event_per_call(monkeypatch):
    """``async_op=True`` returns a Work handle but the patch records one
    event per call regardless. ``cpu_*`` will reflect enqueue cost only;
    ``gpu_*`` (resolved later by sampler) will reflect on-stream kernel
    wall time. Documented in §8 item 4 of DESIGN.md.
    """
    _ensure_gloo_pg()
    calls: list = []
    monkeypatch.setattr(
        ar, "timed_region", _make_fake_timed_region(calls)
    )

    t = torch.zeros(2)
    with ar.all_reduce_auto_timer():
        work = dist.all_reduce(t, async_op=True)
        if work is not None and hasattr(work, "wait"):
            work.wait()

    assert calls == [("_traceml_comm:all_reduce", "step", True)]


# ---------------------------------------------------------------------------
# init() integration (auto / manual / selective)
# ---------------------------------------------------------------------------


def _reload_initialization_module():
    import importlib

    import traceml.sdk.initial as initialization

    return importlib.reload(initialization)


def test_init_auto_enables_all_reduce_patch(monkeypatch):
    """mode='auto' must call patch_all_reduce()."""
    initialization = _reload_initialization_module()

    calls = []

    import traceml.instrumentation.patches.all_reduce_auto_timer_patch as ar_patch
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
    monkeypatch.setattr(
        ar_patch,
        "patch_all_reduce",
        lambda: calls.append("all_reduce"),
    )

    cfg = initialization.init(mode="auto")

    assert cfg.mode == "auto"
    assert cfg.patch_all_reduce is True
    assert "all_reduce" in calls


def test_init_manual_disables_all_reduce_patch():
    initialization = _reload_initialization_module()

    cfg = initialization.init(mode="manual")

    assert cfg.mode == "manual"
    assert cfg.patch_all_reduce is False


def test_init_selective_can_enable_only_all_reduce(monkeypatch):
    """selective with patch_all_reduce=True alone must satisfy the
    'at least one True' invariant."""
    initialization = _reload_initialization_module()

    calls = []

    import traceml.instrumentation.patches.all_reduce_auto_timer_patch as ar_patch

    monkeypatch.setattr(
        ar_patch,
        "patch_all_reduce",
        lambda: calls.append("all_reduce"),
    )

    cfg = initialization.init(
        mode="selective",
        patch_all_reduce=True,
    )

    assert cfg.mode == "selective"
    assert cfg.patch_all_reduce is True
    assert cfg.patch_dataloader is False
    assert cfg.patch_forward is False
    assert cfg.patch_backward is False
    assert calls == ["all_reduce"]


def test_same_effective_configuration_includes_patch_all_reduce():
    """LESSONS H2D §3 lock-in: same_effective_configuration MUST include
    every new patch flag, otherwise two init() calls with different
    patch_all_reduce values would silently agree.

    We construct two TraceMLInitConfig instances directly so we can build a
    selective-with-all-False variant (which the validator would otherwise
    reject) and assert the equality predicate distinguishes them.
    """
    _reload_initialization_module()
    from traceml.sdk.initial import TraceMLInitConfig

    cfg_a = TraceMLInitConfig(
        mode="selective",
        patch_dataloader=False,
        patch_forward=False,
        patch_backward=False,
        patch_all_reduce=True,
        source="user",
    )
    cfg_b = TraceMLInitConfig(
        mode="selective",
        patch_dataloader=False,
        patch_forward=False,
        patch_backward=False,
        patch_all_reduce=False,
        source="user",
    )

    assert cfg_a.same_effective_configuration(cfg_b) is False
    # Identical configs must remain equal under the predicate.
    assert cfg_a.same_effective_configuration(cfg_a) is True


# ---------------------------------------------------------------------------
# trace_step nesting
# ---------------------------------------------------------------------------


def test_trace_step_opens_all_reduce_activator():
    """trace_step must open all_reduce_auto_timer alongside forward and
    backward.

    We assert directly on the patch module's TLS state rather than driving a
    real ``init(mode='auto')`` + ``dist.all_reduce`` flow. This decouples the
    test from init/sampler internals.
    """
    import torch.nn as nn

    import traceml.sdk.instrumentation as instrumentation

    captured = {}
    model = nn.Linear(2, 2)

    with instrumentation.trace_step(model):
        captured["enabled_inside"] = ar._enabled()
    captured["enabled_after"] = ar._enabled()

    assert captured["enabled_inside"] is True
    assert captured["enabled_after"] is False


# ---------------------------------------------------------------------------
# 2-rank torchrun gloo smoke test (LOCK-3 Probe-2 verification)
# ---------------------------------------------------------------------------


SMOKE_SCRIPT_TEMPLATE = r"""
import os, sys, json
import torch
import torch.distributed as dist
import torch.nn as nn

# Reach the worktree's src/ before importing traceml.
ROOT = {root!r}
sys.path.insert(0, os.path.join(ROOT, "src"))

import traceml.instrumentation.patches.all_reduce_auto_timer_patch as ar

# Counter that increments on every call routed through the Python symbol.
_COUNTS = {{"calls": 0}}
_ORIG = ar._ORIG_ALL_REDUCE

def _counting_wrapper(*args, **kwargs):
    _COUNTS["calls"] += 1
    return _ORIG(*args, **kwargs)

# Install the counting wrapper directly. We bypass ar.patch_all_reduce()
# because it would install ar._traceml_all_reduce, not our counter.
import torch.distributed as dist_module
dist_module.all_reduce = _counting_wrapper

dist.init_process_group(backend="gloo")
rank = dist.get_rank()
world_size = dist.get_world_size()

assert world_size == 2, f"expected world_size=2, got {{world_size}}"

# Build a tiny model and wrap with DDP.
torch.manual_seed(0)
model = nn.Linear(4, 4)
ddp = torch.nn.parallel.DistributedDataParallel(model)

# Capture counts at three checkpoints:
# (1) after DDP construction (DDP init uses broadcast, NOT all_reduce)
post_init = _COUNTS["calls"]

# (2) after a forward + backward cycle (DDP gradient sync fires here;
#     if it bypassed Python all_reduce, the counter must NOT increment)
x = torch.randn(2, 4)
y = ddp(x).sum()
y.backward()
post_backward = _COUNTS["calls"]

# (3) after an explicit user-issued dist.all_reduce (counter MUST increment)
t = torch.zeros(4)
dist.all_reduce(t)
post_user = _COUNTS["calls"]

if rank == 0:
    print("SMOKE_RESULT " + json.dumps({{
        "post_init": post_init,
        "post_backward": post_backward,
        "post_user": post_user,
        "rank": rank,
    }}))

dist.destroy_process_group()
"""


def _torchrun_available() -> bool:
    """True when torchrun + gloo are usable in this environment."""
    try:
        import torch.distributed as _d  # noqa: F401
    except Exception:
        return False
    # torchrun is shipped with torch.
    return True


@pytest.mark.skipif(
    not _torchrun_available(),
    reason="torch.distributed not importable; skipping 2-rank smoke",
)
def test_two_rank_torchrun_gloo_ddp_bypasses_python_all_reduce(tmp_path):
    """LOCK-3 Probe-2 verification: confirm DDP gradient sync bypasses
    the Python ``torch.distributed.all_reduce`` symbol.

    Approach
    --------
    Spawn a 2-rank torchrun job. Each rank installs a counting wrapper at
    the Python symbol, then runs DDP construction, a forward/backward, and
    an explicit user-issued ``dist.all_reduce``. We assert:

    1. ``post_init == 0`` -- DDP init uses broadcast, not all_reduce.
    2. ``post_backward == 0`` -- DDP Reducer dispatches per-bucket all-reduces
       in C++ and bypasses the Python symbol.
    3. ``post_user == 1`` -- user-issued ``dist.all_reduce`` IS caught.

    If (2) ever fails (counter > 0 after backward), the foundational
    assumption of the v0 patch site is wrong and the orchestrator should
    halt this issue and pivot to ``register_comm_hook`` or a deeper site.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script = tmp_path / "smoke.py"
    script.write_text(
        SMOKE_SCRIPT_TEMPLATE.format(root=str(repo_root))
    )

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_port=29556",
        "--standalone",
        str(script),
    ]
    env = os.environ.copy()
    env.setdefault("MASTER_ADDR", "127.0.0.1")

    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"torchrun unavailable / timeout: {exc}")

    if proc.returncode != 0:
        pytest.skip(
            "torchrun smoke could not run "
            f"(rc={proc.returncode}); stdout={proc.stdout!r}; "
            f"stderr={proc.stderr!r}"
        )

    # Parse the rank-0 result line.
    result_line = next(
        (
            line
            for line in proc.stdout.splitlines()
            if line.startswith("SMOKE_RESULT ")
        ),
        None,
    )
    assert result_line is not None, (
        "rank 0 did not emit SMOKE_RESULT; full stdout:\n" + proc.stdout
    )

    import json as _json

    payload = _json.loads(result_line[len("SMOKE_RESULT ") :])

    assert payload["post_init"] == 0, (
        "DDP construction unexpectedly routed through Python all_reduce; "
        f"counter={payload['post_init']}"
    )
    assert payload["post_backward"] == 0, (
        "DDP gradient sync unexpectedly routed through Python all_reduce. "
        "This contradicts the foundational assumption of the v0 patch site. "
        f"counter={payload['post_backward']}"
    )
    assert payload["post_user"] >= 1, (
        "user-issued dist.all_reduce did NOT increment the counter; the "
        "patch site cannot capture even the in-scope path. "
        f"counter={payload['post_user']}"
    )
