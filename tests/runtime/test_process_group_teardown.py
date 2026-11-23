"""Teardown must reach grandchildren, not only the direct child.

A training launcher spawns workers of its own, so terminating just the
child leaves those workers holding the GPU and open file handles. The
end-to-end test below builds that exact shape, a child that spawns a
grandchild, and asserts the grandchild is gone after teardown.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from traceml_ai.launcher import process as process_mod

# Child spawns a long-lived grandchild, prints its pid, then sleeps.
_CHILD_SOURCE = """
import subprocess, sys, time
gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(gc.pid, flush=True)
time.sleep(120)
"""


# Stands in for `traceml run`: starts the aggregator the same way, prints
# its pid, then echoes its stderr until killed.
_LAUNCHER_SOURCE = """
import os, sys
from traceml_ai.launcher.process import start_aggregator_process
proc = start_aggregator_process(dict(os.environ), os.getcwd())
print("PID", proc.pid, flush=True)
for line in proc.stderr:
    sys.stdout.write(line.decode(errors="replace"))
    sys.stdout.flush()
"""

# Loaded by every Python process the end-to-end test starts. A process
# still alive after 15 s writes all its thread stacks, so a hang on a CI
# runner shows where it is stuck instead of only timing out.
_STACK_DUMP_SOURCE = """
import faulthandler, os
_dir = os.environ.get("TRACEML_TEST_STACK_DIR")
if _dir:
    _f = open(os.path.join(_dir, f"stacks-{os.getpid()}.txt"), "w")
    faulthandler.dump_traceback_later(15, file=_f)
"""


def _pid_alive(pid: int) -> bool:
    """Return True while ``pid`` is still running."""
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
        ).stdout
        return str(pid) in out

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_gone(pid: int, timeout_sec: float = 10.0) -> bool:
    """Wait for ``pid`` to disappear, returning whether it did."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    return not _pid_alive(pid)


def test_process_group_kwargs_matches_the_platform() -> None:
    """POSIX gets a new session, Windows gets a new process group."""
    kwargs = process_mod.process_group_kwargs()

    if sys.platform == "win32":
        assert kwargs == {
            "creationflags": process_mod._CREATE_NEW_PROCESS_GROUP
        }
        assert kwargs["creationflags"] != 0
    else:
        assert kwargs == {"start_new_session": True}


def test_shutdown_handler_runs_output_cleanup(monkeypatch) -> None:
    handlers = {}
    events = []
    proc = object()
    monkeypatch.setattr(
        process_mod.signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )
    monkeypatch.setattr(
        process_mod,
        "terminate_process_group",
        lambda target, **_kwargs: events.append(("terminate", target)),
    )

    process_mod.install_shutdown_handlers(
        lambda: (proc,),
        cleanup=lambda: events.append(("cleanup", None)),
    )

    with pytest.raises(SystemExit) as exc:
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    assert exc.value.code == process_mod.INTERRUPTED_EXIT_CODE
    assert events == [("terminate", proc), ("cleanup", None)]


def test_windows_branch_kills_the_tree(monkeypatch) -> None:
    """The Windows path must walk the tree instead of ending one process."""
    monkeypatch.setattr(process_mod, "_IS_WINDOWS", True)
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(process_mod.subprocess, "run", fake_run)

    class StillRunning:
        pid = 4321

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("cmd", timeout)

        def terminate(self):
            raise AssertionError("must not fall back while taskkill works")

        def kill(self):
            raise AssertionError("must not fall back while taskkill works")

    process_mod.terminate_process_group(StillRunning(), timeout_sec=0.01)

    assert len(calls) == 2, calls
    graceful, forced = calls
    assert graceful == ["taskkill", "/T", "/PID", "4321"]
    assert forced == ["taskkill", "/F", "/T", "/PID", "4321"]


def test_windows_branch_falls_back_when_taskkill_is_unavailable(
    monkeypatch,
) -> None:
    """A missing taskkill must not leave the child running."""
    monkeypatch.setattr(process_mod, "_IS_WINDOWS", True)
    monkeypatch.setattr(
        process_mod.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
    )
    terminated: list[str] = []

    class StillRunning:
        pid = 99

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("cmd", timeout)

        def terminate(self):
            terminated.append("terminate")

        def kill(self):
            terminated.append("kill")

    process_mod.terminate_process_group(StillRunning(), timeout_sec=0.01)

    assert terminated == ["terminate", "kill"]


def test_forced_termination_reaps_the_direct_child(monkeypatch) -> None:
    """The forced path must populate returncode before returning."""
    monkeypatch.setattr(process_mod, "_IS_WINDOWS", False)
    sigkill = getattr(signal, "SIGKILL", 9)
    monkeypatch.setattr(process_mod.signal, "SIGKILL", sigkill, raising=False)
    signals: list[int] = []
    monkeypatch.setattr(
        process_mod.os,
        "killpg",
        lambda _pid, signum: signals.append(signum),
        raising=False,
    )

    class ReapedAfterKill:
        pid = 4321
        returncode = None
        wait_calls = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired("cmd", timeout)
            self.returncode = -sigkill
            return self.returncode

        def terminate(self):
            raise AssertionError("killpg should handle termination")

        def kill(self):
            raise AssertionError("killpg should handle termination")

    proc = ReapedAfterKill()
    process_mod.terminate_process_group(proc, timeout_sec=0.01)

    assert signals == [signal.SIGTERM, sigkill]
    assert proc.wait_calls == 2
    assert proc.returncode == -sigkill


def test_terminate_process_group_reaps_a_grandchild() -> None:
    """End to end: the grandchild must not survive teardown."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD_SOURCE],
        stdout=subprocess.PIPE,
        text=True,
        **process_mod.process_group_kwargs(),
    )

    try:
        line = proc.stdout.readline().strip()
        grandchild_pid = int(line)
        assert _pid_alive(grandchild_pid), "grandchild never started"

        process_mod.terminate_process_group(proc, timeout_sec=5.0)

        assert _wait_gone(proc.pid), "child survived teardown"
        assert _wait_gone(
            grandchild_pid
        ), f"grandchild {grandchild_pid} survived teardown"
    finally:
        for pid in (locals().get("grandchild_pid"), proc.pid):
            if pid is None:
                continue
            try:
                os.kill(pid, 9)
            except Exception:
                pass


def _pump(stream, sink: queue.Queue) -> None:
    """Copy lines from ``stream`` to ``sink``, then a ``None`` at EOF."""
    for line in stream:
        sink.put(line)
    sink.put(None)


def _diagnostics(lines: list[str], stack_dir: Path) -> str:
    """Return the launcher output and any stack dumps, for a failure."""
    dumps = [
        f"--- {p.name}\n{p.read_text()}"
        for p in sorted(stack_dir.glob("stacks-*.txt"))
        if p.stat().st_size
    ]
    return "".join(lines) + "\n".join(dumps)


def test_aggregator_exits_when_its_launcher_is_killed(tmp_path) -> None:
    """End to end: a killed launcher must not leave the aggregator behind."""
    src = str(Path(process_mod.__file__).resolve().parents[2])
    stack_dir = tmp_path / "stacks"
    stack_dir.mkdir()
    (stack_dir / "sitecustomize.py").write_text(_STACK_DUMP_SOURCE)
    path = os.pathsep.join(
        filter(None, (str(stack_dir), src, os.environ.get("PYTHONPATH")))
    )
    env = {
        **os.environ,
        "PYTHONPATH": path,
        "TRACEML_TEST_STACK_DIR": str(stack_dir),
        "TRACEML_AGGREGATOR_PORT": "0",
        "TRACEML_LOGS_DIR": str(tmp_path),
        "TRACEML_FINALIZE_TIMEOUT_SEC": "1",
    }
    # The launcher's own stdin stays open, so only a pipe of its own that
    # dies with it can tell the aggregator it is gone.
    launcher = subprocess.Popen(
        [sys.executable, "-c", _LAUNCHER_SOURCE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(tmp_path),
        **process_mod.process_group_kwargs(),
    )
    # Read on a thread so a launcher or aggregator that never gets ready
    # fails the test after a bounded wait instead of blocking it.
    out: queue.Queue = queue.Queue()
    threading.Thread(
        target=_pump, args=(launcher.stdout, out), daemon=True
    ).start()
    lines: list[str] = []
    aggregator_pid = None
    try:
        deadline = time.monotonic() + 60.0
        while not any("Aggregator ready" in s for s in lines):
            try:
                line = out.get(timeout=max(deadline - time.monotonic(), 0))
            except queue.Empty:
                line = None
            if line is None:
                pytest.fail(
                    "aggregator never reported ready:\n"
                    + _diagnostics(lines, stack_dir)
                )
            lines.append(line)
            if line.startswith("PID "):
                aggregator_pid = int(line.split()[1])

        launcher.kill()
        launcher.wait(timeout=10.0)

        assert _wait_gone(aggregator_pid, timeout_sec=20.0), _diagnostics(
            lines, stack_dir
        )
    finally:
        launcher.kill()
        if aggregator_pid is not None and _pid_alive(aggregator_pid):
            os.kill(aggregator_pid, 9)
