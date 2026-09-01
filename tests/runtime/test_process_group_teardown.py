"""Teardown must reach grandchildren, not only the direct child.

A training launcher spawns workers of its own, so terminating just the
child leaves those workers holding the GPU and open file handles. The
end-to-end test below builds that exact shape, a child that spawns a
grandchild, and asserts the grandchild is gone after teardown.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from traceml_ai.launcher import process as process_mod

# Child spawns a long-lived grandchild, prints its pid, then sleeps.
_CHILD_SOURCE = """
import subprocess, sys, time
gc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(gc.pid, flush=True)
time.sleep(120)
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
    signals: list[int] = []
    monkeypatch.setattr(
        process_mod.os,
        "killpg",
        lambda _pid, signum: signals.append(signum),
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
            self.returncode = -signal.SIGKILL
            return self.returncode

        def terminate(self):
            raise AssertionError("killpg should handle termination")

        def kill(self):
            raise AssertionError("killpg should handle termination")

    proc = ReapedAfterKill()
    process_mod.terminate_process_group(proc, timeout_sec=0.01)

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert proc.wait_calls == 2
    assert proc.returncode == -signal.SIGKILL


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
