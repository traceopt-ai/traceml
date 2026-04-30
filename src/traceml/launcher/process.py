"""Subprocess lifecycle helpers for the TraceML launcher."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from traceml.launcher.manifest import update_run_manifest

DEFAULT_TCP_READY_TIMEOUT_SEC = 15.0
DEFAULT_SHUTDOWN_TIMEOUT_SEC = 5.0
INTERRUPTED_EXIT_CODE = 130


def build_torchrun_base_cmd(nproc_per_node: int) -> list[str]:
    """Build a torchrun command using the current Python interpreter."""
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={int(nproc_per_node)}",
    ]


def terminate_process_group(
    proc: Optional[subprocess.Popen],
    timeout_sec: float = DEFAULT_SHUTDOWN_TIMEOUT_SEC,
) -> None:
    """Best-effort termination for a subprocess process group."""
    if proc is None or proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        proc.wait(timeout=timeout_sec)
        return
    except Exception:
        pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def wait_for_tcp_listen(
    host: str,
    port: int,
    proc: subprocess.Popen,
    timeout_sec: float = DEFAULT_TCP_READY_TIMEOUT_SEC,
    poll_interval_sec: float = 0.05,
) -> bool:
    """Wait until ``(host, port)`` starts accepting TCP connections."""
    deadline = time.time() + float(timeout_sec)
    last_err: Optional[Exception] = None

    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection((host, int(port)), timeout=0.25):
                return True
        except Exception as exc:
            last_err = exc
            time.sleep(float(poll_interval_sec))

    if last_err is not None:
        print(
            f"[TraceML] Aggregator did not become ready on {host}:{port} "
            f"(last error: {last_err})",
            file=sys.stderr,
        )
    return False


def install_shutdown_handlers(
    get_procs: Callable[[], Iterable[Optional[subprocess.Popen]]],
    manifest_path: Optional[Path] = None,
) -> None:
    """Install SIGINT/SIGTERM handlers that terminate child process groups."""
    already_handled = {"value": False}

    def _handler(signum: int, _frame: Any) -> None:
        if already_handled["value"]:
            raise SystemExit(INTERRUPTED_EXIT_CODE)
        already_handled["value"] = True

        print(
            f"\n[TraceML] Signal {signum} received; terminating processes...",
            file=sys.stderr,
        )

        if manifest_path is not None:
            try:
                update_run_manifest(manifest_path, status="interrupted")
            except Exception:
                pass

        for proc in get_procs():
            terminate_process_group(
                proc, timeout_sec=DEFAULT_SHUTDOWN_TIMEOUT_SEC
            )

        raise SystemExit(INTERRUPTED_EXIT_CODE)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def start_aggregator_process(
    env: dict[str, str], cwd: str
) -> subprocess.Popen:
    """Start the TraceML aggregator in a separate process group."""
    traceml_root = Path(__file__).resolve().parents[1]
    aggregator_path = traceml_root / "aggregator" / "aggregator_main.py"
    if not aggregator_path.exists():
        raise FileNotFoundError(
            f"Aggregator entrypoint not found: {aggregator_path}"
        )

    cmd = [sys.executable, str(aggregator_path)]
    print("[TraceML] Launching TraceML aggregator:", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )


def start_training_process(
    train_cmd: list[str], env: dict[str, str], cwd: str
) -> subprocess.Popen:
    """Start the user training process in a separate process group."""
    print("[TraceML] Launching TraceML executor:", " ".join(train_cmd))
    return subprocess.Popen(
        train_cmd,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )
