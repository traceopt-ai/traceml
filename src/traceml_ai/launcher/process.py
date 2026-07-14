# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess lifecycle helpers for the TraceML launcher."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Optional

from traceml_ai.launcher.manifest import update_run_manifest

DEFAULT_TCP_READY_TIMEOUT_SEC = 15.0
DEFAULT_SHUTDOWN_TIMEOUT_SEC = 5.0
DEFAULT_STDERR_TAIL_BYTES = 64 * 1024
INTERRUPTED_EXIT_CODE = 130


class StderrTailCapture:
    """Continuously tee a binary stream while retaining a bounded tail."""

    def __init__(
        self,
        stream: BinaryIO,
        *,
        max_bytes: int = DEFAULT_STDERR_TAIL_BYTES,
        tee_stream: Optional[BinaryIO] = None,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")

        self._stream = stream
        self._max_bytes = int(max_bytes)
        self._tee_stream = tee_stream
        self._chunks: deque[bytes] = deque()
        self._tail_size = 0
        self._lock = threading.Lock()
        self._error: Optional[Exception] = None
        self._thread = threading.Thread(
            target=self._drain,
            name="traceml-stderr-capture",
            daemon=True,
        )
        self._thread.start()

    def _remember(self, chunk: bytes) -> None:
        with self._lock:
            if len(chunk) >= self._max_bytes:
                self._chunks.clear()
                self._chunks.append(chunk[-self._max_bytes :])
                self._tail_size = self._max_bytes
                return

            self._chunks.append(chunk)
            self._tail_size += len(chunk)
            while self._tail_size > self._max_bytes:
                excess = self._tail_size - self._max_bytes
                first = self._chunks[0]
                if len(first) <= excess:
                    self._tail_size -= len(self._chunks.popleft())
                else:
                    self._chunks[0] = first[excess:]
                    self._tail_size -= excess

    def _tee(self, chunk: bytes) -> None:
        if self._tee_stream is not None:
            self._tee_stream.write(chunk)
            self._tee_stream.flush()
            return

        binary_stderr = getattr(sys.stderr, "buffer", None)
        if binary_stderr is not None:
            binary_stderr.write(chunk)
            binary_stderr.flush()
            return

        sys.stderr.write(chunk.decode(errors="replace"))
        sys.stderr.flush()

    def _drain(self) -> None:
        read = getattr(self._stream, "read1", self._stream.read)
        while True:
            try:
                chunk = read(8192)
            except Exception as exc:
                self._error = exc
                return
            if not chunk:
                return

            try:
                self._tee(chunk)
            except Exception as exc:
                if self._error is None:
                    self._error = exc
            self._remember(chunk)

        def finish(
        self,
        output_path: Path,
        *,
        timeout_sec: float = DEFAULT_SHUTDOWN_TIMEOUT_SEC,
    ) -> Optional[Path]:
        """Stop waiting after a bounded interval and persist the captured tail."""
        self._thread.join(timeout=max(0.0, float(timeout_sec)))
        if self._thread.is_alive():
            try:
                self._stream.close()
            except Exception:
                pass
            self._thread.join(timeout=0.5)
        # If the thread is still alive after the second join, raise an exception
        if self._thread.is_alive():
            raise RuntimeError('Failed to join thread after closing stream')


        with self._lock:
            tail = b"".join(self._chunks)

        path = Path(output_path).resolve()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tail)
        except OSError as exc:
            print(
                f"[TraceML] WARNING: stderr tail could not be written: {exc}",
                file=sys.stderr,
            )
            return None

        if self._error is not None:
            print(
                f"[TraceML] WARNING: stderr capture was incomplete: {self._error}",
                file=sys.stderr,
            )
        return path


def start_stderr_tail_capture(
    proc: subprocess.Popen,
    *,
    max_bytes: int = DEFAULT_STDERR_TAIL_BYTES,
    tee_stream: Optional[BinaryIO] = None,
) -> StderrTailCapture:
    """Start draining stderr for a process launched with ``stderr=PIPE``."""
    if proc.stderr is None:
        raise ValueError("process stderr is not piped")
    return StderrTailCapture(
        proc.stderr,
        max_bytes=max_bytes,
        tee_stream=tee_stream,
    )


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
    proc: Optional[subprocess.Popen] = None,
    timeout_sec: float = DEFAULT_TCP_READY_TIMEOUT_SEC,
    poll_interval_sec: float = 0.05,
) -> bool:
    """Wait until ``(host, port)`` starts accepting TCP connections."""
    deadline = time.time() + float(timeout_sec)
    last_err: Optional[Exception] = None

    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
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
    train_cmd: list[str],
    env: dict[str, str],
    cwd: str,
    *,
    capture_stderr: bool = False,
) -> subprocess.Popen:
    """Start the user training process in a separate process group."""
    print("[TraceML] Launching TraceML executor:", " ".join(train_cmd))
    popen_kwargs = {}
    if capture_stderr:
        popen_kwargs["stderr"] = subprocess.PIPE
    return subprocess.Popen(
        train_cmd,
        env=env,
        cwd=cwd,
        start_new_session=True,
        **popen_kwargs,
    )
