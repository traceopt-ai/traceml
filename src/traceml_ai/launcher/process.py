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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Optional

from traceml_ai.launcher.manifest import update_run_manifest

_IS_WINDOWS = sys.platform == "win32"
# Absent on POSIX, where start_new_session is used instead.
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

DEFAULT_TCP_READY_TIMEOUT_SEC = 15.0
DEFAULT_SHUTDOWN_TIMEOUT_SEC = 5.0
DEFAULT_STDERR_TAIL_BYTES = 64 * 1024
_OUTPUT_READ_BYTES = 64 * 1024
_OUTPUT_STOP_GRACE_SEC = 0.1
INTERRUPTED_EXIT_CODE = 130


@dataclass(frozen=True)
class TrainingOutcome:
    """The result directly observed from the supervised training process.

    A negative ``subprocess`` return code identifies a POSIX signal.  Keep the
    raw value for accurate reporting and convert it only at the CLI boundary.
    A torchrun-reported worker failure is normally a positive exit code and is
    deliberately not reclassified by inspecting its output.
    """

    returncode: int

    @property
    def signal_name(self) -> Optional[str]:
        """Return the directly observed POSIX signal name, when available."""
        if _IS_WINDOWS or self.returncode >= 0:
            return None
        try:
            return signal.Signals(-self.returncode).name
        except ValueError:
            return None

    @property
    def cli_exit_code(self) -> int:
        """Return the conventional shell exit code for this outcome."""
        if not _IS_WINDOWS and self.returncode < 0:
            return 128 - self.returncode
        return self.returncode


@dataclass(frozen=True)
class ProcessOutputResult:
    """Final facts from a bounded subprocess-output drain."""

    stdout_path: Optional[Path]
    stderr_path: Optional[Path]
    stderr_tail: bytes
    warning: Optional[str]


class ProcessOutputDrainer:
    """Concurrently persist and optionally mirror raw process output.

    One worker owns each supplied stream and its optional file. File failures
    switch that stream to its fallback without stopping the drain. ``finish``
    returns one warning value instead of raising into training control flow.
    """

    def __init__(
        self,
        *,
        stdout: Optional[BinaryIO] = None,
        stderr: Optional[BinaryIO] = None,
        stdout_path: Optional[Path | str] = None,
        stderr_path: Optional[Path | str] = None,
        stdout_mirror: Optional[BinaryIO] = None,
        stderr_mirror: Optional[BinaryIO] = None,
        stdout_fallback: Optional[BinaryIO] = None,
        stderr_fallback: Optional[BinaryIO] = None,
        max_stderr_tail_bytes: int = DEFAULT_STDERR_TAIL_BYTES,
    ) -> None:
        if max_stderr_tail_bytes <= 0:
            raise ValueError("max_stderr_tail_bytes must be greater than zero")

        self._max_stderr_tail_bytes = int(max_stderr_tail_bytes)
        self._stderr_tail = bytearray()
        self._warnings: list[str] = []
        self._lock = threading.Lock()
        self._result: Optional[ProcessOutputResult] = None
        self._completed_paths: dict[str, Path] = {}
        self._workers: list[tuple[str, BinaryIO, threading.Thread]] = []

        for name, source, path, mirror, fallback in (
            ("stdout", stdout, stdout_path, stdout_mirror, stdout_fallback),
            ("stderr", stderr, stderr_path, stderr_mirror, stderr_fallback),
        ):
            if source is None:
                continue
            resolved_path = Path(path).resolve() if path is not None else None
            thread = threading.Thread(
                target=self._drain,
                args=(
                    name,
                    source,
                    resolved_path,
                    mirror,
                    fallback,
                ),
                name=f"traceml-process-{name}",
                daemon=True,
            )
            self._workers.append((name, source, thread))
            try:
                thread.start()
            except Exception as exc:
                self._workers.pop()
                self._warn(f"{name} output worker could not start: {exc}")
                self._close(source, f"{name} output stream")

    def _warn(self, message: str) -> None:
        with self._lock:
            if message not in self._warnings:
                self._warnings.append(message)

    def _remember_stderr(self, chunk: bytes) -> None:
        with self._lock:
            self._stderr_tail.extend(chunk)
            excess = len(self._stderr_tail) - self._max_stderr_tail_bytes
            if excess > 0:
                del self._stderr_tail[:excess]

    @staticmethod
    def _write(target: BinaryIO, chunk: bytes) -> None:
        offset = 0
        while offset < len(chunk):
            written = target.write(chunk[offset:])
            if written is None or int(written) <= 0:
                raise OSError("output destination made no write progress")
            offset += int(written)
        target.flush()

    def _drain(
        self,
        name: str,
        source: BinaryIO,
        path: Optional[Path],
        mirror: Optional[BinaryIO],
        fallback: Optional[BinaryIO],
    ) -> None:
        sink: Optional[BinaryIO] = None
        sink_failed = False
        drained = False
        try:
            if path is not None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sink = path.open("wb", buffering=0)
                except Exception as exc:
                    sink_failed = True
                    self._warn(
                        f"{name} output file could not be opened: {exc}"
                    )

            read = getattr(source, "read1", source.read)
            while True:
                chunk = read(_OUTPUT_READ_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise TypeError("subprocess output stream must be binary")
                raw_chunk = bytes(chunk)
                if name == "stderr":
                    self._remember_stderr(raw_chunk)

                if sink is not None:
                    try:
                        self._write(sink, raw_chunk)
                    except Exception as exc:
                        sink_failed = True
                        self._warn(f"{name} output file write failed: {exc}")
                        self._close(sink, f"{name} output file")
                        sink = None

                mirrored = False
                if mirror is not None:
                    try:
                        self._write(mirror, raw_chunk)
                        mirrored = True
                    except Exception as exc:
                        self._warn(f"{name} output mirror failed: {exc}")
                        mirror = None

                if sink_failed and not mirrored and fallback is not None:
                    try:
                        self._write(fallback, raw_chunk)
                    except Exception as exc:
                        self._warn(f"{name} output fallback failed: {exc}")
                        fallback = None
            drained = True
        except Exception as exc:
            self._warn(f"{name} output read failed: {exc}")
        finally:
            self._close(source, f"{name} output stream")
            if sink is not None and not self._close(
                sink, f"{name} output file"
            ):
                sink_failed = True
            if drained and path is not None and not sink_failed:
                with self._lock:
                    self._completed_paths[name] = path

    def _close(self, target: BinaryIO, label: str) -> bool:
        try:
            target.close()
        except Exception as exc:
            self._warn(f"{label} close failed: {exc}")
            return False
        return True

    def finish(
        self,
        *,
        timeout_sec: float = DEFAULT_SHUTDOWN_TIMEOUT_SEC,
    ) -> ProcessOutputResult:
        """Finish once, returning confirmed paths, stderr tail, and warning."""
        if self._result is not None:
            return self._result

        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        for _, _, thread in self._workers:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

        active = [worker for worker in self._workers if worker[2].is_alive()]
        timed_out = {name for name, _, _ in active}
        if active:
            self._warn(
                "output drain timed out for "
                + ", ".join(name for name, _, _ in active)
            )
            for name, source, _ in active:
                try:
                    threading.Thread(
                        target=self._close,
                        args=(source, f"{name} output stream"),
                        daemon=True,
                    ).start()
                except Exception as exc:
                    self._warn(f"{name} output close could not start: {exc}")

            stop_deadline = time.monotonic() + _OUTPUT_STOP_GRACE_SEC
            for _, _, thread in active:
                thread.join(timeout=max(0.0, stop_deadline - time.monotonic()))

        with self._lock:
            stderr_tail = bytes(self._stderr_tail)
            warning = "; ".join(self._warnings) or None
            completed_paths = {
                name: path
                for name, path in self._completed_paths.items()
                if name not in timed_out
            }
        self._result = ProcessOutputResult(
            stdout_path=completed_paths.get("stdout"),
            stderr_path=completed_paths.get("stderr"),
            stderr_tail=stderr_tail,
            warning=warning,
        )
        return self._result


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
            except Exception as exc:
                print(
                    f"[TraceML] WARNING: failed to close stderr capture stream: {exc}",
                    file=sys.stderr,
                )

            self._thread.join(timeout=0.5)

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


def process_group_kwargs() -> dict[str, Any]:
    """Popen keyword arguments that isolate the child in its own group.

    ``start_new_session`` is a ``setsid()`` call and exists only on POSIX.
    On Windows the equivalent is the CREATE_NEW_PROCESS_GROUP creation flag.
    Without one of the two, teardown can only reach the direct child and any
    grandchildren it spawned, such as torchrun or DataLoader workers, are
    left running.
    """
    if _IS_WINDOWS:
        return {"creationflags": _CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _taskkill_tree(pid: int, *, force: bool) -> bool:
    """Terminate a Windows process tree. Return True if taskkill ran.

    Windows has no process-group signal, so the tree is walked by pid.
    ``proc.terminate()`` alone would end only the direct child.
    """
    cmd = ["taskkill", "/T", "/PID", str(pid)]
    if force:
        cmd.insert(1, "/F")

    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=DEFAULT_SHUTDOWN_TIMEOUT_SEC,
        )
    except Exception:
        return False

    # 128 means the pid was already gone, which is a success for our purpose.
    return completed.returncode in (0, 128)


def terminate_process_group(
    proc: Optional[subprocess.Popen],
    timeout_sec: float = DEFAULT_SHUTDOWN_TIMEOUT_SEC,
) -> None:
    """Best-effort termination for a subprocess process group."""
    if proc is None or proc.poll() is not None:
        return

    if _IS_WINDOWS:
        if not _taskkill_tree(proc.pid, force=False):
            try:
                proc.terminate()
            except Exception:
                pass
    else:
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

    if _IS_WINDOWS:
        if not _taskkill_tree(proc.pid, force=True):
            try:
                proc.kill()
            except Exception:
                pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # Reap the direct child after forced termination so callers can rely on
    # ``returncode`` instead of mistaking an unreaped process for a clean exit.
    try:
        proc.wait(timeout=min(timeout_sec, DEFAULT_SHUTDOWN_TIMEOUT_SEC))
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
        **process_group_kwargs(),
    )


def start_training_process(
    train_cmd: list[str],
    env: dict[str, str],
    cwd: str,
    *,
    capture_stderr: bool = False,
) -> subprocess.Popen:
    """Start the user training process in a separate process group."""
    print("[TraceML] Launching training process:", " ".join(train_cmd))
    popen_kwargs = {}
    if capture_stderr:
        popen_kwargs["stderr"] = subprocess.PIPE
    return subprocess.Popen(
        train_cmd,
        env=env,
        cwd=cwd,
        **process_group_kwargs(),
        **popen_kwargs,
    )
