# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from unittest.mock import Mock

from traceml_ai.launcher.process import (
    DEFAULT_STDERR_TAIL_BYTES,
    ProcessOutputDrainer,
    start_aggregator_process,
    start_training_process,
)


class _ChunkStream:
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = deque(chunks)
        self.closed = False

    def read(self, _size: int) -> bytes:
        return self._chunks.popleft() if self._chunks else b""

    def close(self) -> None:
        self.closed = True


class _FailingSink(io.BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self._writes = 0

    def write(self, data: bytes) -> int:
        self._writes += 1
        if self._writes == 2:
            raise OSError("disk unavailable")
        return super().write(data)


class _BlockingStream:
    def __init__(self) -> None:
        self._closed = threading.Event()

    def read(self, _size: int) -> bytes:
        self._closed.wait(timeout=30)
        return b""

    def close(self) -> None:
        self._closed.set()


def test_drains_flooded_streams_as_exact_separate_bytes(tmp_path) -> None:
    stdout_chunk = b"\xffOUT\x00" * 128
    stderr_chunk = b"\xfeERR\n" * 128
    repeats = 1024
    stdout_end = b"stdout-partial"
    stderr_end = b"stderr-partial"
    script = (
        "import os\n"
        "from threading import Thread\n"
        f"stdout_chunk = {stdout_chunk!r}\n"
        f"stderr_chunk = {stderr_chunk!r}\n"
        f"repeats = {repeats}\n"
        "def emit(fd, chunk, end):\n"
        "    for _ in range(repeats):\n"
        "        os.write(fd, chunk)\n"
        "    os.write(fd, end)\n"
        "threads = [\n"
        f"    Thread(target=emit, args=(1, stdout_chunk, {stdout_end!r})),\n"
        f"    Thread(target=emit, args=(2, stderr_chunk, {stderr_end!r})),\n"
        "]\n"
        "[thread.start() for thread in threads]\n"
        "[thread.join() for thread in threads]\n"
    )
    stdout_path = tmp_path / "training.stdout.log"
    stderr_path = tmp_path / "training.stderr.log"
    stdout_mirror = io.BytesIO()
    stderr_mirror = io.BytesIO()
    expected_stdout = stdout_chunk * repeats + stdout_end
    expected_stderr = stderr_chunk * repeats + stderr_end

    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None
    output = ProcessOutputDrainer(
        stdout=proc.stdout,
        stderr=proc.stderr,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_mirror=stdout_mirror,
        stderr_mirror=stderr_mirror,
    )

    assert proc.wait(timeout=10) == 0
    result = output.finish()

    assert result.warning is None
    assert result.stdout_path == stdout_path.resolve()
    assert result.stderr_path == stderr_path.resolve()
    assert stdout_path.read_bytes() == expected_stdout
    assert stderr_path.read_bytes() == expected_stderr
    assert stdout_mirror.getvalue() == expected_stdout
    assert stderr_mirror.getvalue() == expected_stderr
    assert result.stderr_tail == expected_stderr[-DEFAULT_STDERR_TAIL_BYTES:]
    assert proc.stdout.closed and proc.stderr.closed


def test_launcher_boundary_captures_python_native_and_traceback_output(
    tmp_path,
) -> None:
    script = (
        "import os, sys\n"
        "print(f'isatty={sys.stdout.isatty()}')\n"
        "print(f'buffer={hasattr(sys.stdout, \"buffer\")}')\n"
        "print(f'fileno={sys.stdout.fileno()}')\n"
        "os.write(1, b'native stdout\\n')\n"
        "os.write(2, b'native stderr\\n')\n"
        "raise RuntimeError('training failed')\n"
    )
    stdout_path = tmp_path / "training.stdout.log"
    stderr_path = tmp_path / "training.stderr.log"

    proc = start_training_process(
        train_cmd=[sys.executable, "-c", script],
        env=os.environ.copy(),
        cwd=str(tmp_path),
        capture_output=True,
    )
    assert proc.stdout is not None and proc.stderr is not None
    output = ProcessOutputDrainer(
        stdout=proc.stdout,
        stderr=proc.stderr,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    assert proc.wait(timeout=10) != 0
    result = output.finish()

    assert result.warning is None
    stdout = stdout_path.read_bytes()
    assert b"isatty=False" in stdout
    assert b"buffer=True" in stdout
    assert b"fileno=1" in stdout
    assert b"native stdout\n" in stdout
    stderr = stderr_path.read_bytes()
    assert b"native stderr\n" in stderr
    assert b"Traceback (most recent call last)" in stderr
    assert b"RuntimeError: training failed" in stderr


def test_training_process_only_pipes_output_when_requested(
    monkeypatch,
) -> None:
    popen = Mock()
    monkeypatch.setattr(subprocess, "Popen", popen)

    start_training_process(
        train_cmd=["python", "train.py"],
        env={},
        cwd=".",
        capture_output=False,
    )
    assert "stdout" not in popen.call_args.kwargs
    assert "stderr" not in popen.call_args.kwargs

    start_training_process(
        train_cmd=["python", "train.py"],
        env={},
        cwd=".",
        capture_output=True,
    )
    assert popen.call_args.kwargs["stdout"] is subprocess.PIPE
    assert popen.call_args.kwargs["stderr"] is subprocess.PIPE


def test_aggregator_process_pipes_only_stderr(monkeypatch) -> None:
    popen = Mock()
    monkeypatch.setattr(subprocess, "Popen", popen)

    start_aggregator_process(env={}, cwd=".")

    assert popen.call_args.kwargs["stderr"] is subprocess.PIPE
    assert "stdout" not in popen.call_args.kwargs


def test_sink_failure_falls_back_and_keeps_draining(
    monkeypatch, tmp_path
) -> None:
    source = _ChunkStream(b"one", b"two", b"three")
    sink = _FailingSink()
    fallback = io.BytesIO()
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: sink)

    output = ProcessOutputDrainer(
        stderr=source,
        stderr_path=tmp_path / "training.stderr.log",
        stderr_fallback=fallback,
        max_stderr_tail_bytes=8,
    )
    result = output.finish()

    assert result.stderr_path is None
    assert result.stderr_tail == b"twothree"
    assert fallback.getvalue() == b"twothree"
    assert result.warning is not None
    assert result.warning.count("stderr output file write failed") == 1
    assert source.closed


def test_sink_open_failure_falls_back_and_is_reported(
    monkeypatch, tmp_path
) -> None:
    source = _ChunkStream(b"all output")
    fallback = io.BytesIO()

    def fail_open(*_args, **_kwargs):
        raise OSError("read-only directory")

    monkeypatch.setattr(Path, "open", fail_open)
    result = ProcessOutputDrainer(
        stdout=source,
        stdout_path=tmp_path / "training.stdout.log",
        stdout_fallback=fallback,
    ).finish()

    assert result.stdout_path is None
    assert fallback.getvalue() == b"all output"
    assert result.warning is not None
    assert result.warning.count("stdout output file could not be opened") == 1


def test_path_resolution_failure_falls_back_and_keeps_draining(
    monkeypatch, tmp_path
) -> None:
    source = _ChunkStream(b"all output")
    fallback = io.BytesIO()

    def fail_resolve(_path):
        raise OSError("path unavailable")

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    result = ProcessOutputDrainer(
        stdout=source,
        stdout_path=tmp_path / "training.stdout.log",
        stdout_fallback=fallback,
    ).finish()

    assert result.stdout_path is None
    assert fallback.getvalue() == b"all output"
    assert result.warning is not None
    assert result.warning.count("stdout output file could not be opened") == 1


def test_mirror_failure_does_not_invalidate_saved_output(tmp_path) -> None:
    source = _ChunkStream(b"one", b"two")
    mirror = _FailingSink()
    fallback = io.BytesIO()
    output_path = tmp_path / "training.stdout.log"

    result = ProcessOutputDrainer(
        stdout=source,
        stdout_path=output_path,
        stdout_mirror=mirror,
        stdout_fallback=fallback,
    ).finish()

    assert result.stdout_path == output_path.resolve()
    assert output_path.read_bytes() == b"onetwo"
    assert fallback.getvalue() == b""
    assert result.warning is not None
    assert result.warning.count("stdout output mirror failed") == 1


def test_finish_is_bounded_and_idempotent(tmp_path) -> None:
    source = _BlockingStream()
    output = ProcessOutputDrainer(
        stdout=source,
        stdout_path=tmp_path / "training.stdout.log",
    )

    started = time.monotonic()
    result = output.finish(timeout_sec=0.01)
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert result.stdout_path is None
    assert result.warning is not None
    assert "output drain timed out for stdout" in result.warning
    assert output.finish() is result

    def worker_is_alive() -> bool:
        return any(
            thread.name == "traceml-process-stdout"
            for thread in threading.enumerate()
        )

    deadline = time.monotonic() + 2.0
    while worker_is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not worker_is_alive()
