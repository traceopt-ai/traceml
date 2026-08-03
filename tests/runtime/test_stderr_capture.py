# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import io
import os
import sys

from traceml_ai.launcher.commands import _stderr_capture_enabled
from traceml_ai.launcher.process import (
    start_stderr_tail_capture,
    start_training_process,
)


def test_stderr_capture_is_opt_in() -> None:
    args = argparse.Namespace(capture_stderr=False)

    assert not _stderr_capture_enabled(args, {})
    assert _stderr_capture_enabled(args, {"TRACEML_CAPTURE_STDERR": "1"})
    assert not _stderr_capture_enabled(args, {"TRACEML_CAPTURE_STDERR": "0"})
    assert _stderr_capture_enabled(argparse.Namespace(capture_stderr=True), {})


def test_training_process_inherits_stderr_when_capture_is_disabled(
    monkeypatch,
) -> None:
    kwargs = {}

    def fake_popen(*args, **received_kwargs):
        kwargs.update(received_kwargs)
        return object()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    start_training_process([sys.executable, "train.py"], {}, ".")

    assert "stderr" not in kwargs


def test_stderr_capture_tees_flood_and_keeps_bounded_tail(tmp_path) -> None:
    script = tmp_path / "flood_stderr.py"
    script.write_text(
        "import sys\n"
        "chunk = b'0123456789abcdef' * 64\n"
        "for _ in range(256):\n"
        "    sys.stderr.buffer.write(chunk)\n"
        "sys.stderr.buffer.write(b'END\\n')\n"
        "sys.stderr.buffer.flush()\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    chunk = b"0123456789abcdef" * 64
    expected = chunk * 256 + b"END\n"
    tee = io.BytesIO()

    proc = start_training_process(
        [sys.executable, str(script)],
        os.environ.copy(),
        str(tmp_path),
        capture_stderr=True,
    )
    capture = start_stderr_tail_capture(
        proc,
        max_bytes=4096,
        tee_stream=tee,
    )

    assert proc.wait(timeout=10) == 7
    output_path = capture.finish(tmp_path / "crash_stderr.log")

    assert output_path is not None
    assert tee.getvalue() == expected
    assert output_path.read_bytes() == expected[-4096:]
