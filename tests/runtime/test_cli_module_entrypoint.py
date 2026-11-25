# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""``python -m traceml_ai.launcher.cli`` must run the CLI.

Regression test for the missing ``__main__`` guard, which made the module
form import and exit 0 without running anything, silently swallowing the
user's command.
"""

import json
import os
import signal
import struct
import subprocess
import sys
from pathlib import Path

import pytest

import traceml_ai
from traceml_ai.utils.msgpack_codec import encode as encode_msgpack

_SRC_ROOT = str(Path(traceml_ai.__file__).resolve().parents[1])


def _run_module(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC_ROOT
    return subprocess.run(
        [sys.executable, "-m", "traceml_ai.launcher.cli", *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def test_module_form_prints_usage():
    result = _run_module("--help")

    assert result.returncode == 0, result.stderr
    assert "usage: " in result.stdout
    for command in ("watch", "run", "compare", "view"):
        assert command in result.stdout


def test_module_form_reports_an_unknown_command():
    result = _run_module("definitely-not-a-command")

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def _summary(path: Path) -> str:
    payload = {
        "schema_version": 2.0,
        "text": "TraceML Run Summary\n" * 4000,
        "system": {},
        "process": {},
        "step_time": {},
        "step_memory": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _msgpack_log(path: Path) -> str:
    with open(path, "wb") as f:
        for step in range(2000):
            frame = encode_msgpack({"step": step, "loss": 0.5})
            f.write(struct.pack("!I", len(frame)))
            f.write(frame)
    return str(path)


def _command_args(command: str, tmp_path: Path) -> list[str]:
    if command == "view":
        return ["view", _summary(tmp_path / "final_summary.json")]
    if command == "inspect":
        return ["inspect", _msgpack_log(tmp_path / "StepTimeTable.msgpack")]
    return [
        "compare",
        _summary(tmp_path / "run_a.json"),
        _summary(tmp_path / "run_b.json"),
        "--output",
        str(tmp_path / "compare" / "a_vs_b"),
    ]


@pytest.mark.skipif(
    not hasattr(signal, "SIGPIPE"), reason="needs POSIX SIGPIPE"
)
@pytest.mark.parametrize("command", ["view", "inspect", "compare"])
def test_closed_stdout_pipe_stops_quietly(command, tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC_ROOT
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "traceml_ai.launcher.cli",
            *_command_args(command, tmp_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env=env,
    )
    proc.stdout.close()
    stderr = proc.stderr.read().decode("utf-8", "replace")
    proc.stderr.close()
    returncode = proc.wait(timeout=120)

    assert "Traceback" not in stderr
    assert "Broken pipe" not in stderr
    assert returncode == -signal.SIGPIPE
