# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end evidence contract when training dies under ``traceml run``.

The launcher keeps native Python and torchrun evidence plus its own raw
streams. It writes no crash file of its own and does not reclassify a
torchrun worker failure. These tests pin what a user actually gets after a
real native crash, a Python exception, and a clean run.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "Windows has no POSIX signal delivery: os.kill(pid, SIGSEGV) calls "
        "TerminateProcess, so no native crash or faulthandler dump occurs."
    ),
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
# A short settle budget keeps the crash case fast: the dead rank never sends
# its rank-finished marker, so the aggregator waits out this budget.
FINALIZE_TIMEOUT_SEC = 10.0
SUBPROCESS_TIMEOUT_SEC = 240
EXCEPTION_MESSAGE = "traceml crash test python failure"
STDERR_MARKER = "TRACEML_CRASH_TEST_STDERR_BEFORE_EXIT"
STDOUT_MARKER = "TRACEML_CRASH_TEST_STEPS_DONE"
EXCERPT_HEADER = "[TraceML] Training stderr excerpt:"

TRAIN_SCRIPT = f"""\
import os
import signal
import sys

import torch
from torch import nn

import traceml_ai as traceml


def main(outcome):
    traceml.init()
    model = nn.Linear(8, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(3):
        with traceml.trace_step(model):
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(4, 8)).sum().backward()
            optimizer.step()
    print({STDOUT_MARKER!r}, flush=True)
    print({STDERR_MARKER!r}, file=sys.stderr, flush=True)
    if outcome == "sigsegv":
        os.kill(os.getpid(), signal.SIGSEGV)
    elif outcome == "exception":
        raise RuntimeError({EXCEPTION_MESSAGE!r})


if __name__ == "__main__":
    main(sys.argv[1])
"""
CRASH_LINE = next(
    number
    for number, line in enumerate(TRAIN_SCRIPT.splitlines(), start=1)
    if "os.kill(" in line
)


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SRC_DIR), env.get("PYTHONPATH", "")) if part
    )
    return env


@pytest.fixture(scope="module", autouse=True)
def _warm_import_caches(tmp_path_factory) -> None:
    # A cold checkout compiles bytecode on first import, which can push the
    # aggregator past the launcher's fixed readiness window.
    subprocess.run(
        [sys.executable, "-c", "import traceml_ai.aggregator.aggregator_main"],
        cwd=str(tmp_path_factory.mktemp("warm")),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SEC,
    )


def _run_traceml(tmp_path: Path, outcome: str):
    script_path = tmp_path / "crash_train.py"
    script_path.write_text(TRAIN_SCRIPT, encoding="utf-8")
    logs_dir = tmp_path / "logs"
    run_name = f"crash-{outcome}"

    aggregator_port = _free_tcp_port()
    master_port = _free_tcp_port()
    while master_port == aggregator_port:
        master_port = _free_tcp_port()

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from traceml_ai.launcher.cli import main; main()",
            "run",
            str(script_path),
            "--mode",
            "cli",
            "--run-name",
            run_name,
            "--logs-dir",
            str(logs_dir),
            "--aggregator-port",
            str(aggregator_port),
            "--master-port",
            str(master_port),
            "--finalize-timeout-sec",
            str(FINALIZE_TIMEOUT_SEC),
            "--args",
            outcome,
        ],
        cwd=str(tmp_path),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SEC,
    )
    session_root = (logs_dir / run_name).resolve()
    manifest = json.loads(
        (session_root / "manifest.json").read_text(encoding="utf-8")
    )
    return result, script_path.resolve(), session_root, manifest


def _describe(result, session_root: Path) -> str:
    return (
        f"exit={result.returncode}\nArtifacts: {session_root}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def _assert_raw_streams_saved(
    result, session_root: Path, manifest: dict
) -> str:
    """Assert both training streams are saved, recorded, and announced."""
    node_dir = session_root / "nodes" / "node_0"
    stdout_log = node_dir / "training.stdout.log"
    stderr_log = node_dir / "training.stderr.log"
    artifacts = manifest["artifacts"]

    assert artifacts["training_stderr_log"] == str(stderr_log)
    assert artifacts["training_stdout_log"] == str(stdout_log)
    assert Path(artifacts["aggregator_stderr_log"]).is_file()
    assert f"[TraceML] Stderr: {stderr_log}" in result.stderr
    assert f"[TraceML] Stdout: {stdout_log}" in result.stderr
    assert STDOUT_MARKER in stdout_log.read_text(encoding="utf-8")

    saved_stderr = stderr_log.read_text(encoding="utf-8", errors="replace")
    assert STDERR_MARKER in saved_stderr
    return saved_stderr


def _assert_current_final_summary(session_root: Path, manifest: dict) -> None:
    summary_path = session_root / "final_summary.json"
    assert summary_path.is_file()
    assert (session_root / "final_summary.txt").is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert datetime.fromisoformat(payload["generated_at"]) >= (
        datetime.fromisoformat(manifest["lifecycle"]["training_ended_at"])
    )


def _assert_failure_excerpt(result) -> str:
    """Assert cli mode printed the bounded stderr excerpt after failure."""
    lines = result.stderr.splitlines()
    assert EXCERPT_HEADER in lines
    start = lines.index(EXCERPT_HEADER) + 1
    end = next(
        index
        for index in range(start, len(lines))
        if lines[index].startswith("[TraceML] Stderr: ")
    )
    return "\n".join(lines[start:end])


def _assert_torchrun_failure_exit(result) -> None:
    """Worker failures are torchrun's exit code, never reclassified."""
    assert result.returncode != 0
    assert "Training terminated by" not in result.stderr
    assert result.stderr.splitlines()[-1] == (
        "[TraceML] Training failed — torchrun exited with code "
        f"{result.returncode}."
    )


def test_native_sigsegv_keeps_native_evidence_and_finalizes(tmp_path):
    result, script_path, session_root, manifest = _run_traceml(
        tmp_path, "sigsegv"
    )
    details = _describe(result, session_root)

    _assert_torchrun_failure_exit(result)
    assert manifest["status"] == "failed", details
    assert manifest["lifecycle"]["training_ended_at"] is not None

    saved_stderr = _assert_raw_streams_saved(result, session_root, manifest)
    # torchrun's error recorder enables faulthandler in the worker, so the
    # dump names the crashing frame. It is a frame dump, not an exception.
    assert "Fatal Python error: Segmentation fault" in saved_stderr, details
    assert f'File "{script_path}", line {CRASH_LINE} in main' in saved_stderr
    assert "Signal 11 (SIGSEGV)" in saved_stderr, details
    assert EXCEPTION_MESSAGE not in saved_stderr

    excerpt = _assert_failure_excerpt(result)
    assert "SIGSEGV" in excerpt, details

    # The dead rank never reports finished, so the aggregator finalizes on
    # its settle deadline and records a warning instead of hanging.
    assert manifest["telemetry_status"] == "degraded", details
    assert manifest["telemetry_reason"] == "finalization_warning"
    assert manifest["aggregator_exit_code"] == 0
    warning = json.loads(
        (session_root / "aggregator" / "finalization_warning.json").read_text(
            encoding="utf-8"
        )
    )
    assert warning["missing_ranks"] == [0]
    assert (
        "[TraceML] Telemetry degraded: finalization completed with warnings."
        in result.stderr
    )
    _assert_current_final_summary(session_root, manifest)


def test_python_exception_keeps_traceback_and_completes_telemetry(tmp_path):
    result, _, session_root, manifest = _run_traceml(tmp_path, "exception")
    details = _describe(result, session_root)

    _assert_torchrun_failure_exit(result)
    assert manifest["status"] == "failed", details

    saved_stderr = _assert_raw_streams_saved(result, session_root, manifest)
    assert "Traceback (most recent call last):" in saved_stderr
    assert f"RuntimeError: {EXCEPTION_MESSAGE}" in saved_stderr, details
    assert "Fatal Python error" not in saved_stderr

    excerpt = _assert_failure_excerpt(result)
    assert f"RuntimeError: {EXCEPTION_MESSAGE}" in excerpt, details

    # The executor's finally block still stops the runtime, so the rank
    # reports finished and telemetry closes cleanly.
    assert manifest["telemetry_status"] == "complete", details
    assert "telemetry_reason" not in manifest
    assert not (
        session_root / "aggregator" / "finalization_warning.json"
    ).exists()
    assert "[TraceML] Telemetry complete." in result.stderr
    _assert_current_final_summary(session_root, manifest)


def test_clean_run_has_no_failure_markers(tmp_path):
    result, _, session_root, manifest = _run_traceml(tmp_path, "clean")
    details = _describe(result, session_root)

    assert result.returncode == 0, details
    assert result.stderr.splitlines()[-1] == (
        "[TraceML] Training completed successfully (exit code 0)."
    )
    assert manifest["status"] == "completed"

    saved_stderr = _assert_raw_streams_saved(result, session_root, manifest)
    assert "Fatal Python error" not in saved_stderr
    assert "Traceback (most recent call last):" not in saved_stderr
    assert EXCERPT_HEADER not in result.stderr

    assert manifest["telemetry_status"] == "complete", details
    assert "telemetry_reason" not in manifest
    assert not (
        session_root / "aggregator" / "finalization_warning.json"
    ).exists()
    assert not (
        session_root / "aggregator" / "finalization_error.json"
    ).exists()
    _assert_current_final_summary(session_root, manifest)
