# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Real-runtime Hugging Face conformance gate for issue #320.

The ordinary integration fixture deliberately replaces runtime startup so its
fast framework tests can inspect in-process events. This test crosses the
actual product boundaries in a subprocess instead:

Trainer -> sampler -> exporter -> TCP -> aggregator -> SQLite -> summary.

It is intentionally CPU-only in CI. CUDA event correctness is release-tested
with ``src/dev/repro/hf_accelerate_h2d_window.py`` on a GPU.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
_WORKLOAD = _SRC / "dev/repro/hf_accelerate_h2d_window.py"
_RUN_NAME = "hf-real-runtime-cpu"
_EXPECTED_STEPS = 3
_TIMEOUT_SECONDS = 240

_REQUIRED_STREAMS = {
    "_traceml_internal:step_time",
    "_traceml_internal:forward_time",
    "_traceml_internal:backward_time",
    "_traceml_internal:optimizer_step",
    "_traceml_internal:dataloader_next",
}


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _persisted_step_events(db_path: Path) -> tuple[list[int], set[str]]:
    steps: list[int] = []
    names: set[str] = set()
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("""
            SELECT step, events_json
            FROM step_time_samples
            ORDER BY step, id
            """).fetchall()

    for step, events_json in rows:
        steps.append(int(step))
        names.update(json.loads(events_json))
    return steps, names


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="The launcher real-runtime smoke is currently POSIX-only.",
)
def test_hf_cpu_reaches_real_runtime_sqlite_and_summary(
    tmp_path: Path,
) -> None:
    logs_dir = tmp_path / "logs"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(_SRC), env.get("PYTHONPATH", "")) if part
    )

    command = [
        sys.executable,
        "-m",
        "traceml_ai.launcher.cli",
        "run",
        "--mode",
        "summary",
        "--logs-dir",
        str(logs_dir),
        "--run-name",
        _RUN_NAME,
        "--aggregator-port",
        str(_free_tcp_port()),
        "--master-port",
        str(_free_tcp_port()),
        "--finalize-timeout-sec",
        "60",
        str(_WORKLOAD),
        "--args",
        "--workload",
    ]
    result = subprocess.run(
        command,
        check=False,
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, (
        f"real-runtime HF validation exited with {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    session_root = logs_dir / _RUN_NAME
    summary_path = session_root / "final_summary.json"
    summary_text_path = session_root / "final_summary.txt"
    db_path = session_root / "aggregator/telemetry"
    manifest_path = session_root / "manifest.json"
    request_path = session_root / "control/final_summary_request.json"
    response_path = session_root / "control/final_summary_response.json"

    for artifact in (
        summary_path,
        summary_text_path,
        db_path,
        manifest_path,
        request_path,
        response_path,
    ):
        assert artifact.is_file(), f"missing runtime artifact: {artifact}"

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    step_time = payload["step_time"]
    metadata = step_time["metadata"]
    window = step_time["global"]["window"]
    average = step_time["global"]["average"]

    assert metadata["training_total_steps"] == _EXPECTED_STEPS
    assert metadata["training_latest_step"] == _EXPECTED_STEPS
    assert metadata["global_ranks_seen"] == 1
    assert metadata["global_ranks_used"] == 1
    assert window["steps_analyzed"] == _EXPECTED_STEPS
    assert window["diagnosis_clock"] == "cpu"

    for metric in (
        "input_wait_ms",
        "step_time_ms",
        "traced_step_time_ms",
        "dataloader_fetch_cpu_ms",
        "compute_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
    ):
        assert average[metric] is not None, f"missing CPU metric: {metric}"
        assert average[metric] > 0.0, f"non-positive CPU metric: {metric}"

    assert average["h2d_ms"] is None
    assert average["step_time_gpu_ms"] is None
    assert average["traced_step_time_gpu_ms"] is None

    steps, persisted_names = _persisted_step_events(db_path)
    assert steps == list(range(1, _EXPECTED_STEPS + 1))
    assert _REQUIRED_STREAMS <= persisted_names
    assert "_traceml_internal:h2d_time" not in persisted_names

    request = json.loads(request_path.read_text(encoding="utf-8"))
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["request_id"] == request["request_id"]
    assert response["status"] == "ok"
    assert Path(response["summary_json_path"]).resolve() == (
        summary_path.resolve()
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    # Covers degraded telemetry outside the step-time streams checked above.
    assert manifest["telemetry_status"] == "complete"
