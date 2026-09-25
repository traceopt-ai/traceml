# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

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

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
DDP_SCRIPT = REPO_ROOT / "examples" / "distributed" / "ddp_minimal.py"
RUN_NAME = "smoke-test"
DDP_RUN_NAME = "ddp-smoke-test"
FINALIZE_TIMEOUT_SEC = 60.0
SUBPROCESS_TIMEOUT_SEC = 240

TRAIN_SCRIPT = """\
import torch
from torch import nn

import traceml_ai as traceml


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    traceml.init()

    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(80):
        x = torch.randn(16, 32)
        y = torch.randint(0, 4, (16,))
        with traceml.trace_step(model):
            optimizer.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            optimizer.step()

    traceml.summary(print_text=False)


if __name__ == "__main__":
    main()
"""


def _free_tcp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Aggregator uses socket.SO_REUSEPORT, unavailable on Windows.",
)
def test_final_summary_json_smoke(tmp_path):
    script_path = tmp_path / "smoke_train.py"
    script_path.write_text(TRAIN_SCRIPT, encoding="utf-8")

    logs_dir = tmp_path / "logs"

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SRC_DIR), env.get("PYTHONPATH", "")) if part
    )

    # Warm bytecode/import caches so the aggregator binds within the launcher's
    # fixed 15s readiness window. On a cold checkout the first import compiles
    # .pyc for the whole stack, which can otherwise blow the startup budget.
    subprocess.run(
        [sys.executable, "-c", "import traceml_ai.aggregator.aggregator_main"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SEC,
    )

    cmd = [
        sys.executable,
        "-c",
        "from traceml_ai.launcher.cli import main; main()",
        "run",
        str(script_path),
        "--run-name",
        RUN_NAME,
        "--logs-dir",
        str(logs_dir),
        "--aggregator-port",
        str(_free_tcp_port()),
        "--finalize-timeout-sec",
        str(FINALIZE_TIMEOUT_SEC),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SEC,
    )

    assert result.returncode == 0, (
        f"traceml run exited with {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    session_root = logs_dir / RUN_NAME
    assert (session_root / "final_summary.json").is_file()
    assert (session_root / "final_summary.txt").is_file()

    payload = json.loads(
        (session_root / "final_summary.json").read_text(encoding="utf-8")
    )
    required = (
        "schema_version",
        "system",
        "process",
        "step_time",
        "step_memory",
    )
    for key in required:
        assert key in payload, f"final_summary.json missing key: {key!r}"

    average = payload["step_time"]["global"]["average"]
    for metric in (
        "traced_step_time_ms",
        "compute_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
    ):
        assert average[metric] is not None, f"missing CPU metric: {metric}"
        assert average[metric] > 0.0, f"non-positive CPU metric: {metric}"

    manifest = json.loads(
        (session_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["telemetry_status"] == "complete"
    training_ended_at = manifest["lifecycle"]["training_ended_at"]
    assert payload["duration_s"] > 0.0
    assert datetime.fromisoformat(payload["generated_at"]) >= (
        datetime.fromisoformat(training_ended_at)
    )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="End-to-end torchrun smoke run not yet verified on Windows.",
)
def test_two_rank_ddp_final_summary_smoke(tmp_path):
    from traceml_ai.reporting.final import SCHEMA_VERSION

    logs_dir = tmp_path / "logs"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SRC_DIR), env.get("PYTHONPATH", "")) if part
    )

    aggregator_port = _free_tcp_port()
    master_port = _free_tcp_port()
    while master_port == aggregator_port:
        master_port = _free_tcp_port()

    cmd = [
        sys.executable,
        "-c",
        "from traceml_ai.launcher.cli import main; main()",
        "run",
        str(DDP_SCRIPT),
        "--mode",
        "summary",
        "--run-name",
        DDP_RUN_NAME,
        "--logs-dir",
        str(logs_dir),
        "--nproc-per-node",
        "2",
        "--master-port",
        str(master_port),
        "--aggregator-port",
        str(aggregator_port),
        "--finalize-timeout-sec",
        str(FINALIZE_TIMEOUT_SEC),
        "--args",
        "--steps",
        "20",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SEC,
    )

    session_root = logs_dir / DDP_RUN_NAME
    assert result.returncode == 0, (
        f"two-rank traceml run exited with {result.returncode}\n"
        f"Artifacts: {session_root}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    summary_path = session_root / "final_summary.json"
    manifest_path = session_root / "manifest.json"
    assert summary_path.is_file()
    assert manifest_path.is_file()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION

    # Shared CPU runners are timing-variable. Pin the all-rank artifact
    # contract here; diagnosis and performance thresholds have separate tests.
    step_time = payload["step_time"]
    assert step_time["metadata"]["global_ranks_seen"] == 2
    assert step_time["metadata"]["global_ranks_used"] == 2
    assert set(step_time["groups"]["rows"]) == {"0", "1"}
    assert step_time["global"]["window"]["steps_analyzed"] == 20

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["telemetry_status"] == "complete"
    assert manifest["launch"]["nproc_per_node"] == 2
