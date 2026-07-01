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
from pathlib import Path

import pytest

pytest.importorskip("torch")

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
RUN_NAME = "smoke-test"
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

    cmd = [
        sys.executable,
        "-c",
        "from traceml_ai.launcher.cli import main; main()",
        "run",
        str(script_path),
        "--mode=summary",
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
