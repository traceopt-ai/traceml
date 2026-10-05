"""Small subprocess runner shared by automatic-integration smoke tests."""

import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _run(
    tmp_path: Path,
    script: Path,
    *,
    disabled: bool = False,
    run_name="hf-auto",
    command="run",
):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SRC), env.get("PYTHONPATH", "")) if part
    )
    command = [
        sys.executable,
        "-m",
        "traceml_ai.launcher.cli",
        command,
        "--mode",
        "summary",
        "--logs-dir",
        str(tmp_path / "logs"),
        "--run-name",
        run_name,
        "--aggregator-port",
        _port(),
        "--master-port",
        _port(),
        "--finalize-timeout-sec",
        "60",
    ]
    if disabled:
        command.append("--disable-traceml")
    command.append(str(script))
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
