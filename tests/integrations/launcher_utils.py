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
    run_name: str = "hf-auto",
    command: str = "run",
    nproc_per_node: int = 1,
) -> subprocess.CompletedProcess[str]:
    """Run a script through the TraceML launcher in an isolated test session.

    ``nproc_per_node`` selects the number of local CPU workers. The launcher
    validates the value and uses torchrun only when more than one worker is
    requested.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SRC), env.get("PYTHONPATH", "")) if part
    )
    argv = [
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
        "--nproc-per-node",
        str(nproc_per_node),
        "--finalize-timeout-sec",
        "60",
    ]
    if disabled:
        argv.append("--disable-traceml")
    argv.append(str(script))
    return subprocess.run(
        argv,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
