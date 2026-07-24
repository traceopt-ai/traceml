"""Launch helpers used by local and distributed benchmark runners."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def benchmark_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def stable_cell_port(base_port: int, cell_index: int) -> int:
    """Return a fixed, non-overlapping port for one planned benchmark cell."""
    return int(base_port) + int(cell_index) * 2


def run_capture(
    cmd: list[str], *, cwd: Path, env: dict[str, str], timeout: float
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "wall_time_sec": time.perf_counter() - started,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "wall_time_sec": time.perf_counter() - started,
            "timed_out": True,
        }


def traceml_launcher_prefix(repo_root: Path) -> list[str]:
    del repo_root
    return [
        sys.executable,
        "-c",
        "from traceml_ai.launcher.cli import main; main()",
        "run",
    ]


def command_text(cmd: list[str]) -> str:
    return " ".join(cmd)
