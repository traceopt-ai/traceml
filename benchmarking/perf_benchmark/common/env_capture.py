"""Environment capture for reproducible benchmark runs."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.io_utils import write_json


def _run_command(
    cmd: list[str], *, cwd: Path, env: dict[str, str]
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20.0,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        return {"error": repr(exc)}


def capture_environment(results_dir: Path, repo_root: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "argv": sys.argv,
    }

    commands = {
        "git_rev_parse_head": ["git", "rev-parse", "HEAD"],
        "git_status_short": ["git", "status", "--short"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
    }
    for name, cmd in commands.items():
        payload[name] = _run_command(cmd, cwd=repo_root, env=env)

    try:
        import torch

        payload["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "cudnn_version": torch.backends.cudnn.version(),
        }
        if torch.cuda.is_available():
            payload["torch"]["devices"] = [
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": torch.cuda.get_device_capability(idx),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(idx).total_memory
                    ),
                }
                for idx in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        payload["torch"] = {"error": repr(exc)}

    write_json(results_dir / "environment.json", payload)
    return payload
