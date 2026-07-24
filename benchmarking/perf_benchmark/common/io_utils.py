"""I/O helpers for benchmark artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def make_results_dir(output_root: Path, run_id: str, prefix: str) -> Path:
    resolved = run_id.strip() or datetime.now().strftime(
        f"{prefix}_%Y%m%d_%H%M%S"
    )
    results_dir = (output_root / resolved).resolve()
    results_dir.mkdir(parents=True, exist_ok=False)
    (results_dir / "runs").mkdir()
    (results_dir / "traceml-logs").mkdir()
    return results_dir
