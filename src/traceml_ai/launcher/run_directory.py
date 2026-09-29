# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Run directory ownership: node 0 reserves it, other nodes join it."""

from __future__ import annotations

import json
from pathlib import Path

from traceml_ai.launcher.manifest import node_artifact_dir

_JOINABLE_STATUSES = ("starting", "running")


class RunDirectoryError(RuntimeError):
    """The run directory cannot be used by this launch."""


def _earlier_launch_error(run_root: Path) -> RunDirectoryError:
    return RunDirectoryError(
        f"run directory {run_root} belongs to an earlier launch and was left "
        "untouched. Choose a new --run-name on every node."
    )


def reserve_run_root(run_root: Path) -> None:
    """Atomically create the run directory and node 0's folder."""
    run_root = Path(run_root).resolve()
    run_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive mkdir, so only one launcher can win a given run name.
        run_root.mkdir()
    except FileExistsError:
        raise RunDirectoryError(
            f"run directory already exists: {run_root}. A run name identifies "
            "one execution, so the existing run was left untouched. Choose a "
            "new --run-name, or omit it to generate one."
        ) from None
    claim_node_dir(run_root, 0)


def claim_node_dir(run_root: Path, node_rank: int) -> None:
    """Create this node's folder inside an existing run directory."""
    run_root = Path(run_root).resolve()
    # No parents, so a missing run directory is never created here.
    (run_root / "nodes").mkdir(exist_ok=True)
    try:
        node_artifact_dir(run_root, node_rank).mkdir()
    except FileExistsError:
        raise _earlier_launch_error(run_root) from None


def join_run_root(run_root: Path, *, run_name: str, node_rank: int) -> None:
    """Join the active run directory node 0 reserved for this run name."""
    run_root = Path(run_root).resolve()
    try:
        with open(run_root / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        raise RunDirectoryError(
            f"node 0 has not reserved run directory {run_root}. Start node 0 "
            "with the same --run-name and --logs-dir, and put --logs-dir on "
            "storage that every node can see."
        ) from None

    run = manifest.get("run") if isinstance(manifest, dict) else None
    found_name = run.get("run_name") if isinstance(run, dict) else None
    if found_name != run_name:
        raise RunDirectoryError(
            f"run directory {run_root} belongs to run {found_name!r}, not "
            f"{run_name!r}, and was left untouched."
        )
    if manifest.get("status") not in _JOINABLE_STATUSES:
        raise _earlier_launch_error(run_root)
    claim_node_dir(run_root, node_rank)


__all__ = [
    "RunDirectoryError",
    "claim_node_dir",
    "join_run_root",
    "reserve_run_root",
]
