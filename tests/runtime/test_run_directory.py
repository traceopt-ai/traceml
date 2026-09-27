# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from traceml_ai.launcher.run_directory import (
    RunDirectoryError,
    claim_node_dir,
    join_run_root,
    reserve_run_root,
)


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): (
            path.read_bytes() if path.is_file() else b"<dir>"
        )
        for path in sorted(root.rglob("*"))
    }


def _write_manifest(run_root: Path, payload: dict) -> None:
    (run_root / "manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_reserve_creates_run_and_node_zero_folders(tmp_path) -> None:
    run_root = tmp_path / "logs" / "fresh"

    reserve_run_root(run_root)

    assert (run_root / "nodes" / "node_0").is_dir()


@pytest.mark.parametrize("with_files", [False, True])
def test_reserve_refuses_an_existing_directory(tmp_path, with_files) -> None:
    run_root = tmp_path / "logs" / "taken"
    run_root.mkdir(parents=True)
    if with_files:
        _write_manifest(run_root, {"status": "completed"})
    before = _tree(run_root)

    with pytest.raises(RunDirectoryError, match="already exists"):
        reserve_run_root(run_root)

    assert _tree(run_root) == before


def test_only_one_concurrent_reservation_wins(tmp_path) -> None:
    run_root = tmp_path / "logs" / "race"
    racers = 8
    barrier = threading.Barrier(racers)

    def reserve(_index: int) -> bool:
        barrier.wait()
        try:
            reserve_run_root(run_root)
        except RunDirectoryError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=racers) as pool:
        results = list(pool.map(reserve, range(racers)))

    assert results.count(True) == 1


def test_claim_never_creates_a_missing_run_root(tmp_path) -> None:
    run_root = tmp_path / "logs" / "missing"

    with pytest.raises(FileNotFoundError):
        claim_node_dir(run_root, 1)

    assert not run_root.exists()


def test_join_claims_this_node_in_a_reserved_run(tmp_path) -> None:
    run_root = tmp_path / "logs" / "shared"
    reserve_run_root(run_root)
    _write_manifest(run_root, {"status": "running", "run": {"launch_id": "a"}})

    join_run_root(run_root, 1)

    assert (run_root / "nodes" / "node_1").is_dir()


@pytest.mark.parametrize(
    ("manifest", "node_one_exists", "message"),
    [
        (None, False, "node 0 has not reserved"),
        ({"status": "running", "run": {}}, False, "earlier launch"),
        (
            {"status": "completed", "run": {"launch_id": "a"}},
            False,
            "earlier launch",
        ),
        (
            {"status": "running", "run": {"launch_id": "a"}},
            True,
            "earlier launch",
        ),
    ],
)
def test_join_refuses_a_run_not_reserved_by_this_launch(
    tmp_path, manifest, node_one_exists, message
) -> None:
    run_root = tmp_path / "logs" / "old"
    reserve_run_root(run_root)
    if manifest is not None:
        _write_manifest(run_root, manifest)
    if node_one_exists:
        (run_root / "nodes" / "node_1").mkdir()
    before = _tree(run_root)

    with pytest.raises(RunDirectoryError, match=message):
        join_run_root(run_root, 1)

    assert _tree(run_root) == before
