# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Atomicity of ``utils/atomic_io.py`` writes (issue #326).

``tests/runtime/test_launcher.py::test_run_manifest_write_and_update_merge_correctly``
only checked the merged JSON content, never a crash mid-write, so it pinned
no atomicity guarantee despite its name. These tests exercise the actual
mechanism (temp file + fsync + ``os.replace``): a failure during the
replace step must leave the target path exactly as it was and must not
strand a temp file, which is the deterministic, non-flaky half of "atomic"
(the write itself becomes visible to readers all at once, by construction
of ``os.replace``; a concurrent-reader race is not simulated here since
POSIX rename atomicity is not this library's claim to test).
"""

from __future__ import annotations

import json

import pytest

from traceml_ai.utils.atomic_io import write_json_atomic, write_text_atomic


def _tmp_siblings(directory) -> list[str]:
    return [p.name for p in directory.iterdir() if p.name.startswith(".")]


def test_write_json_atomic_creates_the_file_with_exact_content(tmp_path):
    target = tmp_path / "manifest.json"

    write_json_atomic(target, {"status": "starting"})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "status": "starting"
    }
    assert _tmp_siblings(tmp_path) == []


def test_write_json_atomic_overwrites_in_place(tmp_path):
    target = tmp_path / "manifest.json"
    write_json_atomic(target, {"status": "starting"})

    write_json_atomic(target, {"status": "completed"})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "status": "completed"
    }
    assert _tmp_siblings(tmp_path) == []


def test_write_json_atomic_leaves_existing_file_untouched_on_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "manifest.json"
    write_json_atomic(target, {"status": "starting"})

    monkeypatch.setattr(
        "traceml_ai.utils.atomic_io.os.replace",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        write_json_atomic(target, {"status": "completed"})

    # The failed write never touched the original: no torn/partial file,
    # and the old content is still exactly what a reader would see.
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "status": "starting"
    }
    # The temp file used for the failed attempt was cleaned up, not left
    # behind for a future collision or a stray-file leak.
    assert _tmp_siblings(tmp_path) == []


def test_write_json_atomic_leaves_no_file_on_failure_for_new_path(
    tmp_path, monkeypatch
):
    target = tmp_path / "manifest.json"

    monkeypatch.setattr(
        "traceml_ai.utils.atomic_io.os.replace",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        write_json_atomic(target, {"status": "starting"})

    assert not target.exists()
    assert _tmp_siblings(tmp_path) == []


def test_write_json_atomic_cleans_up_temp_file_when_fsync_fails(
    tmp_path, monkeypatch
):
    target = tmp_path / "manifest.json"
    write_json_atomic(target, {"status": "starting"})

    monkeypatch.setattr(
        "traceml_ai.utils.atomic_io.os.fsync",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        write_json_atomic(target, {"status": "completed"})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "status": "starting"
    }
    assert _tmp_siblings(tmp_path) == []


def test_write_text_atomic_matches_write_json_atomic_semantics(
    tmp_path, monkeypatch
):
    target = tmp_path / "final_summary.txt"
    write_text_atomic(target, "old report\n")

    monkeypatch.setattr(
        "traceml_ai.utils.atomic_io.os.replace",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        write_text_atomic(target, "new report\n")

    assert target.read_text(encoding="utf-8") == "old report\n"
    assert _tmp_siblings(tmp_path) == []
