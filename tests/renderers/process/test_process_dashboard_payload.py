# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""What the Process compute layer produces for the dashboard.

Characterization tests, written against version_0.3.7 behavior and kept
passing unchanged while the layer is split into repository / models /
compute. Anything that changes here is a behavior change.
"""

from __future__ import annotations

import pytest

from tests.renderers.process.conftest import GB
from traceml_ai.renderers.process.dashboard_compute import (
    ProcessDashboardComputer,
)


def payload(db, **kwargs):
    return ProcessDashboardComputer(db_path=db.path, **kwargs).compute()


# --- the history ---------------------------------------------------------
def test_one_entry_per_committed_seq(process_db):
    for seq in (1, 2, 3):
        process_db.sample(seq=seq, rank=0, cpu=float(seq * 10))
    out = payload(process_db)
    assert [e["seq"] for e in out["history"]] == [1, 2, 3]
    assert [e["cpu_max"] for e in out["history"]] == [10.0, 20.0, 30.0]


def test_history_stops_at_the_slowest_rank(process_db):
    """Rank 1 is two steps behind, so the block shows only what both did."""
    for seq in (1, 2, 3):
        process_db.sample(seq=seq, rank=0)
    process_db.sample(seq=1, rank=1)
    out = payload(process_db)
    assert [e["seq"] for e in out["history"]] == [1]


def test_history_is_empty_when_nothing_has_been_written(process_db):
    out = payload(process_db)
    assert out["history"] == []
    assert out["gpu_used_imbalance"] is None
    assert out["series"] == {
        "x_time": [],
        "cpu_max": [],
        "ram_used_max": [],
        "gpu_used": [],
    }


def test_a_second_call_appends_only_the_new_seqs(process_db):
    computer = ProcessDashboardComputer(db_path=process_db.path)
    process_db.sample(seq=1, rank=0)
    first = computer.compute()
    process_db.sample(seq=2, rank=0)
    second = computer.compute()
    assert [e["seq"] for e in first["history"]] == [1]
    assert [e["seq"] for e in second["history"]] == [1, 2]


def test_history_is_capped_and_keeps_the_newest(process_db):
    for seq in range(1, 8):
        process_db.sample(seq=seq, rank=0)
    out = payload(process_db, dashboard_max_rows=3)
    assert [e["seq"] for e in out["history"]] == [5, 6, 7]


# --- GPU presence --------------------------------------------------------
def test_gpu_keys_are_absent_on_a_cpu_only_run(process_db):
    process_db.sample(seq=1, rank=0)
    entry = payload(process_db)["history"][0]
    for key in ("gpu_used", "gpu_total", "gpu_headroom", "gpu_rank"):
        assert key not in entry


def test_gpu_keys_are_present_once_a_rank_reports_a_gpu(process_db):
    process_db.sample(
        seq=1,
        rank=0,
        gpu_used=4.0 * GB,
        gpu_reserved=6.0 * GB,
        gpu_total=16.0 * GB,
    )
    entry = payload(process_db)["history"][0]
    assert entry["gpu_used"] == pytest.approx(4.0 * GB)
    assert entry["gpu_total"] == pytest.approx(16.0 * GB)
    assert entry["gpu_headroom"] == pytest.approx(10.0 * GB)
    assert entry["gpu_rank"] == 0


def test_top_level_imbalance_mirrors_the_newest_entry(process_db):
    process_db.sample(
        seq=1,
        rank=0,
        gpu_used=2.0 * GB,
        gpu_reserved=3.0 * GB,
        gpu_total=16.0 * GB,
    )
    process_db.sample(
        seq=1,
        rank=1,
        gpu_used=6.0 * GB,
        gpu_reserved=7.0 * GB,
        gpu_total=16.0 * GB,
    )
    out = payload(process_db)
    assert out["gpu_used_imbalance"] == pytest.approx(4.0 * GB)
    assert out["history"][-1]["gpu_used_imbalance"] == pytest.approx(4.0 * GB)


def test_single_gpu_rank_has_zero_imbalance_not_none(process_db):
    process_db.sample(
        seq=1,
        rank=0,
        gpu_used=4.0 * GB,
        gpu_reserved=6.0 * GB,
        gpu_total=16.0 * GB,
    )
    assert payload(process_db)["gpu_used_imbalance"] == pytest.approx(0.0)


# --- the series ----------------------------------------------------------
def test_series_track_the_history(process_db):
    process_db.sample(seq=1, rank=0, cpu=10.0, ram=1.0 * GB)
    process_db.sample(seq=2, rank=0, cpu=20.0, ram=2.0 * GB)
    series = payload(process_db)["series"]
    assert series["cpu_max"] == [10.0, 20.0]
    assert series["ram_used_max"] == [1.0 * GB, 2.0 * GB]
    assert series["gpu_used"] == []
    assert len(series["x_time"]) == 2
    assert series["x_time"][0].startswith("2023-")


def test_a_non_positive_timestamp_renders_as_an_empty_label(process_db):
    process_db.sample(seq=1, rank=0, ts=0.0)
    assert payload(process_db)["series"]["x_time"] == [""]


# --- degraded reads ------------------------------------------------------
def test_a_failed_read_reuses_the_last_good_payload(process_db, monkeypatch):
    computer = ProcessDashboardComputer(db_path=process_db.path)
    process_db.sample(seq=1, rank=0, cpu=33.0)
    good = computer.compute()
    assert good["history"]

    def boom(*_a, **_k):
        raise sqlite_error()

    def sqlite_error():
        return RuntimeError("database is locked")

    monkeypatch.setattr(computer._db, "connect", boom)
    assert computer.compute() == good


def test_the_last_good_payload_expires(process_db, monkeypatch):
    computer = ProcessDashboardComputer(
        db_path=process_db.path, stale_ttl_s=0.0
    )
    process_db.sample(seq=1, rank=0)
    assert computer.compute()["history"]

    monkeypatch.setattr(
        computer._db,
        "connect",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("gone")),
    )
    after = computer.compute()
    assert after["history"] == []
    assert after["gpu_used_imbalance"] is None


def test_a_failure_before_any_good_read_returns_an_empty_payload(
    process_db, monkeypatch
):
    computer = ProcessDashboardComputer(db_path=process_db.path)
    monkeypatch.setattr(
        computer._db,
        "connect",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("gone")),
    )
    out = computer.compute()
    assert out["history"] == []
    assert out["series"]["cpu_max"] == []
