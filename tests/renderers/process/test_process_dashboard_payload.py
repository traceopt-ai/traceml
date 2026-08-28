# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""What the Process compute layer produces for the dashboard.

Characterization tests. Every expected VALUE here was pinned against
version_0.3.7 before the layer was split; only the way the payload is
reached changed, from dictionary keys to typed fields. A changed number is
a changed behavior.
"""

from __future__ import annotations

import pytest

from tests.renderers.process.conftest import GB
from traceml_ai.renderers.process.dashboard_compute import (
    ProcessDashboardComputer,
    percentile,
)


def payload(db, **kwargs):
    return ProcessDashboardComputer(db_path=db.path, **kwargs).compute()


# --- the history ---------------------------------------------------------
def test_one_entry_per_committed_seq(process_db):
    for seq in (1, 2, 3):
        process_db.sample(seq=seq, rank=0, cpu=float(seq * 10))
    out = payload(process_db)
    assert [e.seq for e in out.history] == [1, 2, 3]
    assert [e.cpu_percent_max for e in out.history] == [10.0, 20.0, 30.0]


def test_history_stops_at_the_slowest_rank(process_db):
    """Rank 1 is two steps behind, so the block shows only what both did."""
    for seq in (1, 2, 3):
        process_db.sample(seq=seq, rank=0)
    process_db.sample(seq=1, rank=1)
    assert [e.seq for e in payload(process_db).history] == [1]


def test_history_is_empty_when_nothing_has_been_written(process_db):
    out = payload(process_db)
    assert out.history == ()
    assert out.has_data is False
    assert out.gpu_used_imbalance_bytes is None
    assert out.chart is None
    assert out.cpu is None


def test_a_second_call_appends_only_the_new_seqs(process_db):
    computer = ProcessDashboardComputer(db_path=process_db.path)
    process_db.sample(seq=1, rank=0)
    first = computer.compute()
    process_db.sample(seq=2, rank=0)
    second = computer.compute()
    assert [e.seq for e in first.history] == [1]
    assert [e.seq for e in second.history] == [1, 2]


def test_history_is_capped_and_keeps_the_newest(process_db):
    for seq in range(1, 8):
        process_db.sample(seq=seq, rank=0)
    out = payload(process_db, dashboard_max_rows=3)
    assert [e.seq for e in out.history] == [5, 6, 7]


def test_the_described_window_is_the_last_hundred_steps(process_db):
    for seq in range(1, 151):
        process_db.sample(seq=seq, rank=0)
    out = payload(process_db, dashboard_max_rows=200)
    assert out.window_len == 100
    assert out.history[-1].seq == 150


# --- GPU presence --------------------------------------------------------
def test_a_cpu_only_run_carries_no_gpu_block(process_db):
    process_db.sample(seq=1, rank=0)
    out = payload(process_db)
    assert out.history[0].gpu is None
    assert out.gpu_available is False
    assert out.gpu is None


def test_a_gpu_run_carries_the_least_headroom_rank(process_db):
    process_db.sample(
        seq=1,
        rank=0,
        gpu_used=4.0 * GB,
        gpu_reserved=6.0 * GB,
        gpu_total=16.0 * GB,
    )
    gpu = payload(process_db).history[0].gpu
    assert gpu is not None
    assert gpu.used_bytes == pytest.approx(4.0 * GB)
    assert gpu.total_bytes == pytest.approx(16.0 * GB)
    assert gpu.headroom_bytes == pytest.approx(10.0 * GB)
    assert gpu.rank == 0


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
    assert out.gpu_used_imbalance_bytes == pytest.approx(4.0 * GB)
    assert out.history[-1].gpu.used_imbalance_bytes == pytest.approx(4.0 * GB)


def test_single_gpu_rank_has_zero_imbalance_not_none(process_db):
    process_db.sample(
        seq=1,
        rank=0,
        gpu_used=4.0 * GB,
        gpu_reserved=6.0 * GB,
        gpu_total=16.0 * GB,
    )
    assert payload(process_db).gpu_used_imbalance_bytes == pytest.approx(0.0)


# --- the rollups the card states -----------------------------------------
@pytest.mark.parametrize(
    "values, p, expected",
    [
        ([], 50, 0.0),
        ([5.0], 95, 5.0),
        ([1.0, 2.0, 3.0, 4.0], 50, 2.5),
        ([1.0, 2.0, 3.0, 4.0], 0, 1.0),
        ([1.0, 2.0, 3.0, 4.0], 100, 4.0),
        ([3.0, 1.0, 2.0], 50, 2.0),
        ([1.0, None, 3.0], 50, 2.0),
    ],
)
def test_percentile(values, p, expected):
    assert percentile(values, p) == pytest.approx(expected)


def test_cpu_rollup_reports_the_newest_and_its_percentiles(process_db):
    for seq in range(1, 5):
        process_db.sample(seq=seq, rank=0, cpu=float(seq * 10))
    cpu = payload(process_db).cpu
    assert cpu.now == pytest.approx(40.0)
    assert cpu.p50 == pytest.approx(25.0)
    assert cpu.p95 == pytest.approx(38.5)


def test_ram_rollup_carries_its_denominator(process_db):
    process_db.sample(seq=1, rank=0, ram=3.0 * GB, ram_total=64.0 * GB)
    ram = payload(process_db).ram
    assert ram.now == pytest.approx(3.0 * GB)
    assert ram.total == pytest.approx(64.0 * GB)


def test_gpu_rollup_is_absent_rather_than_zero_without_a_gpu(process_db):
    process_db.sample(seq=1, rank=0)
    assert payload(process_db).gpu is None


# --- the chart -----------------------------------------------------------
def test_ram_trace_is_a_share_of_its_own_total(process_db):
    process_db.sample(seq=1, rank=0, ram=4.0 * GB, ram_total=16.0 * GB)
    chart = payload(process_db).chart
    assert chart.ram_percent.values == pytest.approx((25.0,))
    assert chart.gpu_percent is None


def test_gpu_trace_is_a_share_of_its_own_total(process_db):
    process_db.sample(
        seq=1,
        rank=0,
        gpu_used=4.0 * GB,
        gpu_reserved=5.0 * GB,
        gpu_total=16.0 * GB,
    )
    chart = payload(process_db).chart
    assert chart.gpu_percent.values == pytest.approx((25.0,))


def test_traces_carry_sample_times_not_formatted_labels(process_db):
    process_db.sample(seq=1, rank=0, ts=1_700_000_001.0)
    chart = payload(process_db).chart
    assert chart.ram_percent.timestamps == (1_700_000_001.0,)


# --- degraded reads ------------------------------------------------------
def test_a_failed_read_reuses_the_last_good_payload(process_db, monkeypatch):
    computer = ProcessDashboardComputer(db_path=process_db.path)
    process_db.sample(seq=1, rank=0, cpu=33.0)
    good = computer.compute()
    assert good.has_data

    monkeypatch.setattr(
        computer._db,
        "connect",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("locked")),
    )
    assert computer.compute() == good


def test_the_last_good_payload_expires(process_db, monkeypatch):
    computer = ProcessDashboardComputer(
        db_path=process_db.path, stale_ttl_s=0.0
    )
    process_db.sample(seq=1, rank=0)
    assert computer.compute().has_data

    monkeypatch.setattr(
        computer._db,
        "connect",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("gone")),
    )
    after = computer.compute()
    assert after.has_data is False
    assert after.gpu_used_imbalance_bytes is None


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
    assert out.has_data is False
    assert out.chart is None
