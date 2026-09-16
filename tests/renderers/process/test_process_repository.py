# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""What the Process read layer returns from `process_samples`.

These are characterization tests: they were written against the behavior
shipping on version_0.3.7 and must keep passing unchanged while the code
underneath moves into its own repository module. A change here is a change
in behavior, not a change in structure.
"""

from __future__ import annotations

import sqlite3

import pytest

from tests.renderers.process.conftest import GB
from traceml_ai.renderers.process.repository import ProcessRepository
from traceml_ai.renderers.shared.run_series import RunSeriesPlan


@pytest.fixture
def repo(process_db):
    return ProcessRepository(db_path=process_db.path), process_db


class _FailingConnection:
    def execute(self, *_args, **_kwargs):
        raise sqlite3.OperationalError("query failed")


def test_latest_seq_is_the_newest_row_carrying_one(repo):
    repository, db = repo
    db.sample(seq=1, rank=0)
    db.sample(seq=4, rank=0)
    db.insert(rank=0, seq=None, cpu_percent=99.0)
    with repository.connect() as conn:
        assert repository.fetch_latest_seq(conn) == 4


def test_latest_seq_is_none_on_an_empty_table(repo):
    repository, _db = repo
    with repository.connect() as conn:
        assert repository.fetch_latest_seq(conn) is None


def test_latest_seq_per_rank_reports_each_rank_independently(repo):
    repository, db = repo
    db.sample(seq=1, rank=0)
    db.sample(seq=2, rank=0)
    db.sample(seq=1, rank=1)
    with repository.connect() as conn:
        assert repository.fetch_latest_seq_per_rank(conn) == {0: 2, 1: 1}


def test_committed_seq_is_the_slowest_rank(repo):
    """The whole block advances only as fast as its slowest rank.

    This is the rule that makes a dashboard entry mean "every rank finished
    this step", and it is why one lagging rank holds the history back.
    """
    repository, db = repo
    db.sample(seq=1, rank=0)
    db.sample(seq=5, rank=0)
    db.sample(seq=2, rank=1)
    with repository.connect() as conn:
        assert repository.fetch_committed_seq(conn) == 2


def test_committed_seq_is_none_when_no_rank_has_reported(repo):
    repository, _db = repo
    with repository.connect() as conn:
        assert repository.fetch_committed_seq(conn) is None


def test_seq_range_takes_the_max_across_ranks_for_cpu_and_ram(repo):
    repository, db = repo
    db.sample(seq=1, rank=0, cpu=10.0, ram=1.0 * GB, ram_total=16.0 * GB)
    db.sample(seq=1, rank=1, cpu=70.0, ram=3.0 * GB, ram_total=16.0 * GB)
    with repository.connect() as conn:
        rows = repository.fetch_seq_range_aggregates(
            conn, start_seq=1, end_seq=1
        )
    assert len(rows) == 1
    assert rows[0]["cpu_max"] == pytest.approx(70.0)
    assert rows[0]["ram_used_max"] == pytest.approx(3.0 * GB)
    assert rows[0]["ram_total"] == pytest.approx(16.0 * GB)


def test_seq_range_picks_the_gpu_rank_with_the_least_headroom(repo):
    """Not the largest user: the one closest to running out.

    Rank 1 uses less memory but has reserved more of a smaller card, so it
    is the rank at risk and the one the tile must describe.
    """
    repository, db = repo
    db.sample(
        seq=1,
        rank=0,
        gpu_used=8.0 * GB,
        gpu_reserved=9.0 * GB,
        gpu_total=40.0 * GB,
    )
    db.sample(
        seq=1,
        rank=1,
        gpu_used=5.0 * GB,
        gpu_reserved=15.0 * GB,
        gpu_total=16.0 * GB,
    )
    with repository.connect() as conn:
        rows = repository.fetch_seq_range_aggregates(
            conn, start_seq=1, end_seq=1
        )
    assert rows[0]["gpu_rank"] == 1
    assert rows[0]["gpu_used"] == pytest.approx(5.0 * GB)
    assert rows[0]["gpu_headroom"] == pytest.approx(1.0 * GB)


def test_seq_range_imbalance_is_the_used_spread_across_ranks(repo):
    repository, db = repo
    db.sample(
        seq=1,
        rank=0,
        gpu_used=2.0 * GB,
        gpu_reserved=3.0 * GB,
        gpu_total=16.0 * GB,
    )
    db.sample(
        seq=1,
        rank=1,
        gpu_used=9.0 * GB,
        gpu_reserved=10.0 * GB,
        gpu_total=16.0 * GB,
    )
    with repository.connect() as conn:
        rows = repository.fetch_seq_range_aggregates(
            conn, start_seq=1, end_seq=1
        )
    assert rows[0]["gpu_used_imbalance"] == pytest.approx(7.0 * GB)


def test_seq_range_leaves_gpu_columns_null_on_a_cpu_only_run(repo):
    repository, db = repo
    db.sample(seq=1, rank=0)
    with repository.connect() as conn:
        rows = repository.fetch_seq_range_aggregates(
            conn, start_seq=1, end_seq=1
        )
    assert rows[0]["gpu_used"] is None
    assert rows[0]["gpu_rank"] is None
    assert rows[0]["gpu_used_imbalance"] is None


def test_seq_range_skips_a_rank_missing_any_gpu_column(repo):
    """A partial GPU row cannot be ranked by headroom, so it is not a
    candidate. It still contributes to the imbalance spread, which only
    needs the used bytes."""
    repository, db = repo
    db.sample(
        seq=1,
        rank=0,
        gpu_used=2.0 * GB,
        gpu_reserved=None,
        gpu_total=16.0 * GB,
    )
    db.sample(
        seq=1,
        rank=1,
        gpu_used=9.0 * GB,
        gpu_reserved=10.0 * GB,
        gpu_total=16.0 * GB,
    )
    with repository.connect() as conn:
        rows = repository.fetch_seq_range_aggregates(
            conn, start_seq=1, end_seq=1
        )
    assert rows[0]["gpu_rank"] == 1
    assert rows[0]["gpu_used_imbalance"] == pytest.approx(7.0 * GB)


def test_seq_range_returns_one_row_per_seq_in_order(repo):
    repository, db = repo
    for seq in (3, 1, 2):
        db.sample(seq=seq, rank=0, cpu=float(seq))
    with repository.connect() as conn:
        rows = repository.fetch_seq_range_aggregates(
            conn, start_seq=1, end_seq=3
        )
    assert [int(r["seq"]) for r in rows] == [1, 2, 3]


def test_seq_range_is_empty_when_the_bounds_are_inverted(repo):
    repository, db = repo
    db.sample(seq=1, rank=0)
    with repository.connect() as conn:
        assert (
            repository.fetch_seq_range_aggregates(conn, start_seq=5, end_seq=1)
            == []
        )


def test_rows_for_one_seq_come_back_ordered_by_rank(repo):
    repository, db = repo
    db.sample(seq=1, rank=2)
    db.sample(seq=1, rank=0)
    db.sample(seq=1, rank=1)
    with repository.connect() as conn:
        rows = repository.fetch_rows_for_seq_all_ranks(conn, seq=1)
    assert [int(r["rank"]) for r in rows] == [0, 1, 2]


def test_connect_yields_named_row_access(repo):
    repository, db = repo
    db.sample(seq=1, rank=0, cpu=42.0)
    with repository.connect() as conn:
        row = conn.execute(
            "SELECT cpu_percent FROM process_samples"
        ).fetchone()
    assert isinstance(row, sqlite3.Row)
    assert row["cpu_percent"] == pytest.approx(42.0)


def test_run_stats_sql_errors_propagate(repo):
    repository, _db = repo
    with pytest.raises(sqlite3.OperationalError, match="query failed"):
        repository.cpu_capacity_run_stats(_FailingConnection())


def test_run_history_sql_errors_propagate(repo):
    repository, _db = repo
    plan = RunSeriesPlan(
        window_s=30.0,
        cadence_s=2.0,
        stride=1,
        max_points=120,
        sample_count=100,
    )
    with pytest.raises(sqlite3.OperationalError, match="query failed"):
        repository.fetch_cpu_capacity_run(_FailingConnection(), plan)
