# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""What SystemRepository does today, pinned before part 5b moves it.

There were no SystemRepository tests at all before this file, while the
Process repository has had them since #410. Part 5b replaces this module's
hand-rolled window planning with `renderers/shared/run_series.py`, so the
current behaviour is described here first and the move is then provably a
move except where issue #419 declared a change.

Where a rule is odd or is about to change, the test says so rather than
asserting the nicer rule the code does not yet implement. Three are
deliberately pinned as they are, because 5b changes them and the update
should be a stated decision rather than a silent one:

* the rolling frame counts ROWS, not seconds, so it spans more wall clock
  than it claims across a gap,
* the retained-vs-recent gate is a bare `span > min_span_s`, with the 1.2x
  hysteresis held by the caller in a constant that duplicates the shared
  policy's,
* a failed read returns an empty result rather than raising, which turns a
  database failure into a confident "no data".
"""

from __future__ import annotations

import sqlite3

import pytest

from tests.renderers.system.conftest import gpu
from traceml_ai.renderers.shared.run_series import (
    DEFAULT_RUN_SERIES_POLICY,
    plan_run_series,
)
from traceml_ai.renderers.system.repository import SystemRepository


def _repo(path: str) -> SystemRepository:
    return SystemRepository(db_path=path)


def _cpu_run(repo, conn, hostname=None):
    """Plan and read one whole-run CPU series, as the computer does."""
    stats = repo.cpu_run_stats(conn, hostname=hostname)
    if stats is None:
        return None, []
    plan = plan_run_series(
        span_s=stats.span_s,
        sample_count=stats.sample_count,
        policy=DEFAULT_RUN_SERIES_POLICY,
    )
    if plan is None:
        return stats, []
    return stats, repo.fetch_cpu_run(conn, plan, hostname=hostname)


def _power_run(repo, conn, hostname=None):
    """Plan and read one whole-run power series, as the computer does."""
    stats = repo.gpu_power_run_stats(conn, hostname=hostname)
    if stats is None:
        return None, []
    width = DEFAULT_RUN_SERIES_POLICY.bucket_width_for(stats.span_s)
    rows = repo.fetch_gpu_power_run(
        conn,
        width_s=width,
        first_ts=stats.first_ts,
        hostname=hostname,
    )
    per_gpu = {}
    for idx, ts, avg, low, _high in rows:
        entry = per_gpu.setdefault(idx, {"t": [], "avg": [], "min": []})
        entry["t"].append(ts)
        entry["avg"].append(avg)
        entry["min"].append(low)
    return stats, [per_gpu[i] for i in sorted(per_gpu)]


# --- the window ladder ---------------------------------------------------
def test_the_shared_ladder_is_the_ladder_system_had():
    """The five rungs that used to live in `common.choose_window_s`.

    Kept verbatim through the move, because this is the assertion that
    makes the consolidation a consolidation: if the shared policy chose
    different windows, every whole-run chart would change span and the
    label a card prints would change with it.
    """
    from traceml_ai.renderers.shared.run_series import (
        DEFAULT_RUN_SERIES_POLICY as policy,
    )

    assert policy.window_for(4 * 60) == 30.0  # the floor
    assert policy.window_for(23 * 60) == 30.0
    assert policy.window_for(96 * 60) == 120.0
    assert policy.window_for(178 * 60) == 300.0
    assert policy.window_for(48 * 3600) == 300.0  # the cap


def test_a_short_run_is_kept_off_the_expensive_query_by_the_mode(system_db):
    """The gate moved UP, out of the read and into the decision.

    The repository used to take `min_span_s` and return an empty result,
    which meant a short run still paid for two queries to be told no. The
    computer now asks `mode_for` first and only reads when the answer is
    `retained`, so the read is not reached at all.
    """
    path = system_db(ticks=10, cadence_s=2.0)  # 18 s span
    repo = _repo(path)
    with repo.connect() as conn:
        stats = repo.cpu_run_stats(conn)
    assert stats is not None
    assert stats.span_s == pytest.approx(18.0)
    assert DEFAULT_RUN_SERIES_POLICY.mode_for(stats.span_s, 60.0) == "recent"


def test_the_hysteresis_factor_now_has_one_owner(system_db):
    """The repository compares; the hysteresis lives above it.

    Corrected after reading the code rather than the issue summary. The
    1.2x factor that stops a chart flipping between views every tick
    already exists, as `_RUN_VIEW_FACTOR` in `dashboard_compute.py`, and is
    multiplied in before `min_span_s` arrives here. So the repository's own
    test is a bare comparison and that is correct.

    What 5b changes is ownership, not behaviour: `_RUN_VIEW_FACTOR` is a
    local copy of `RunSeriesPolicy.retained_factor`, which is the same
    1.2, and the decision moves to `policy.mode_for` in the compute layer
    where the rest of the planning now lives.
    """
    path = system_db(ticks=40, cadence_s=2.0)  # 78 s span
    repo = _repo(path)
    with repo.connect() as conn:
        stats = repo.cpu_run_stats(conn)
    assert stats is not None
    mode = DEFAULT_RUN_SERIES_POLICY.mode_for
    # 78 s of run against a 60 s window is 1.3x, past the 1.2x factor.
    assert mode(stats.span_s, 60.0) == "retained"
    assert mode(stats.span_s, 70.0) == "recent"


def test_fewer_than_two_samples_cannot_be_planned(system_db):
    path = system_db(ticks=1)
    repo = _repo(path)
    with repo.connect() as conn:
        _stats, rows = _cpu_run(repo, conn)
    assert rows == []


# --- the point budget ----------------------------------------------------
def test_the_whole_run_series_respects_the_120_point_cap(system_db):
    path = system_db(ticks=600, cadence_s=2.0)
    repo = _repo(path)
    with repo.connect() as conn:
        _stats, rows = _cpu_run(repo, conn)
    assert 2 < len(rows) <= 120


def test_eligible_points_under_the_cap_all_survive(system_db):
    """The stride is planned over the points that will be EMITTED.

    A 30 s window at a 2 s cadence excludes the first 14 samples, whose
    rolling window is incomplete. The remaining 116 fit under the cap, so
    none of them should be decimated away.
    """
    path = system_db(ticks=130, cadence_s=2.0)
    repo = _repo(path)
    with repo.connect() as conn:
        _stats, rows = _cpu_run(repo, conn)
    assert len(rows) == 116


def test_the_window_and_span_come_from_the_plan_and_the_stats(system_db):
    """Both used to be assembled inside the read; they are inputs now."""
    path = system_db(ticks=600, cadence_s=2.0)
    repo = _repo(path)
    with repo.connect() as conn:
        stats = repo.cpu_run_stats(conn)
    assert stats is not None
    plan = plan_run_series(
        span_s=stats.span_s,
        sample_count=stats.sample_count,
        policy=DEFAULT_RUN_SERIES_POLICY,
    )
    assert plan is not None
    assert plan.window_s == 30.0
    assert stats.span_s == pytest.approx(2.0 * 599)


# --- the rolling frame, which 5b changes ---------------------------------
def test_the_frame_measures_seconds_so_a_gap_does_not_widen_it(system_db):
    """The behaviour change this PR exists to make, asserted directly.

    A ROWS frame reaches a fixed number of ROWS back regardless of how much
    wall clock they span. Across a 60 s hole in a 2 s run it averaged in
    pre-gap values a 30 s window should never have seen, so the chart's own
    label stopped being true exactly when someone was reading it to find
    out what happened.

    `RunSeriesPlan.frame_clause` emits `RANGE BETWEEN window_s PRECEDING`
    on SQLite 3.28 and later, which counts seconds. The first sample after
    the gap now averages only what is genuinely within 30 s of it, which is
    itself.

    This test was written in the previous commit asserting the OPPOSITE,
    against untouched code, so the inversion here is the record of the
    change rather than a claim about it.
    """
    path = system_db(
        ticks=200,
        cadence_s=2.0,
        cpu=lambda seq: 90.0 if seq < 100 else 10.0,
        gaps={100: 60.0},
    )
    repo = _repo(path)
    with repo.connect() as conn:
        _stats, rows = _cpu_run(repo, conn)

    first_after_gap = next(
        avg for ts, avg, _mx in rows if ts >= 1000.0 + 260.0
    )
    assert first_after_gap == pytest.approx(10.0, abs=0.5)


# --- node scope ----------------------------------------------------------
def test_history_is_scoped_to_one_host(system_db):
    """System telemetry is per machine; a two-host window is not pooled."""
    path = system_db(
        ticks=200,
        cadence_s=2.0,
        cpu=lambda seq: 10.0,
        hostnames=("node-a", "node-b"),
    )
    repo = _repo(path)
    with repo.connect() as conn:
        _stats, rows = _cpu_run(repo, conn, hostname="node-a")
    assert rows
    assert max(r[2] for r in rows) == pytest.approx(10.0)


def test_the_newest_sample_carries_the_run_identity(system_db):
    path = system_db(ticks=5)
    repo = _repo(path)
    with repo.connect() as conn:
        row = repo.fetch_latest_system_sample(conn)
    assert row is not None
    assert row["hostname"] == "box"
    assert row["cpu_percent"] == 10.0


def test_the_recent_window_returns_at_most_what_was_asked_for(system_db):
    path = system_db(ticks=50)
    repo = _repo(path)
    with repo.connect() as conn:
        rows = repo.fetch_recent_system_samples(conn, 10)
    assert len(rows) == 10


# --- the GPU power history, which buckets rather than rolls --------------
def test_power_history_buckets_per_gpu(system_db):
    path = system_db(
        ticks=600, cadence_s=2.0, gpus=lambda seq: [gpu(0), gpu(1)]
    )
    repo = _repo(path)
    with repo.connect() as conn:
        stats, out = _power_run(repo, conn)
    assert stats is not None
    assert len(out) == 2
    for entry in out:
        assert len(entry["t"]) == len(entry["avg"]) == len(entry["min"])
    assert stats.span_s == pytest.approx(2.0 * 599)


def test_power_history_is_capped_on_a_long_run(system_db):
    """A long run widens its buckets instead of growing its point count.

    This used to be pinned at 288 buckets per GPU for 24 hours, because
    the width saturated at the 300 s window ceiling and the count was
    just `span / 300`. It rose linearly forever: 576 at 48 hours, 2016 at
    seven days, and each entry carries four parallel arrays.
    """
    path = system_db(
        ticks=720, cadence_s=120.0, gpus=lambda seq: [gpu(0)]
    )  # a 24 hour run
    repo = _repo(path)
    with repo.connect() as conn:
        stats, out = _power_run(repo, conn)

    budget = DEFAULT_RUN_SERIES_POLICY.max_points
    assert len(out[0]["t"]) <= budget
    assert len(out[0]["t"]) == 120
    # The width did the work, and it is wider than the window ceiling.
    assert stats is not None
    width = DEFAULT_RUN_SERIES_POLICY.bucket_width_for(stats.span_s)
    assert width > DEFAULT_RUN_SERIES_POLICY.roll_max_s


def test_a_short_run_keeps_its_window_sized_buckets(system_db):
    """The cap is a ceiling, not a reshaping. Below it nothing moves.

    A two hour run's buckets are still the rolling window, so the chart
    keeps the resolution it had and the "rolling N min" label a reader
    sees does not change.
    """
    path = system_db(ticks=60, cadence_s=120.0, gpus=lambda seq: [gpu(0)])
    repo = _repo(path)
    with repo.connect() as conn:
        stats, out = _power_run(repo, conn)

    assert stats is not None
    assert DEFAULT_RUN_SERIES_POLICY.bucket_width_for(
        stats.span_s
    ) == DEFAULT_RUN_SERIES_POLICY.window_for(stats.span_s)
    assert len(out[0]["t"]) < DEFAULT_RUN_SERIES_POLICY.max_points


def test_a_legacy_zero_placeholder_is_not_a_power_reading(system_db):
    """An all-zero placeholder is unreported, not a 0 W observation."""

    def rows(seq):
        if seq >= 500:
            return [gpu(0, power=0.0, limit=0.0, mem_total=0.0, temp=0.0)]
        return [gpu(0, power=66.0)]

    path = system_db(ticks=600, cadence_s=2.0, gpus=rows)
    repo = _repo(path)
    with repo.connect() as conn:
        _stats, out = _power_run(repo, conn)
    assert out
    assert min(out[0]["min"]) == pytest.approx(66.0)


def test_a_cpu_only_run_has_no_power_history(system_db):
    path = system_db(ticks=600, cadence_s=2.0)
    repo = _repo(path)
    with repo.connect() as conn:
        assert repo.gpu_power_run_stats(conn) is None


# --- error handling, which 5b deliberately does NOT change ---------------
class _FailingConnection:
    def execute(self, *_args, **_kwargs):
        raise sqlite3.OperationalError("query failed")


def test_a_failed_read_returns_empty_instead_of_raising(system_db):
    """Pinned as-is, and it is a defect: see the follow-up issue.

    The one error boundary is `SystemDashboardComputer.compute()`, which
    returns the last good payload marked stale. Swallowing here turns a
    database failure into a confident "no data" and that boundary never
    runs. It is NOT fixed in 5b: `SystemRollups.status` is written by the
    boundary and read nowhere in `system_section.py`, so propagating alone
    would replace two empty charts with a frozen card carrying no stale
    marker, which is worse. The fix needs the card note too.

    The Process repository already propagates (`bde1ca6`), so this is also
    the assertion that records the two blocks disagreeing.
    """
    path = system_db(ticks=5)
    repo = _repo(path)
    assert repo.cpu_run_stats(_FailingConnection()) is None
    assert repo.gpu_power_run_stats(_FailingConnection()) is None
    assert (
        repo.fetch_gpu_power_run(
            _FailingConnection(), width_s=30.0, first_ts=0.0
        )
        == []
    )


def test_one_swallow_is_a_capability_fallback_not_a_defect(system_db):
    """Named so the follow-up does not delete all five uniformly.

    The `except` around the rolling query stands in for "this SQLite has no
    window functions" (< 3.25), where the whole-run view is genuinely
    unavailable and the recent window still stands. Removing it with the
    others would turn a missing chart into a dead card on an old host.
    """
    source = (
        __import__(
            "traceml_ai.renderers.system.repository",
            fromlist=["repository"],
        ).__file__
        or ""
    )
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "Window functions need SQLite >= 3.25" in text


# --- NULL sample timestamps (#418) ---------------------------------------
def test_a_null_timestamp_row_is_not_a_sample(system_db):
    """`sample_ts_s` is nullable, and a row without a clock is not usable.

    Reproduced before it was fixed: with 30 such rows the CPU read raised
    `TypeError` out of `float(r[0])`. It is masked below a threshold, which
    is why nothing caught it. SQLite sorts NULLs first and the read already
    drops the first `preceding_rows` as an incomplete window, so the crash
    only appears once the NULL count exceeds that prefix, and the prefix
    itself moves with the cadence.

    `TypeError` is not `sqlite3.Error`, so it escaped the reader's own
    handler and reached `compute()`, where the whole-run chart silently
    vanished and the card fell back to the recent view with no reason
    given.
    """
    path = system_db(ticks=200, cadence_s=2.0, null_ts_before=30)
    repo = _repo(path)
    with repo.connect() as conn:
        stats = repo.cpu_run_stats(conn)
        assert stats is not None
        # The count must describe rows that can be placed on a clock, or
        # the cadence derived from it is wrong for every one of them.
        assert stats.sample_count == 170
        _stats, rows = _cpu_run(repo, conn)
    assert rows
    assert all(ts is not None for ts, _a, _m in rows)


def test_a_null_timestamp_row_is_not_a_power_sample(system_db):
    """The same rule on the GPU path, which has the same nullable column."""
    path = system_db(
        ticks=200,
        cadence_s=2.0,
        gpus=lambda seq: [gpu(0)],
        null_ts_before=30,
    )
    repo = _repo(path)
    with repo.connect() as conn:
        stats = repo.gpu_power_run_stats(conn)
        assert stats is not None
        assert stats.sample_count == 170
        _stats, out = _power_run(repo, conn)
    assert out
    assert all(ts is not None for ts in out[0]["t"])


def test_one_unclocked_row_does_not_erase_the_whole_run_chart(system_db):
    """The costly half of #418, and the half the issue did not describe.

    The issue reported a crash, which needs more unclocked rows than the
    read's dropped prefix. The common case is one, and it never crashed:
    `float(r["sample_ts_s"] or 0.0)` in the compute layer turned a missing
    clock into epoch 1970, so the recent window appeared to span 54 years,
    the run could never outgrow it, and the whole-run chart silently
    vanished with the card falling back to the recent view.

    Silent and wrong is worse than loud and wrong, and one row is far more
    likely than thirty.
    """
    from traceml_ai.renderers.system.dashboard_compute import (
        SystemDashboardComputer,
    )

    path = system_db(ticks=200, cadence_s=2.0)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE system_samples SET sample_ts_s = NULL "
            "WHERE seq = (SELECT MAX(seq) FROM system_samples)"
        )
    out = SystemDashboardComputer(path).compute(window_n=100)

    assert out.series.cpu_run_mode == "retained"
    assert len(out.series.cpu_run.t) > 2
    # And no drawn timestamp is the empty string the 1970 value formatted to.
    assert all(x for x in out.series.x_time)


def test_a_dip_survives_the_cap_on_a_long_run(system_db):
    """Why the buckets widen instead of the series being strided.

    The chart is labelled "average and lowest". Widening keeps both words
    true, because the buckets TILE the run: every instant belongs to
    exactly one plotted slice, so a momentary dip is the minimum of the
    slice containing it and cannot fall between two plotted points. A
    strided series samples every Nth bucket and would leave holes.

    One tick out of 720 drops to 11 W.
    """
    cadence = 120.0

    def rows(seq):
        return [gpu(0, power=11.0 if seq == 431 else 66.0)]

    path = system_db(ticks=720, cadence_s=cadence, gpus=rows)
    repo = _repo(path)
    with repo.connect() as conn:
        stats, out = _power_run(repo, conn)

    assert stats is not None
    assert min(out[0]["min"]) == 11.0
    assert sum(1 for v in out[0]["min"] if v == 11.0) == 1
    # Averaged INTO its slice rather than becoming it, the stated cost.
    assert 11.0 < min(out[0]["avg"]) < 66.0

    # The three assertions above hold under the OLD narrow buckets too,
    # so alone they do not test this change at all. Coverage is what
    # does: consecutive buckets start within one width (plus the sample
    # cadence, since a bucket begins at its first sample rather than on
    # the boundary), which is only true while the slices are contiguous.
    width = DEFAULT_RUN_SERIES_POLICY.bucket_width_for(stats.span_s)
    stamps = out[0]["t"]
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert gaps, "a capped run still has more than one bucket"
    assert max(gaps) <= width + cadence
    # And the slices reach both ends of the run.
    assert stamps[0] == pytest.approx(stats.first_ts)
    assert stats.last_ts - stamps[-1] <= width + cadence


def test_a_sample_older_than_the_run_start_cannot_break_the_cap(system_db):
    """The bound is proved above the SQL, so the SQL has to hold it.

    `first_ts` comes from a separate query than the fetch, so on a live
    dashboard a delayed sample can arrive between the two carrying a
    timestamp older than the one already read. `CAST` truncates toward
    zero rather than flooring, so such a row would land in a NEGATIVE
    bucket and push the count past the budget the caller asserts.

    Clamped at zero, it folds into the first bucket instead.
    """
    path = system_db(ticks=720, cadence_s=120.0, gpus=lambda seq: [gpu(0)])
    repo = _repo(path)
    with repo.connect() as conn:
        stats = repo.gpu_power_run_stats(conn)
        assert stats is not None
        width = DEFAULT_RUN_SERIES_POLICY.bucket_width_for(stats.span_s)
        # A sample from well before the start the planner saw.
        rows = repo.fetch_gpu_power_run(
            conn, width_s=width, first_ts=stats.first_ts + 2.0 * width
        )

    budget = DEFAULT_RUN_SERIES_POLICY.max_points
    assert len(rows) <= budget
    # Every bucket index is non-negative, so the series still reads left
    # to right in time.
    stamps = [r[1] for r in rows]
    assert stamps == sorted(stamps)
