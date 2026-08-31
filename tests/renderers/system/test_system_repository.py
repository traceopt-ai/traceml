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
* the retained-vs-recent gate is a bare `span > min_span_s` with no
  hysteresis,
* a failed read returns an empty result rather than raising, which turns a
  database failure into a confident "no data".
"""

from __future__ import annotations

import sqlite3

import pytest

from tests.renderers.system.conftest import gpu
from traceml_ai.renderers.system.repository import SystemRepository


def _repo(path: str) -> SystemRepository:
    return SystemRepository(db_path=path)


# --- the window ladder ---------------------------------------------------
def test_the_rolling_window_is_chosen_from_the_run_length(system_db):
    """Round steps, so the label a card prints stays recognisable."""
    from traceml_ai.renderers.system.common import choose_window_s

    assert choose_window_s(4 * 60) == 30.0  # the floor
    assert choose_window_s(23 * 60) == 30.0
    assert choose_window_s(96 * 60) == 120.0
    assert choose_window_s(178 * 60) == 300.0
    assert choose_window_s(48 * 3600) == 300.0  # the cap


def test_a_run_shorter_than_the_gate_has_no_whole_run_history(system_db):
    """`min_span_s` is what keeps a short run off the expensive query."""
    path = system_db(ticks=10, cadence_s=2.0)  # 18 s span
    repo = _repo(path)
    with repo.connect() as conn:
        out = repo.fetch_cpu_run_history(conn, min_span_s=60.0)
    assert out["t"] == []
    assert out["span_s"] == 0.0


def test_the_gate_is_a_bare_comparison_with_no_hysteresis(system_db):
    """Pinned because 5b replaces it with `policy.mode_for`.

    A span one microsecond over the threshold produces the whole-run view
    today. `mode_for` requires the run to outgrow the window by 1.2x first,
    so a chart near the boundary cannot flip back and forth every tick.
    """
    path = system_db(ticks=40, cadence_s=2.0)  # 78 s span
    repo = _repo(path)
    with repo.connect() as conn:
        assert repo.fetch_cpu_run_history(conn, min_span_s=77.9)["t"] != []
        assert repo.fetch_cpu_run_history(conn, min_span_s=78.1)["t"] == []


def test_fewer_than_two_samples_cannot_be_planned(system_db):
    path = system_db(ticks=1)
    repo = _repo(path)
    with repo.connect() as conn:
        assert repo.fetch_cpu_run_history(conn)["t"] == []


# --- the point budget ----------------------------------------------------
def test_the_whole_run_series_respects_the_120_point_cap(system_db):
    path = system_db(ticks=600, cadence_s=2.0)
    repo = _repo(path)
    with repo.connect() as conn:
        out = repo.fetch_cpu_run_history(conn)
    assert 2 < len(out["t"]) <= 120
    assert len(out["t"]) == len(out["avg"]) == len(out["max"])


def test_eligible_points_under_the_cap_all_survive(system_db):
    """The stride is planned over the points that will be EMITTED.

    A 30 s window at a 2 s cadence excludes the first 14 samples, whose
    rolling window is incomplete. The remaining 116 fit under the cap, so
    none of them should be decimated away.
    """
    path = system_db(ticks=130, cadence_s=2.0)
    repo = _repo(path)
    with repo.connect() as conn:
        out = repo.fetch_cpu_run_history(conn)
    assert len(out["t"]) == 116


def test_the_window_travels_on_the_payload(system_db):
    path = system_db(ticks=600, cadence_s=2.0)
    repo = _repo(path)
    with repo.connect() as conn:
        out = repo.fetch_cpu_run_history(conn)
    assert out["window_s"] == 30.0
    assert out["span_s"] == pytest.approx(2.0 * 599)


# --- the rolling frame, which 5b changes ---------------------------------
def test_the_frame_counts_rows_so_a_gap_widens_it(system_db):
    """The defect issue #419 documents, pinned before it is fixed.

    A ROWS frame reaches a fixed number of rows back regardless of how much
    wall clock those rows span. Across a 60 s hole in a 2 s run it averages
    in pre-gap values that a 30 s window should never have seen, so the
    chart's own label stops being true exactly when someone is reading it
    to find out what happened.

    5b swaps this for `RANGE BETWEEN window_s PRECEDING`, and this
    assertion inverts as the stated behaviour change.
    """
    path = system_db(
        ticks=200,
        cadence_s=2.0,
        cpu=lambda seq: 90.0 if seq < 100 else 10.0,
        gaps={100: 60.0},
    )
    repo = _repo(path)
    with repo.connect() as conn:
        out = repo.fetch_cpu_run_history(conn)

    first_after_gap = next(
        avg for t, avg in zip(out["t"], out["avg"]) if t >= 1000.0 + 260.0
    )
    # A time frame would report 10.0 here: nothing else is within 30 s.
    assert first_after_gap > 50.0


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
        out = repo.fetch_cpu_run_history(conn, hostname="node-a")
    assert out["t"]
    assert max(out["max"]) == pytest.approx(10.0)


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
        out = repo.fetch_gpu_power_run_history(conn)
    assert [entry["gpu_idx"] for entry in out] == [0, 1]
    for entry in out:
        assert len(entry["t"]) == len(entry["avg"]) == len(entry["min"])
        assert entry["span_s"] == pytest.approx(2.0 * 599)


def test_power_history_has_no_point_cap(system_db):
    """Pinned as a known gap, deferred to its own issue.

    The CPU chart budgets 120 points. This path buckets at one bucket per
    `choose_window_s`, which saturates at 300 s, so bucket count grows
    without bound: about 288 per GPU on a 24 hour run.
    """
    path = system_db(
        ticks=720, cadence_s=120.0, gpus=lambda seq: [gpu(0)]
    )  # a 24 hour run
    repo = _repo(path)
    with repo.connect() as conn:
        out = repo.fetch_gpu_power_run_history(conn)
    # 86400 s of run at the 300 s window cap is 288 buckets, against the
    # 120 the CPU chart budgets for. On an 8-GPU host that is ~2300 points.
    assert len(out[0]["t"]) == 288


def test_the_sampler_zero_fallback_is_not_a_power_reading(system_db):
    """An all-zero capacity row is NVML failing, not a 0 W observation."""

    def rows(seq):
        if seq >= 500:
            return [gpu(0, power=0.0, limit=0.0, mem_total=0.0, temp=0.0)]
        return [gpu(0, power=66.0)]

    path = system_db(ticks=600, cadence_s=2.0, gpus=rows)
    repo = _repo(path)
    with repo.connect() as conn:
        out = repo.fetch_gpu_power_run_history(conn)
    assert out
    assert min(out[0]["min"]) == pytest.approx(66.0)


def test_a_cpu_only_run_has_no_power_history(system_db):
    path = system_db(ticks=600, cadence_s=2.0)
    repo = _repo(path)
    with repo.connect() as conn:
        assert repo.fetch_gpu_power_run_history(conn) == []


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
    assert repo.fetch_cpu_run_history(_FailingConnection())["t"] == []
    assert repo.fetch_gpu_power_run_history(_FailingConnection()) == []


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
