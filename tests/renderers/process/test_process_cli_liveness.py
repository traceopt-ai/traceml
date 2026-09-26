# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""A rank that stops reporting is named on the terminal (issue #358).

The terminal snapshot anchors on the slowest rank's latest seq, so a dead
rank freezes every figure on its last sample. These tests pin that the
terminal path carries each rank's last-seen age and state, judged by the
same rule the dashboard uses, and that the Process panel says which rank
went quiet instead of silently holding its numbers.
"""

from __future__ import annotations

import sqlite3

import pytest
from rich.console import Console

from tests.renderers.process.conftest import GB
from traceml_ai.renderers.process.cli_compute import ProcessCLIComputer
from traceml_ai.renderers.process.computer import ProcessMetricsComputer
from traceml_ai.renderers.process.dashboard_compute import (
    ProcessDashboardComputer,
)
from traceml_ai.renderers.process.renderer import ProcessRenderer

T0 = 1_700_000_000.0


def _run(db, *, ranks=2, samples=60, cadence_s=2.0, dies=None):
    """Every rank samples each tick; ``dies=(rank, seq)`` stops one."""
    for seq in range(1, samples + 1):
        for rank in range(ranks):
            if dies is not None and rank == dies[0] and seq > dies[1]:
                continue
            db.insert(
                recv_ts_ns=int((T0 + seq * cadence_s) * 1e9),
                rank=rank,
                global_rank=rank,
                node_rank=0,
                seq=seq,
                sample_ts_s=T0 + seq * cadence_s,
                cpu_percent=200.0 + 100.0 * rank,
                cpu_logical_core_count=8,
                ram_used_bytes=2.0 * GB,
                ram_total_bytes=64.0 * GB,
            )


def _render(renderable) -> str:
    console = Console(
        force_terminal=True, color_system=None, width=140, record=True
    )
    console.print(renderable)
    return console.export_text()


# --- the computer --------------------------------------------------------
def test_cli_snapshot_names_the_rank_that_stopped(process_db):
    """Rank 1 last spoke at seq 20; rank 0 is still reporting at seq 60."""
    _run(process_db, dies=(1, 20))
    snap = ProcessMetricsComputer(
        db_path=process_db.path, sampler_interval_s=2.0
    ).compute_cli()

    by_rank = {r["global_rank"]: r for r in snap["rank_liveness"]}
    assert sorted(by_rank) == [0, 1]
    assert by_rank[0]["freshness"] == "fresh"
    assert by_rank[0]["age_s"] == pytest.approx(0.0)
    assert by_rank[1]["freshness"] == "stale"
    # Seq 20 arrived at T0+40 s, the newest arrival is T0+120 s.
    assert by_rank[1]["age_s"] == pytest.approx(80.0)
    assert by_rank[1]["last_seen_s"] == pytest.approx(T0 + 40.0)


def test_cli_snapshot_keeps_the_held_figures_rather_than_zeroing(
    process_db,
):
    """The frozen reading stays a real reading; it is marked, not erased."""
    _run(process_db, dies=(1, 20))
    snap = ProcessMetricsComputer(
        db_path=process_db.path, sampler_interval_s=2.0
    ).compute_cli()
    assert snap["seq"] == 20
    assert snap["cpu_used"] == pytest.approx(300.0)


def test_cli_snapshot_reports_every_rank_fresh_when_all_report(process_db):
    _run(process_db)
    snap = ProcessMetricsComputer(
        db_path=process_db.path, sampler_interval_s=2.0
    ).compute_cli()
    assert [r["freshness"] for r in snap["rank_liveness"]] == [
        "fresh",
        "fresh",
    ]


def test_a_failed_heartbeat_read_on_a_metrics_tick_keeps_the_last_verdicts(
    process_db, monkeypatch
):
    """Unreadable is not "no rank stopped", even when the figures read.

    The last good verdicts answer for a failed read as long as a cached
    payload may (30 s here), then there is no verdict at all.
    """
    import traceml_ai.renderers.process.cli_compute as cli_compute

    _run(process_db, dies=(1, 20))
    now = [T0 + 121.0]
    computer = ProcessCLIComputer(
        db_path=process_db.path,
        sampler_interval_s=2.0,
        now_fn=lambda: now[0],
    )
    first = computer.compute()
    stale = [r for r in first["rank_liveness"] if r["freshness"] == "stale"]
    assert [r["global_rank"] for r in stale] == [1]

    def unreadable(*_args, **_kwargs):
        raise sqlite3.OperationalError("heartbeat unreadable")

    monkeypatch.setattr(cli_compute, "read_rank_clock", unreadable)
    now[0] += 10.0
    held = computer.compute()
    assert held["seq"] == 20
    assert held["rank_liveness"] == first["rank_liveness"]

    now[0] += 21.0  # 31 s since the last good read
    expired = computer.compute()
    assert expired["seq"] == 20
    assert expired["rank_liveness"] is None


def test_unreadable_figures_keep_the_verdict_the_heartbeat_gave(process_db):
    """A metric cell that cannot be read costs the figures, not the marker.

    The heartbeat read succeeds and names rank 1; the committed seq's
    figures then fail to parse, and no good snapshot is held yet.
    """
    _run(process_db, dies=(1, 20))
    conn = sqlite3.connect(process_db.path)
    conn.execute(
        "UPDATE process_samples SET cpu_percent = 'abc' "
        "WHERE rank = 1 AND seq = 20"
    )
    conn.commit()
    conn.close()

    renderer = ProcessRenderer(db_path=process_db.path, sampler_interval_s=2.0)
    snap = renderer._computer.compute_cli()
    text = _render(renderer.get_panel_renderable())

    assert snap["seq"] is None
    assert [
        (r["global_rank"], r["freshness"]) for r in snap["rank_liveness"]
    ] == [(0, "fresh"), (1, "stale")]
    assert "rank 1: no data for 80s (stale)" in text


def test_cli_and_dashboard_judge_every_rank_identically(process_db):
    """One owner for the judgement: both surfaces must agree per rank."""
    _run(process_db, ranks=4, samples=200, dies=(3, 20))
    cli = ProcessMetricsComputer(
        db_path=process_db.path, sampler_interval_s=2.0
    ).compute_cli()["rank_liveness"]
    dash = ProcessDashboardComputer(
        db_path=process_db.path, sampler_interval_s=2.0
    ).compute()

    assert [(r["global_rank"], r["age_s"], r["freshness"]) for r in cli] == [
        (r.global_rank, r.age_s, r.freshness) for r in dash.ranks
    ]


# --- the dashboard, characterized before the judgement was shared --------
def test_dashboard_freshness_values_are_unchanged(process_db):
    """Pinned on version_0.4.2 before the per-rank clock moved out.

    A changed age or state here is a changed dashboard.
    """
    _run(process_db, ranks=4, samples=200, dies=(3, 20))
    out = ProcessDashboardComputer(
        db_path=process_db.path, sampler_interval_s=2.0
    ).compute()

    assert [(r.global_rank, r.age_s, r.freshness) for r in out.ranks] == [
        (0, 0.0, "fresh"),
        (1, 0.0, "fresh"),
        (2, 0.0, "fresh"),
        (3, 360.0, "stale"),
    ]
    coverage = out.coverage
    assert (
        coverage.total,
        coverage.live,
        coverage.stale,
        coverage.unknown,
    ) == (4, 3, 1, 0)


def test_a_failed_rank_read_serves_the_last_good_dashboard_payload(
    process_db, monkeypatch
):
    """A failed rank read is a failed read, as on version_0.4.2.

    The last good payload answers for it, rank rows included, and is
    not replaced by one with the rank rows blanked. A read that then
    fails outright still gets the genuinely good payload.
    """
    import traceml_ai.renderers.process.dashboard_compute as dashboard

    _run(process_db, dies=(1, 20))
    computer = ProcessDashboardComputer(
        db_path=process_db.path, sampler_interval_s=2.0
    )
    first = computer.compute()
    assert [(r.global_rank, r.freshness) for r in first.ranks] == [
        (0, "fresh"),
        (1, "stale"),
    ]

    def unreadable(*_args, **_kwargs):
        raise sqlite3.OperationalError("heartbeat unreadable")

    monkeypatch.setattr(dashboard, "read_rank_clock", unreadable)
    held = computer.compute()
    assert len(held.ranks) == 2
    assert held is first

    monkeypatch.setattr(computer._db, "connect", unreadable)
    later = computer.compute()
    assert len(later.ranks) == 2
    assert later is first


# --- the terminal panel --------------------------------------------------
def test_process_panel_marks_the_stale_rank(process_db):
    _run(process_db, dies=(1, 20))
    text = _render(
        ProcessRenderer(
            db_path=process_db.path, sampler_interval_s=2.0
        ).get_panel_renderable()
    )
    assert "rank 1: no data for 80s (stale)" in text
    assert "rank 0:" not in text
    # The held reading is still printed, not replaced by zero.
    assert "3.00 cores" in text


def test_process_panel_has_no_marker_when_every_rank_reports(process_db):
    _run(process_db)
    text = _render(
        ProcessRenderer(
            db_path=process_db.path, sampler_interval_s=2.0
        ).get_panel_renderable()
    )
    assert "(stale)" not in text
    assert "no data for" not in text


def test_unparseable_rank_cell_is_skipped_and_the_other_ranks_judged(
    process_db,
):
    """One bad heartbeat row costs its own row, never every verdict."""
    _run(process_db, dies=(1, 20))
    process_db.insert(
        recv_ts_ns=int((T0 + 2.0) * 1e9),
        rank=0,
        global_rank="not-a-rank",
        seq=1,
        sample_ts_s=T0 + 2.0,
        cpu_percent=1.0,
    )

    snap = ProcessMetricsComputer(
        db_path=process_db.path, sampler_interval_s=2.0
    ).compute_cli()
    text = _render(
        ProcessRenderer(
            db_path=process_db.path, sampler_interval_s=2.0
        ).get_panel_renderable()
    )

    assert snap["seq"] == 20
    assert snap["cpu_used"] == pytest.approx(300.0)
    assert [
        (r["global_rank"], r["freshness"]) for r in snap["rank_liveness"]
    ] == [(0, "fresh"), (1, "stale")]
    assert "rank 1: no data for 80s (stale)" in text
    assert "3.00 cores" in text
