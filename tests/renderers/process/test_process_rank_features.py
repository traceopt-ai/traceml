# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Per-rank facts, rollups, coverage and the whole-run series.

The payload only: nothing here imports NiceGUI, so a failure points at the
computation rather than at a card.
"""

from __future__ import annotations

import pytest

from tests.renderers.process.conftest import GB
from traceml_ai.renderers.process.dashboard_compute import (
    ProcessDashboardComputer,
)
from traceml_ai.renderers.process.dashboard_models import RankSnapshot
from traceml_ai.renderers.process.renderer import ProcessRenderer


def payload(db, **kw):
    return ProcessDashboardComputer(db_path=db.path, **kw).compute()


def _run(db, *, ranks=4, samples=120, cores=8, hot_rank=None, dies=None):
    """A multi-rank run. ``hot_rank`` burns CPU; ``dies`` stops early."""
    for seq in range(1, samples + 1):
        for rank in range(ranks):
            if dies is not None and rank == dies[0] and seq > dies[1]:
                continue
            db.insert(
                recv_ts_ns=int((1_700_000_000 + seq * 2) * 1e9),
                rank=rank,
                global_rank=rank,
                node_rank=0,
                seq=seq,
                sample_ts_s=1_700_000_000.0 + seq * 2,
                cpu_percent=(700.0 if rank == hot_rank else 200.0),
                cpu_logical_core_count=cores,
                ram_used_bytes=(2.0 + rank * 0.5) * GB,
                ram_total_bytes=64.0 * GB,
                gpu_available=1,
                gpu_device_index=rank,
                gpu_mem_used_bytes=(4.0 + rank) * GB,
                gpu_mem_reserved_bytes=(6.0 + rank * 2) * GB,
                gpu_mem_total_bytes=40.0 * GB,
            )


# --- per-rank history ----------------------------------------------------
def test_every_rank_appears_with_its_own_facts(process_db):
    _run(process_db, ranks=4, samples=30)
    out = payload(process_db, sampler_interval_s=2.0)
    assert [r.global_rank for r in out.ranks] == [0, 1, 2, 3]
    assert all(r.cpu_capacity_percent is not None for r in out.ranks)


def test_cpu_is_normalized_against_the_core_count(process_db):
    """psutil sums CPU over cores, so a healthy 8-core rank reads 200%.

    Divided by its cores that is 25% of the host, which is bounded and
    comparable between ranks on different machines.
    """
    _run(process_db, ranks=1, samples=30, cores=8)
    rank = payload(process_db, sampler_interval_s=2.0).ranks[0]
    assert rank.cpu_capacity_percent == pytest.approx(25.0)


def test_a_rank_that_stopped_is_kept_and_marked_stale(process_db):
    """Dropping it would forget the dead rank exactly when it matters."""
    _run(process_db, ranks=4, samples=200, dies=(3, 20))
    out = payload(process_db, sampler_interval_s=2.0)
    assert [r.global_rank for r in out.ranks] == [0, 1, 2, 3]
    dead = next(r for r in out.ranks if r.global_rank == 3)
    assert dead.freshness == "stale"
    assert dead.age_s is not None and dead.age_s > 0
    assert all(r.freshness == "fresh" for r in out.ranks if r.global_rank != 3)


def test_an_unknown_age_is_counted_apart_from_live_and_stale(process_db):
    """The three states stay three; none of them absorbs another.

    `FreshnessPolicy.state_of` answers fresh / stale / unknown, and an
    unknown age means a rank sent data with no usable arrival clock. Our
    own writer cannot produce that (`recv_ts_ns` is `INTEGER NOT NULL`),
    so this is a guard rather than a live path, and it is asserted here
    rather than through the database for exactly that reason.

    The rank still counts toward the aggregate: a clock problem is not
    evidence that a rank stopped working. It is counted in its own bucket
    so that "live" never quietly means "not proven dead".
    """
    computer = ProcessDashboardComputer(
        db_path=process_db.path, sampler_interval_s=2.0
    )
    ranks = (
        RankSnapshot(global_rank=0, freshness="fresh", age_s=1.0),
        RankSnapshot(global_rank=1, freshness="stale", age_s=900.0),
        RankSnapshot(global_rank=2, freshness="unknown", age_s=None),
    )

    coverage = computer._build_payload(ranks).coverage
    assert (coverage.total, coverage.live, coverage.stale) == (3, 1, 1)
    assert coverage.unknown == 1
    assert coverage.live + coverage.stale + coverage.unknown == coverage.total


def test_coverage_states_who_is_reporting(process_db):
    _run(process_db, ranks=4, samples=200, dies=(3, 20))
    coverage = payload(process_db, sampler_interval_s=2.0).coverage
    assert (coverage.total, coverage.live, coverage.stale) == (4, 3, 1)
    assert coverage.unknown == 0
    assert coverage.excluding_stale is True


def test_a_healthy_run_is_not_marked_as_excluding_anything(process_db):
    _run(process_db, ranks=4, samples=30)
    assert (
        payload(process_db, sampler_interval_s=2.0).coverage.excluding_stale
        is False
    )


def test_renderer_passes_configured_cadence_to_freshness(process_db):
    """The configured cadence reaches the policy before it can be observed."""
    base = 1_700_000_000.0
    for rank, arrival_offset_s in ((0, 0.0), (1, 10.0)):
        process_db.insert(
            recv_ts_ns=int((base + arrival_offset_s) * 1e9),
            rank=rank,
            global_rank=rank,
            seq=1,
            sample_ts_s=base + arrival_offset_s,
            cpu_percent=20.0,
            cpu_logical_core_count=8,
            ram_used_bytes=2.0 * GB,
            ram_total_bytes=16.0 * GB,
        )

    out = ProcessRenderer(
        db_path=process_db.path,
        sampler_interval_s=60.0,
    ).get_dashboard_renderable()

    assert [rank.freshness for rank in out.ranks] == ["fresh", "fresh"]


# --- rollups -------------------------------------------------------------
def test_cpu_leads_with_the_worst_rank_and_names_it(process_db):
    """A bottleneck finder has to answer "which rank"."""
    _run(process_db, ranks=4, samples=30, hot_rank=2)
    cpu = payload(process_db, sampler_interval_s=2.0).cpu_capacity
    assert cpu.worst_rank == 2
    assert cpu.now == pytest.approx(87.5)
    assert cpu.p50 == pytest.approx(25.0)


def test_cuda_reports_the_least_headroom_rank_not_the_largest_user(
    process_db,
):
    _run(process_db, ranks=4, samples=30)
    gpu = payload(process_db, sampler_interval_s=2.0).gpu_reserved
    assert gpu.worst_rank == 3  # reserved 12 GB of 40, least headroom
    assert gpu.now == pytest.approx(12.0 * GB)


def test_reserved_imbalance_is_a_spread_across_ranks(process_db):
    _run(process_db, ranks=4, samples=30)
    out = payload(process_db, sampler_interval_s=2.0)
    # reserved runs 6, 8, 10, 12 GB -> (12-6)/12
    assert out.reserved_imbalance_percent == pytest.approx(50.0)


def test_a_single_rank_has_no_imbalance_to_report(process_db):
    _run(process_db, ranks=1, samples=30)
    assert (
        payload(process_db, sampler_interval_s=2.0).reserved_imbalance_percent
        is None
    )


def test_aggregates_describe_the_live_ranks(process_db):
    """A dead rank keeps its row but does not drag a headline."""
    _run(process_db, ranks=4, samples=200, hot_rank=3, dies=(3, 20))
    out = payload(process_db, sampler_interval_s=2.0)
    assert out.cpu_capacity.worst_rank != 3
    assert any(r.global_rank == 3 for r in out.ranks)


# --- the whole-run series ------------------------------------------------
def test_a_short_run_stays_on_the_recent_view(process_db):
    """Recent, and carrying data: the live view is the common case.

    A run only outgrows its window after minutes, so nearly every chart a
    reader sees is this one. An earlier version of this test asserted the
    recent chart was EMPTY, which described the code rather than the
    intent and left both charts blank on every short run.
    """
    _run(process_db, ranks=2, samples=20)
    out = payload(process_db, sampler_interval_s=2.0)
    chart = out.cpu_capacity_chart
    assert chart.mode == "recent"
    assert chart.is_retained is False
    assert len(chart.traces) == 2
    assert all(trace.timestamps for trace in chart.traces)
    assert all(
        len(trace.timestamps) == len(trace.values) for trace in chart.traces
    )


def test_the_recent_view_moves_between_ticks(process_db):
    """It must not be served from the whole-run cache.

    The retained chart is a rolling mean over minutes and is cached
    because rebuilding it every tick was this block's largest cost. The
    recent chart is the live one, and a cached copy of it is a chart that
    stops moving.
    """
    _run(process_db, ranks=2, samples=20)
    computer = ProcessDashboardComputer(
        db_path=process_db.path, sampler_interval_s=2.0
    )
    first = computer.compute().rss_chart
    _run(process_db, ranks=2, samples=40)
    second = computer.compute().rss_chart

    assert first.mode == second.mode == "recent"
    longest_before = max(len(t.timestamps) for t in first.traces)
    longest_after = max(len(t.timestamps) for t in second.traces)
    assert longest_after > longest_before


def test_every_rank_gets_its_own_recent_line(process_db):
    _run(process_db, ranks=4, samples=20)
    chart = payload(process_db, sampler_interval_s=2.0).rss_chart
    assert [t.global_rank for t in chart.traces] == [0, 1, 2, 3]


def test_a_long_run_switches_to_the_retained_view(process_db):
    _run(process_db, ranks=2, samples=300)
    chart = payload(process_db, sampler_interval_s=2.0).cpu_capacity_chart
    assert chart.mode == "retained"
    assert chart.is_retained is True
    assert chart.window_s is not None
    assert len(chart.traces) == 2


def test_the_mode_is_stated_never_inferred_from_an_empty_field(process_db):
    """The previous payload left the view to guess from whichever series
    happened to be populated. The mode is a field now."""
    _run(process_db, ranks=2, samples=300)
    out = payload(process_db, sampler_interval_s=2.0)
    for chart in (out.cpu_capacity_chart, out.rss_chart):
        assert chart.mode in ("recent", "retained")


def test_the_point_budget_holds_for_every_rank(process_db):
    """The stride is applied per rank, so it must be planned on the
    BUSIEST rank. Planned on the average, a four-rank run with one early
    death produced 145 points against a budget of 120."""
    _run(process_db, ranks=4, samples=300, dies=(3, 100))
    out = payload(process_db, sampler_interval_s=2.0)
    for chart in (out.cpu_capacity_chart, out.rss_chart):
        for trace in chart.traces:
            assert len(trace.values) <= 120, (
                chart.mode,
                trace.global_rank,
                len(trace.values),
            )


def test_each_retained_trace_carries_its_own_rank_and_peaks(process_db):
    _run(process_db, ranks=3, samples=300)
    chart = payload(process_db, sampler_interval_s=2.0).rss_chart
    assert [t.global_rank for t in chart.traces] == [0, 1, 2]
    for trace in chart.traces:
        assert len(trace.timestamps) == len(trace.values) == len(trace.peaks)


# --- the older payload keeps working -------------------------------------
def test_the_fields_the_card_already_reads_are_unchanged(process_db):
    """PR 4 changes the card; this PR must not."""
    _run(process_db, ranks=2, samples=30)
    out = payload(process_db, sampler_interval_s=2.0)
    assert out.has_data is True
    assert out.window_len > 0
    assert out.gpu_available is True
    assert out.chart is not None and out.chart.ram_percent.values
