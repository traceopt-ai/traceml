# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Window planning, cadence, and the point budget."""

from __future__ import annotations

import pytest

from traceml_ai.renderers.shared.run_series import (
    DEFAULT_RUN_SERIES_POLICY,
    RunSeriesPolicy,
    cadence_of,
    plan_run_series,
    stride_for,
)

POLICY = DEFAULT_RUN_SERIES_POLICY


# --- the rolling window --------------------------------------------------
@pytest.mark.parametrize(
    "span_s, expected",
    [
        (0.0, 30.0),
        (-5.0, 30.0),
        (60.0, 30.0),
        (30 * 60.0, 60.0),
        (90 * 60.0, 120.0),
        (3 * 3600.0, 300.0),
        (24 * 3600.0, 300.0),
    ],
)
def test_the_window_climbs_a_ladder_of_round_steps(span_s, expected):
    """Round steps so the label a card prints stays a number a reader knows."""
    assert POLICY.window_for(span_s) == pytest.approx(expected)


def test_the_window_never_leaves_its_bounds():
    for span in (1.0, 10.0, 1e3, 1e6, 1e9):
        assert (
            POLICY.roll_min_s <= POLICY.window_for(span) <= POLICY.roll_max_s
        )


def test_the_window_ladder_is_monotonic():
    spans = [float(s) for s in range(0, 40_000, 137)]
    windows = [POLICY.window_for(s) for s in spans]
    assert windows == sorted(windows)


# --- recent vs retained --------------------------------------------------
def test_a_short_run_stays_on_the_recent_view():
    assert POLICY.mode_for(run_span_s=100.0, window_span_s=200.0) == "recent"


def test_a_run_that_outgrows_its_window_switches_to_retained():
    assert POLICY.mode_for(run_span_s=600.0, window_span_s=200.0) == "retained"


def test_the_switch_has_hysteresis_so_a_chart_does_not_flip_each_tick():
    """At exactly the window length, and just past it, the view holds."""
    assert POLICY.mode_for(run_span_s=200.0, window_span_s=200.0) == "recent"
    assert POLICY.mode_for(run_span_s=239.0, window_span_s=200.0) == "recent"
    assert POLICY.mode_for(run_span_s=241.0, window_span_s=200.0) == "retained"


def test_an_unknown_window_is_treated_as_recent():
    assert POLICY.mode_for(run_span_s=999.0, window_span_s=0.0) == "recent"


# --- cadence -------------------------------------------------------------
def test_cadence_is_measured_from_the_span_and_the_count():
    assert cadence_of(100.0, 51) == pytest.approx(2.0)


@pytest.mark.parametrize("span, count", [(0.0, 10), (100.0, 1), (100.0, 0)])
def test_cadence_is_unknown_rather_than_guessed(span, count):
    assert cadence_of(span, count) is None


# --- the point budget ----------------------------------------------------
def test_stride_uses_ceiling_division_so_the_budget_holds():
    """Floor division ships more points than the budget promises.

    600 eligible samples over a 120-point budget is 5 exactly; 601 needs 6,
    because 601 // 120 == 5 would yield 121 points.
    """
    assert stride_for(600, 120) == 5
    assert stride_for(601, 120) == 6
    assert stride_for(719, 120) == 6
    assert stride_for(721, 120) == 7


@pytest.mark.parametrize("count", [0, -1])
def test_stride_is_one_when_there_is_nothing_to_thin(count):
    assert stride_for(count, 120) == 1


def test_stride_never_drops_below_one():
    assert stride_for(10, 0) == 1
    assert stride_for(10, 1000) == 1


def test_the_budget_is_actually_respected_across_run_lengths():
    for count in (200, 601, 5_000, 50_000):
        plan = plan_run_series(span_s=count * 2.0, sample_count=count)
        assert plan is not None
        points = -(-plan.eligible_count // plan.stride)  # ceiling
        assert points <= plan.max_points, (count, points)


# --- the plan ------------------------------------------------------------
def test_a_plan_needs_at_least_two_samples():
    assert plan_run_series(span_s=10.0, sample_count=1) is None
    assert plan_run_series(span_s=0.0, sample_count=100) is None


def test_a_supplied_cadence_is_preferred_over_a_measured_one():
    plan = plan_run_series(span_s=1000.0, sample_count=101, cadence_s=5.0)
    assert plan is not None and plan.cadence_s == pytest.approx(5.0)


def test_partial_windows_are_excluded_before_the_stride_is_chosen():
    """The head of a run cannot carry a full rolling window.

    Those rows average fewer samples than they claim, so they are dropped
    from the budget rather than thinned into it.
    """
    plan = plan_run_series(span_s=2_000.0, sample_count=1_001)
    assert plan is not None
    assert plan.preceding_rows > 0
    assert plan.eligible_count == plan.sample_count - plan.preceding_rows


def test_the_frame_covers_the_window_it_names():
    plan = plan_run_series(span_s=3_600.0, sample_count=1_801)
    assert plan is not None
    clause = plan.frame_clause()
    if "RANGE" in clause:
        assert f"{plan.window_s:.6f}" in clause
    else:
        assert f"{plan.preceding_rows} PRECEDING" in clause


def test_a_custom_policy_is_honoured():
    tight = RunSeriesPolicy(max_points=10, roll_min_s=10.0, roll_max_s=20.0)
    assert tight.window_for(1e6) == pytest.approx(20.0)
    plan = plan_run_series(span_s=1_000.0, sample_count=501, policy=tight)
    assert plan is not None and plan.max_points == 10


# --- corrupt numbers ------------------------------------------------------
NAN, INF = float("nan"), float("-inf")


@pytest.mark.parametrize("bad", [NAN, INF, float("inf")])
def test_a_corrupt_span_plans_nothing_rather_than_crashing(bad):
    """NaN walks past every ordinary guard.

    `nan <= 0` is False, so a corrupt sample clears a positivity check and
    only fails later, at an int() that cannot convert it. Measured before
    this guard existed: ValueError, cannot convert float NaN to integer.
    """
    assert plan_run_series(span_s=bad, sample_count=101) is None


@pytest.mark.parametrize("bad", [NAN, INF, float("inf")])
def test_a_corrupt_cadence_plans_nothing(bad):
    assert (
        plan_run_series(span_s=1000.0, sample_count=101, cadence_s=bad) is None
    )


def test_a_corrupt_span_yields_no_cadence():
    assert cadence_of(NAN, 100) is None
    assert cadence_of(float("inf"), 100) is None


def test_a_corrupt_span_falls_back_to_the_smallest_window():
    assert DEFAULT_RUN_SERIES_POLICY.window_for(NAN) == pytest.approx(30.0)
    assert DEFAULT_RUN_SERIES_POLICY.mode_for(NAN, 100.0) == "recent"
    assert DEFAULT_RUN_SERIES_POLICY.mode_for(100.0, NAN) == "recent"


def test_a_hand_built_plan_with_corrupt_numbers_still_answers():
    """The dataclass is public, so it is guarded a second time."""
    from traceml_ai.renderers.shared.run_series import RunSeriesPlan

    plan = RunSeriesPlan(
        window_s=NAN, cadence_s=NAN, stride=1, max_points=120, sample_count=10
    )
    assert plan.preceding_rows == 1
    assert "nan" not in plan.frame_clause().lower()


def test_bucket_width_caps_the_point_count_on_a_long_run():
    """Past the window ceiling the bucket widens instead of multiplying.

    `window_for` saturates at its 300 s ceiling, so a bucket per window
    means the count is `span / 300` and rises linearly with run length
    forever: 288 per GPU at 24 hours, 2016 at 7 days, against the 120
    the CPU chart budgets.
    """
    policy = RunSeriesPolicy()

    for span in (86_400.0, 172_800.0, 604_800.0):
        width = policy.bucket_width_for(span)
        assert int(span / width) + 1 <= policy.max_points

    one_point = RunSeriesPolicy(max_points=1)
    assert int(86_400.0 / one_point.bucket_width_for(86_400.0)) + 1 == 1


def test_bucket_width_leaves_short_runs_alone():
    """A run whose window already fits the budget is not re-shaped.

    The cap is a ceiling, not a replacement rule: below it the bucket is
    still the rolling window, so the chart keeps the resolution it has
    and the label a reader sees does not move.
    """
    policy = RunSeriesPolicy()

    assert policy.bucket_width_for(3_600.0) == policy.window_for(3_600.0)
    assert policy.bucket_width_for(21_600.0) == policy.window_for(21_600.0)
    assert 3_600.0 / policy.bucket_width_for(3_600.0) == 30
    assert 21_600.0 / policy.bucket_width_for(21_600.0) == 72


def test_a_bucket_is_never_narrower_than_the_rolling_window():
    """Widening only. The bucket cannot drop below the window.

    A narrower bucket would give the whole-run chart finer resolution
    than the live window it is supposed to summarise.
    """
    policy = RunSeriesPolicy()

    for span in (0.0, 60.0, 3_600.0, 86_400.0, 604_800.0, 6_048_000.0):
        assert policy.bucket_width_for(span) >= policy.window_for(span)


def test_a_degenerate_span_still_yields_a_usable_width():
    """No division by zero, and never a zero or negative width."""
    policy = RunSeriesPolicy()

    for span in (0.0, -1.0, float("nan"), float("inf")):
        assert policy.bucket_width_for(span) > 0.0
