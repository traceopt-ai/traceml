# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Executable invariants for the Step Time window (issue #257 review).

These pin the *classes* of defect surfaced in review, not single
instances, by fuzzing the dimension each rule quantifies over:

- F1 availability is metric-specific: a required (non-occurrence) metric
  present in only some steps drops availability instead of averaging a
  fragment; an occurrence-driven metric stays available.
- F2 a metric summary's ``worst_total`` and ``worst_rank`` always come
  from one ordering: the named rank produced the reported worst value.
- F3 typed values preserve absence: a metric the window
  declined to derive is ``None``, never a fabricated number.

The drift guard fails when a new selected metric is added without a
conscious occurrence-vs-required classification.
"""

from __future__ import annotations

import itertools

import pytest

from traceml_ai.step_time.analysis import (
    OCCURRENCE_METRICS,
    SELECTED_METRICS,
)
from tests.step_time.factories import (
    rank_average,
    values_from_mapping,
    window_from_events,
    window_from_rank_averages,
)

_EVENT_NAMES = {
    "input_wait": "_traceml_internal:dataloader_next",
    "h2d": "_traceml_internal:h2d_time",
    "forward": "_traceml_internal:forward_time",
    "backward": "_traceml_internal:backward_time",
    "optimizer_step": "_traceml_internal:optimizer_step",
    "step_time": "_traceml_internal:step_time",
}

_BASE_MS = {
    "input_wait": 4.0,
    "h2d": 1.0,
    "forward": 20.0,
    "backward": 30.0,
    "optimizer_step": 10.0,
    "step_time": 64.0,
}

# Every metric that is not occurrence-driven must be measured on every
# aligned step to stay available.
_REQUIRED_METRICS = tuple(
    m for m in SELECTED_METRICS if m not in OCCURRENCE_METRICS
)

_WINDOW = 6


def _stat(value: float) -> dict:
    return {
        "cpu": {
            "duration_ms": value,
            "cpu_ms": value,
            "gpu_ms": None,
            "n_calls": 1,
        }
    }


def _window_from_presence(present_steps: dict[str, set[int]]):
    per_rank = {0: {}}
    for step in range(_WINDOW):
        events = {}
        for metric, steps in present_steps.items():
            if step in steps:
                events[_EVENT_NAMES[metric]] = _stat(_BASE_MS[metric])
        per_rank[0][step] = events
    return window_from_events(per_rank, max_rows=_WINDOW)


def _all_steps() -> dict[str, set[int]]:
    return {m: set(range(_WINDOW)) for m in _EVENT_NAMES}


# ---------------------------------------------------------------------------
# Invariant helper reused across cases
# ---------------------------------------------------------------------------


def assert_metric_worst_is_self_consistent(window) -> None:
    """F2: every metric's worst value and worst rank share one ordering."""
    for metric in window.metrics:
        worst_rank = metric.worst_rank
        if worst_rank is None:
            continue
        rank_facts = window.rank(worst_rank)
        assert (
            rank_facts is not None
        ), f"{metric.metric}: worst_rank {worst_rank} is not a window rank"
        rank_value = rank_facts.average.value(metric.metric)
        assert (
            rank_value is not None
        ), f"{metric.metric}: worst_rank {worst_rank} never measured it"
        assert rank_value == pytest.approx(metric.worst_total), (
            f"{metric.metric}: reported worst {metric.worst_total} "
            f"but rank {worst_rank} has {rank_value}"
        )


# ---------------------------------------------------------------------------
# F1: availability is metric-specific
# ---------------------------------------------------------------------------


def test_window_carries_canonical_metric_rank_availability() -> None:
    presence_by_rank = {
        0: _all_steps(),
        1: _all_steps(),
    }
    del presence_by_rank[1]["forward"]
    del presence_by_rank[0]["input_wait"]
    del presence_by_rank[1]["input_wait"]

    per_rank = {0: {}, 1: {}}
    for rank, presence in presence_by_rank.items():
        for step in range(_WINDOW):
            events = {}
            for metric, present_steps in presence.items():
                if step in present_steps:
                    value = (
                        0.0
                        if rank == 0 and metric == "backward"
                        else _BASE_MS[metric]
                    )
                    events[_EVENT_NAMES[metric]] = _stat(value)
            per_rank[rank][step] = events

    window = window_from_events(
        per_rank,
        max_rows=_WINDOW,
        expected_ranks=(0, 1),
    )
    assert window.rank_universe == (0, 1)
    assert window.ranks_for("input_wait") == ()
    assert window.ranks_for("forward") == (0,)
    assert window.ranks_for("backward") == (0, 1)
    assert rank_average(window, 0).backward_ms == pytest.approx(0.0)
    assert window.ranks_for("input_wait") != window.rank_universe
    assert window.ranks_for("forward") != window.rank_universe
    assert window.ranks_for("backward") == window.rank_universe
    assert tuple(
        sorted(
            set(window.ranks_for("forward"))
            & set(window.ranks_for("backward"))
        )
    ) == (0,)


def test_empty_window_keeps_expected_rank_universe() -> None:
    window = window_from_events(
        {},
        max_rows=_WINDOW,
        expected_ranks=(0, 1),
    )
    assert window.rank_universe == (0, 1)
    assert all(window.ranks_for(metric) == () for metric in SELECTED_METRICS)


def test_skew_requires_two_measured_ranks_but_zero_remains_measured() -> None:
    single = _window_from_presence(_all_steps())
    single_step = next(m for m in single.metrics if m.metric == "step_time")
    assert single_step.skew_pct is None

    presence_by_rank = {0: _all_steps(), 1: _all_steps()}
    per_rank = {0: {}, 1: {}}
    for rank, presence in presence_by_rank.items():
        for step in range(_WINDOW):
            events = {
                _EVENT_NAMES[metric]: _stat(
                    0.0 if metric == "backward" else _BASE_MS[metric]
                )
                for metric, present_steps in presence.items()
                if step in present_steps
            }
            per_rank[rank][step] = events

    window = window_from_events(
        per_rank,
        max_rows=_WINDOW,
        expected_ranks=(0, 1),
    )
    backward = next(m for m in window.metrics if m.metric == "backward")
    assert backward.skew_pct == pytest.approx(0.0)


def test_skew_is_unavailable_when_zero_median_hides_a_nonzero_rank() -> None:
    per_rank = {rank: {} for rank in range(3)}
    for rank in per_rank:
        for step in range(_WINDOW):
            per_rank[rank][step] = {
                _EVENT_NAMES[metric]: _stat(
                    10.0
                    if metric == "backward" and rank == 2
                    else (0.0 if metric == "backward" else _BASE_MS[metric])
                )
                for metric in _EVENT_NAMES
            }

    window = window_from_events(
        per_rank,
        max_rows=_WINDOW,
        expected_ranks=(0, 1, 2),
    )
    backward = next(m for m in window.metrics if m.metric == "backward")
    assert backward.median_total == pytest.approx(0.0)
    assert backward.worst_total == pytest.approx(10.0)
    assert backward.skew_pct is None


def test_iteration_share_preserves_unavailable_and_measured_zero() -> None:
    rows = {
        0: {
            "input_wait": 100.0,
            "step_time": 100.0,
            "residual_proxy": 10.0,
        }
    }
    assert window_from_rank_averages(rows).residual_share == pytest.approx(
        0.05
    )
    rows[0]["residual_proxy"] = 0.0
    assert window_from_rank_averages(rows).residual_share == pytest.approx(0.0)
    del rows[0]["residual_proxy"]
    assert window_from_rank_averages(rows).residual_share is None

    invalid_compute = {
        0: {
            "input_wait": 10.0,
            "step_time": 100.0,
            "backward": 20.0,
            "optimizer_step": 10.0,
        }
    }
    assert window_from_rank_averages(invalid_compute).compute_share is None


@pytest.mark.parametrize("metric", _REQUIRED_METRICS)
@pytest.mark.parametrize("present_count", [1, _WINDOW // 2, _WINDOW - 1])
def test_required_metric_partial_presence_drops_availability(
    metric: str, present_count: int
) -> None:
    presence = _all_steps()
    presence[metric] = set(range(present_count))  # present in a strict subset
    window = _window_from_presence(presence)

    assert rank_average(window, 0).value(metric) is None, (
        f"{metric} present in {present_count}/{_WINDOW} steps must be "
        "unavailable, not averaged from a fragment"
    )
    assert metric not in {m.metric for m in window.metrics}


@pytest.mark.parametrize("metric", sorted(OCCURRENCE_METRICS))
@pytest.mark.parametrize("present_count", [1, _WINDOW // 2, _WINDOW - 1])
def test_occurrence_metric_partial_presence_stays_available(
    metric: str, present_count: int
) -> None:
    presence = _all_steps()
    presence[metric] = set(range(present_count))
    window = _window_from_presence(presence)

    assert rank_average(window, 0).value(metric) is not None
    # Averaged over the whole window: absent steps are true zeros.
    assert rank_average(window, 0).value(metric) == pytest.approx(
        _BASE_MS[metric] * present_count / _WINDOW
    )


def test_required_metric_full_presence_is_available() -> None:
    window = _window_from_presence(_all_steps())
    for metric in _REQUIRED_METRICS:
        assert rank_average(window, 0).value(metric) is not None


# ---------------------------------------------------------------------------
# F2: worst value and worst rank come from one ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slow_rank_lacks_input", [True, False])
def test_worst_value_and_rank_are_self_consistent(
    slow_rank_lacks_input: bool,
) -> None:
    # Two ranks, the slower one optionally missing input wait so its
    # total_step is underivable. The step-time summary must never pair a
    # value from one rank with a rank id from another.
    def fill(target, rank, step_time, drop_input):
        row = {}
        for metric in _EVENT_NAMES:
            if drop_input and metric == "input_wait":
                continue
            value = step_time if metric == "step_time" else _BASE_MS[metric]
            row[_EVENT_NAMES[metric]] = _stat(value)
        for step in range(_WINDOW):
            target[rank][step] = dict(row)

    per_rank = {0: {}, 1: {}}
    fill(per_rank, 0, 100.0, False)
    fill(per_rank, 1, 400.0, slow_rank_lacks_input)
    window = window_from_events(per_rank, max_rows=_WINDOW)

    assert_metric_worst_is_self_consistent(window)
    step_metric = next(m for m in window.metrics if m.metric == "step_time")
    assert step_metric.worst_total == pytest.approx(400.0)
    assert step_metric.worst_rank == 1


def test_self_consistency_holds_across_many_presence_patterns() -> None:
    # Fuzz a two-rank window over which metrics each rank measures every
    # step; the self-consistency invariant must hold for all of them.
    metrics = list(_EVENT_NAMES)
    checked = 0
    for drop_a, drop_b in itertools.product(
        [(), ("h2d",), ("forward",), ("input_wait",)],
        [(), ("optimizer_step",), ("backward",), ("input_wait",)],
    ):
        per_rank = {0: {}, 1: {}}
        for rank, dropped, scale in ((0, drop_a, 1.0), (1, drop_b, 1.7)):
            for step in range(_WINDOW):
                events = {}
                for metric in metrics:
                    if metric in dropped:
                        continue
                    events[_EVENT_NAMES[metric]] = _stat(
                        _BASE_MS[metric] * scale
                    )
                per_rank[rank][step] = events
        window = window_from_events(per_rank, max_rows=_WINDOW)
        assert_metric_worst_is_self_consistent(window)
        checked += 1
    assert checked == 16


# ---------------------------------------------------------------------------
# F3: typed values preserve absence
# ---------------------------------------------------------------------------


_COMPUTE_FIELD = {
    "forward": "forward_ms",
    "backward": "backward_ms",
    "optimizer_step": "optimizer_step_ms",
}


@pytest.mark.parametrize("missing", sorted(_COMPUTE_FIELD))
def test_typed_values_null_residual_when_compute_incomplete(
    missing: str,
) -> None:
    presence = _all_steps()
    del presence[missing]
    window = _window_from_presence(presence)
    values = rank_average(window, 0)

    assert getattr(values, _COMPUTE_FIELD[missing]) is None
    assert values.compute_ms is None
    assert values.residual_ms is None
    # A measured metric on the same row stays a real number.
    assert values.step_time_ms is not None


def test_typed_values_keep_measured_zero_distinct_from_absent() -> None:
    measured_zero = values_from_mapping(
        {"h2d": 0.0, "input_wait": 4.0, "step_time": 64.0}
    )
    absent = values_from_mapping({"input_wait": 4.0, "step_time": 64.0})

    assert measured_zero.h2d_ms == 0.0
    assert absent.h2d_ms is None


# ---------------------------------------------------------------------------
# Drift guard: a new metric forces a classification decision
# ---------------------------------------------------------------------------


def test_metric_partition_is_exhaustive_and_intended() -> None:
    # Every selected metric is classified exactly once. Occurrence-driven
    # metrics are the closed set below; adding a new selected metric
    # without deciding its class fails here (the single-source guard).
    assert OCCURRENCE_METRICS <= set(SELECTED_METRICS)
    assert OCCURRENCE_METRICS == {"optimizer_step", "h2d"}
    assert set(_REQUIRED_METRICS) == set(SELECTED_METRICS) - OCCURRENCE_METRICS
    # Every required metric has a partial-presence test above via
    # parametrization over _REQUIRED_METRICS, so a new required metric is
    # automatically covered by test_required_metric_partial_presence.
    assert set(_REQUIRED_METRICS) == {
        "input_wait",
        "forward",
        "backward",
        "step_time",
    }
