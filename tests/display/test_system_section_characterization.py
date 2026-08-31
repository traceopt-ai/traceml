# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""What the System card decides today, pinned before it is moved.

Part 5a moves metric interpretation out of `system_section.py` and into the
compute layer. These tests exist so that move is provably behaviour
preserving: they describe the CURRENT rules, including the ones that look
wrong, so a refactor that changes any of them fails loudly instead of
quietly.

Where a rule is odd, the test says so in its docstring rather than
asserting a nicer rule that the code does not implement. Two of them are
odd enough to be worth deciding about separately, and both are recorded on
the issue rather than fixed here:

* the outlier split marks the FASTER GPU on a two-GPU host,
* the card holds two different definitions of "this GPU never reported",
  one of which disagrees with the compute layer's.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui_sections import (  # noqa: E402
    system_section,
)
from traceml_ai.renderers.system import dashboard_compute  # noqa: E402


@pytest.fixture
def process_free_db(tmp_path):
    """An empty system database, computed through the real compute layer."""
    import sqlite3

    from traceml_ai.aggregator.sqlite_writers.system import init_schema
    from traceml_ai.renderers.system.dashboard_compute import (
        SystemDashboardComputer,
    )

    path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(path)
    init_schema(conn)
    conn.commit()
    conn.close()
    return SystemDashboardComputer(db_path=str(path)).compute()


def _gpu(idx: int, **kw):
    row = {
        "gpu_idx": idx,
        "util_p50": None,
        "util_now": None,
        "mem_total": None,
        "power": None,
        "reported": False,
    }
    row.update(kw)
    return row


# --- the outlier split, now owned by the compute layer -------------------
# The rule moved out of the card in part 5a-ii. Its assertions are
# unchanged, which is the point: the move preserved behaviour exactly.
def test_one_busy_gpu_among_four_is_the_marked_one():
    gpus = [_gpu(i, util_p50=u) for i, u in enumerate([100.0, 2.0, 2.0, 2.0])]
    assert dashboard_compute._odd_gpus(gpus) == [0]


def test_one_idle_gpu_among_four_is_the_marked_one():
    """The smaller group is marked whichever side of the split it is on."""
    gpus = [_gpu(i, util_p50=u) for i, u in enumerate([98.0, 98.0, 98.0, 3.0])]
    assert dashboard_compute._odd_gpus(gpus) == [3]


def test_a_two_gpu_host_marks_the_faster_one():
    """The tie rule, and it is worth stating plainly.

    With two GPUs one always lands above the midpoint and one below, so the
    groups are the same size and the tie goes to the higher group. On a
    two-GPU host the busier card is therefore always the one highlighted,
    even when the interesting one is the card that fell behind.
    """
    gpus = [_gpu(0, util_p50=95.0), _gpu(1, util_p50=10.0)]
    assert dashboard_compute._odd_gpus(gpus) == [0]


def test_evenly_loaded_gpus_mark_nothing():
    gpus = [_gpu(i, util_p50=50.0) for i in range(4)]
    assert dashboard_compute._odd_gpus(gpus) == []


def test_a_single_gpu_marks_nothing():
    assert dashboard_compute._odd_gpus([_gpu(0, util_p50=99.0)]) == []


def test_the_split_falls_back_to_the_instantaneous_reading():
    """`util_p50` is preferred; `util_now` is used when it is absent."""
    gpus = [_gpu(0, util_now=100.0), _gpu(1, util_now=1.0)]
    assert dashboard_compute._odd_gpus(gpus) == [0]


def test_a_gpu_with_no_utilisation_at_all_is_not_in_either_group():
    gpus = [_gpu(0, util_p50=90.0), _gpu(1, util_p50=5.0), _gpu(2)]
    assert 2 not in dashboard_compute._odd_gpus(gpus)


# --- the two definitions of "never reported" -----------------------------
def test_the_card_asks_the_computer_whether_a_gpu_reported():
    """Updated deliberately: the computer's rule replaced the card's.

    The card used to derive this itself, asking whether `mem_total` and
    `power` were both absent. That is stricter than the computer's rule,
    which asks whether `mem_total` or `power_limit_w` is present, so the
    two disagreed about a GPU carrying a power limit and no memory total.
    The computer decides what a metric means, so its answer is the one
    that survives and the card now reads the flag.

    The consequence is visible: a row with no values but `reported` true
    is a device that spoke and had nothing to say, and it no longer makes
    the card announce that the whole GPU sample is missing.
    """
    none_reported = [_gpu(0), _gpu(1)]
    assert system_section.gpus_unreported(none_reported) is True

    one_reported = [_gpu(0, reported=True), _gpu(1)]
    assert system_section.gpus_unreported(one_reported) is False

    # The case the two rules disagreed about: values absent, device present.
    valueless_but_present = [_gpu(0, reported=True)]
    assert system_section.gpus_unreported(valueless_but_present) is False


def test_no_gpus_at_all_is_not_the_same_as_unreported():
    assert system_section.gpus_unreported([]) is False


# --- one rule for "is this the whole-run view" ---------------------------
def test_both_charts_answer_the_whole_run_question_the_same_way():
    """The card asked two different questions and could disagree.

    It tested `len(t) > 2` of the CPU history and merely non-empty of the
    power history, so on a run with one or two power buckets the power
    chart claimed the whole-run view while the CPU chart did not, and the
    pair stopped sharing a clock.
    """
    two_points = {"t": [1.0, 2.0], "avg": [1.0, 2.0]}
    three_points = {"t": [1.0, 2.0, 3.0], "avg": [1.0, 2.0, 3.0]}
    assert dashboard_compute._has_whole_run(two_points) is False
    assert dashboard_compute._has_whole_run(three_points) is True

    # The bucket list form answers the same question the same way.
    assert dashboard_compute._has_whole_run([{"t": [1.0, 2.0]}]) is False
    assert dashboard_compute._has_whole_run([{"t": [1.0, 2.0, 3.0]}]) is True
    assert dashboard_compute._has_whole_run([]) is False
    assert dashboard_compute._has_whole_run({}) is False


def test_a_short_run_puts_neither_chart_in_the_whole_run_view(process_free_db):
    """End to end: the flags travel on the payload and agree."""
    out = process_free_db
    assert out["series"]["cpu_run_whole"] is False
    assert out["series"]["power_run_whole"] is False
