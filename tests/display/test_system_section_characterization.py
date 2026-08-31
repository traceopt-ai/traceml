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


def _gpu(idx: int, **kw):
    row = {
        "gpu_idx": idx,
        "util_p50": None,
        "util_now": None,
        "mem_total": None,
        "power": None,
    }
    row.update(kw)
    return row


# --- the outlier split ---------------------------------------------------
def test_one_busy_gpu_among_four_is_the_marked_one():
    gpus = [_gpu(i, util_p50=u) for i, u in enumerate([100.0, 2.0, 2.0, 2.0])]
    assert system_section.odd_ones_out(gpus) == {0}


def test_one_idle_gpu_among_four_is_the_marked_one():
    """The smaller group is marked whichever side of the split it is on."""
    gpus = [_gpu(i, util_p50=u) for i, u in enumerate([98.0, 98.0, 98.0, 3.0])]
    assert system_section.odd_ones_out(gpus) == {3}


def test_a_two_gpu_host_marks_the_faster_one():
    """The tie rule, and it is worth stating plainly.

    With two GPUs one always lands above the midpoint and one below, so the
    groups are the same size and the tie goes to the higher group. On a
    two-GPU host the busier card is therefore always the one highlighted,
    even when the interesting one is the card that fell behind.
    """
    gpus = [_gpu(0, util_p50=95.0), _gpu(1, util_p50=10.0)]
    assert system_section.odd_ones_out(gpus) == {0}


def test_evenly_loaded_gpus_mark_nothing():
    gpus = [_gpu(i, util_p50=50.0) for i in range(4)]
    assert system_section.odd_ones_out(gpus) == set()


def test_a_single_gpu_marks_nothing():
    assert system_section.odd_ones_out([_gpu(0, util_p50=99.0)]) == set()


def test_the_split_falls_back_to_the_instantaneous_reading():
    """`util_p50` is preferred; `util_now` is used when it is absent."""
    gpus = [_gpu(0, util_now=100.0), _gpu(1, util_now=1.0)]
    assert system_section.odd_ones_out(gpus) == {0}


def test_a_gpu_with_no_utilisation_at_all_is_not_in_either_group():
    gpus = [_gpu(0, util_p50=90.0), _gpu(1, util_p50=5.0), _gpu(2)]
    assert 2 not in system_section.odd_ones_out(gpus)


# --- the two definitions of "never reported" -----------------------------
def test_the_card_calls_a_gpu_unreported_only_when_both_fields_are_absent():
    """The card's rule is `mem_total is None AND power is None`.

    The compute layer answers the same question with OR over `mem_total`
    and `power_limit_w`. The two therefore disagree about a GPU that
    reports one field and not the other, and this test pins the card's
    side of that disagreement so moving the rule has to confront it.
    """
    both_absent = [_gpu(0), _gpu(1)]
    assert system_section.gpus_unreported(both_absent) is True

    power_only = [_gpu(0, power=42.0), _gpu(1, power=41.0)]
    assert system_section.gpus_unreported(power_only) is False

    mem_only = [_gpu(0, mem_total=16.0e9)]
    assert system_section.gpus_unreported(mem_only) is False


def test_no_gpus_at_all_is_not_the_same_as_unreported():
    assert system_section.gpus_unreported([]) is False
