# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The shared Run step-timing tree read by the terminal and tracker cards."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from traceml_ai.reporting.terminal_card.step_time import (
    StepTimingRow,
    step_timing_tree,
)

_AVERAGE = {
    "step_time_ms": 100.0,
    "input_wait_ms": 60.0,
    "compute_ms": 30.0,
    "forward_ms": 10.0,
    "backward_ms": 15.0,
    "optimizer_ms": 5.0,
    "h2d_ms": None,
    "residual_ms": 10.0,
    "dataloader_fetch_cpu_ms": 58.0,
}


def _section(*, clock: str = "cpu") -> Dict[str, Any]:
    rank1 = dict(_AVERAGE, step_time_ms=120.0, input_wait_ms=80.0)
    return {
        "global": {
            "average": dict(_AVERAGE),
            "median": {"step_time_ms": {"value": 120.0, "idx": "1"}},
            "window": {"diagnosis_clock": clock},
        },
        "groups": {
            "rows": {
                "0": {
                    "identity": {"global_rank": 0, "node_rank": 0},
                    "metrics": dict(_AVERAGE, step_time_ms=90.0),
                },
                "1": {
                    "identity": {"global_rank": 1, "node_rank": 0},
                    "metrics": rank1,
                },
            }
        },
    }


def test_single_rank_tree_reads_the_window_average() -> None:
    tree = step_timing_tree({"kind": "INPUT_BOUND"}, _section(), multi=False)
    assert tree.selection == "Window Average"
    assert tree.clock == "CPU"
    assert tree.error is None
    assert tree.dataloader_fetch_ms == 58.0
    assert tree.rows == (
        StepTimingRow("Step Time", "step_time_ms", 0, "", 100.0, False),
        StepTimingRow("Input Wait", "input_wait_ms", 1, "├─ ", 60.0, True),
        StepTimingRow("Compute", "compute_ms", 1, "├─ ", 30.0, False),
        StepTimingRow("Forward", "forward_ms", 2, "│  ├─ ", 10.0, False),
        StepTimingRow("Backward", "backward_ms", 2, "│  ├─ ", 15.0, False),
        StepTimingRow("Optimizer", "optimizer_ms", 2, "│  └─ ", 5.0, False),
        StepTimingRow("Residual", "residual_ms", 1, "└─ ", 10.0, False),
    )


def test_multi_rank_tree_reads_the_median_rank_row() -> None:
    tree = step_timing_tree(
        {"kind": "COMPUTE_BOUND"}, _section(clock="gpu"), multi=True
    )
    assert tree.selection == "Median R1/N0"
    assert tree.clock == "GPU"
    values = {row.metric: row.value_ms for row in tree.rows}
    assert values["step_time_ms"] == 120.0
    assert values["input_wait_ms"] == 80.0
    assert [row.metric for row in tree.rows if row.is_cause] == ["compute_ms"]


def test_compute_children_use_a_blank_ancestor_when_compute_is_last() -> None:
    section = _section()
    section["global"]["average"]["residual_ms"] = None
    tree = step_timing_tree({}, section, multi=False)
    assert [row.glyph + row.label for row in tree.rows] == [
        "Step Time",
        "├─ Input Wait",
        "└─ Compute",
        "   ├─ Forward",
        "   ├─ Backward",
        "   └─ Optimizer",
    ]


@pytest.mark.parametrize(
    "kind",
    [
        "INPUT_STRAGGLER",
        "COMPUTE_STRAGGLER",
        "H2D_STRAGGLER",
        "STRAGGLER",
        "NO_CLEAR_PERFORMANCE_BOTTLENECK",
        "",
    ],
)
def test_no_row_is_the_cause_without_a_bound_phase(kind: str) -> None:
    tree = step_timing_tree({"kind": kind}, _section(), multi=False)
    assert tree.rows
    assert not any(row.is_cause for row in tree.rows)


def test_missing_median_rank_is_an_error_with_no_rows() -> None:
    section = _section()
    section["global"]["median"] = {}
    tree = step_timing_tree({"kind": "INPUT_BOUND"}, section, multi=True)
    assert tree.rows == ()
    assert tree.selection == "Median Rank Unavailable"
    assert tree.error == (
        "STEP TIMING unavailable: no median rank was selected."
    )
    assert tree.dataloader_fetch_ms is None


@pytest.mark.parametrize("step_time_ms", [None, 0.0])
def test_unmeasured_or_zero_step_time_draws_no_rows(step_time_ms) -> None:
    section = _section()
    section["global"]["average"]["step_time_ms"] = step_time_ms
    tree = step_timing_tree({"kind": "INPUT_BOUND"}, section, multi=False)
    assert tree.rows == ()
    assert tree.error is None
