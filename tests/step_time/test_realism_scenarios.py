# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The time-varying scenarios behave as their names say.

They feed the public-schema golden, so each is pinned here by what it is
meant to represent: per-step variation that is the same on every run, a
rank that drops out mid-window, and metrics that do not occur every step.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.step_time.scenarios import (
    SCENARIOS_BY_NAME,
    create_step_time_database,
)
from traceml_ai.reporting.sections.step_time import StepTimeSummarySection
from traceml_ai.step_time.model import StepTimeLoadRequest
from traceml_ai.step_time.pipeline import LiveStepTimeSession


def _events(db_path: Path) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT global_rank, step, events_json FROM step_time_samples "
            "ORDER BY global_rank, step"
        ).fetchall()


def _summary(tmp_path: Path, name: str) -> dict:
    db_path = tmp_path / f"{name}.db"
    create_step_time_database(db_path, SCENARIOS_BY_NAME[name])
    return StepTimeSummarySection().build(str(db_path)).payload


def test_jitter_varies_per_step_and_is_the_same_on_every_run(
    tmp_path: Path,
) -> None:
    scenario = SCENARIOS_BY_NAME["jittered_ddp"]
    first, second = tmp_path / "a.db", tmp_path / "b.db"
    create_step_time_database(first, scenario)
    create_step_time_database(second, scenario)

    rows = _events(first)
    assert rows == _events(second)
    rank0 = [events for rank, _, events in rows if rank == 0]
    assert len(set(rank0)) == len(scenario.steps)


def test_a_rank_that_misses_steps_drops_them_from_the_aligned_window(
    tmp_path: Path,
) -> None:
    scenario = SCENARIOS_BY_NAME["rank_missing_steps"]
    db_path = tmp_path / "missing.db"
    create_step_time_database(db_path, scenario)

    window = (
        LiveStepTimeSession(
            str(db_path),
            request=StepTimeLoadRequest(
                window_size=len(scenario.steps), lookback_factor=4
            ),
        )
        .refresh()
        .analysis.window
    )

    missing = set(scenario.missing_steps[1])
    assert window.steps == [s for s in scenario.steps if s not in missing]
    assert window.coverage.ranks_present == 2


@pytest.mark.parametrize(
    ("metric", "rank0", "rank1"),
    [
        # Occurrence-driven: measured when it occurs, zero on other steps.
        ("optimizer_ms", 2.5, 2.5),
        ("h2d_ms", 2.5, 2.5),
        # Must occur on every aligned step: one missing step makes the
        # rank's value unknown, and everything derived from it too.
        ("forward_ms", 30.0, None),
        ("compute_ms", 77.5, None),
        ("residual_ms", 15.0, None),
        ("backward_ms", 45.0, 45.0),
    ],
)
def test_intermittent_metrics_follow_their_own_availability_rule(
    tmp_path: Path,
    metric: str,
    rank0: float | None,
    rank1: float | None,
) -> None:
    rows = _summary(tmp_path, "intermittent_metrics")["groups"]["rows"]
    for rank, expected in (("0", rank0), ("1", rank1)):
        actual = rows[rank]["metrics"][metric]
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected)
