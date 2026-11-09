# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Live Step Memory aligns ranks on the latest steps they all completed."""

from __future__ import annotations

from typing import Mapping, Sequence

from tests.sqlite_fixtures import (
    init_summary_schema,
    insert_step_memory_sample,
    sqlite_database,
)
from traceml_ai.renderers.step_memory.common import (
    StepMemoryMetricsDB,
    build_step_memory_combined_result,
)


def _write_steps(path: str, steps_by_rank: Mapping[int, Sequence[int]]):
    world_size = len(steps_by_rank)
    with sqlite_database(path, init_summary_schema) as conn:
        row_id = 0
        for rank, steps in sorted(steps_by_rank.items()):
            for step in steps:
                row_id += 1
                insert_step_memory_sample(
                    conn,
                    row_id=row_id,
                    rank=rank,
                    step=step,
                    alloc=100.0 + rank,
                    reserved=200.0 + rank,
                    world_size=world_size,
                    local_world_size=world_size,
                )


def _steps_per_metric(path: str, window_size: int) -> dict:
    db = StepMemoryMetricsDB(path)
    with db.connect() as conn:
        result = build_step_memory_combined_result(
            conn, db=db, window_size=window_size
        )
    return {m.metric: m.series.steps for m in result.metrics}


def test_window_skips_a_step_one_rank_missed(tmp_path) -> None:
    path = str(tmp_path / "telemetry")
    _write_steps(
        path,
        {
            0: range(10),
            1: [step for step in range(10) if step != 7],
        },
    )

    assert _steps_per_metric(path, window_size=4) == {
        "peak_allocated": [5, 6, 8, 9],
        "peak_reserved": [5, 6, 8, 9],
    }


def test_window_stops_at_the_slowest_ranks_last_step(tmp_path) -> None:
    path = str(tmp_path / "telemetry")
    _write_steps(path, {0: range(13), 1: range(10)})

    assert _steps_per_metric(path, window_size=3) == {
        "peak_allocated": [7, 8, 9],
        "peak_reserved": [7, 8, 9],
    }


def test_window_is_short_when_fewer_common_steps_exist(tmp_path) -> None:
    path = str(tmp_path / "telemetry")
    _write_steps(path, {0: [2, 3, 4], 1: [3, 4]})

    assert _steps_per_metric(path, window_size=10) == {
        "peak_allocated": [3, 4],
        "peak_reserved": [3, 4],
    }
