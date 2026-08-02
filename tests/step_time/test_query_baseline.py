# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""SQLite query topology for Step Time consumers.

These assertions enforce rank-independent repository statement counts. They
are architecture contracts, not wall-clock performance budgets.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest

from tests.step_time.scenarios import (
    BALANCED_PROFILE,
    SQLiteSelectRecorder,
    StepTimeScenario,
    create_step_time_database,
)
from traceml_ai.renderers.model_diagnostics.renderer import (
    ModelDiagnosticsRenderer,
)
from traceml_ai.renderers.step_time.compute import StepCombinedComputer
from traceml_ai.renderers.step_time.renderer import StepCombinedRenderer
from traceml_ai.reporting.sections.step_time import StepTimeSummarySection


@pytest.fixture(params=(1, 8, 32), ids=lambda ranks: f"{ranks}-ranks")
def ranked_db(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> tuple[int, Path]:
    """Create a complete four-step CPU window at the requested rank count."""
    rank_count = int(request.param)
    scenario = StepTimeScenario(
        name=f"query_{rank_count}_ranks",
        profiles={rank: dict(BALANCED_PROFILE) for rank in range(rank_count)},
        steps=(1, 2, 3, 4),
    )
    db_path = tmp_path / f"query-{rank_count}.db"
    create_step_time_database(db_path, scenario)
    return rank_count, db_path


def _record_selects(call: Callable[[], object]) -> SQLiteSelectRecorder:
    """Run one production callback with test-side SQLite tracing enabled."""
    recorder = SQLiteSelectRecorder()
    with patch.object(sqlite3, "connect", recorder.connect):
        call()
    return recorder


def test_live_query_count_is_constant_two(
    ranked_db: tuple[int, Path],
) -> None:
    """One set-based source read plus one strategy read serves all ranks."""
    _, db_path = ranked_db
    computer = StepCombinedComputer(str(db_path), window_size=4)

    recorder = _record_selects(computer.compute_cli)

    assert recorder.count == 2


def test_dashboard_query_count_is_constant_four(
    ranked_db: tuple[int, Path],
) -> None:
    """The two current providers each perform one constant-cost load."""
    _, db_path = ranked_db
    hero = StepCombinedRenderer(str(db_path))
    diagnostics = ModelDiagnosticsRenderer(str(db_path))

    recorder = _record_selects(
        lambda: (
            hero.get_dashboard_renderable(),
            diagnostics._step_time.compute_dashboard(),
        )
    )

    assert recorder.count == 4


def test_summary_query_count_is_constant_two(
    ranked_db: tuple[int, Path],
) -> None:
    """Summary receives progress and identities with its source snapshot."""
    _, db_path = ranked_db
    section = StepTimeSummarySection(max_rows=4)

    recorder = _record_selects(lambda: section.build(str(db_path)))

    assert recorder.count == 2
