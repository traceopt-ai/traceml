# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Stable SQLite query topology for the three Step Time surfaces."""

from __future__ import annotations

import sqlite3
from pathlib import Path
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
from traceml_ai.renderers.step_memory.schema import StepMemoryCombinedResult
from traceml_ai.reporting.sections.step_time import StepTimeSummarySection
from traceml_ai.step_time.model import StepTimeLoadRequest
from traceml_ai.step_time.pipeline import LiveStepTimeSession


@pytest.mark.parametrize("surface", ("live", "dashboard", "summary"))
def test_step_time_surface_uses_exactly_two_selects(
    surface: str,
    tmp_path: Path,
) -> None:
    """One set-based source read and one strategy read serve every surface."""
    db_path = tmp_path / f"{surface}.db"
    create_step_time_database(
        db_path,
        StepTimeScenario(
            name=f"query_topology_{surface}",
            profiles={rank: BALANCED_PROFILE for rank in range(8)},
            steps=(1, 2, 3, 4),
        ),
    )
    request = StepTimeLoadRequest(window_size=4, lookback_factor=4)
    session = LiveStepTimeSession(str(db_path), request=request)

    if surface == "summary":
        operation = lambda: StepTimeSummarySection().build(str(db_path))
    elif surface == "dashboard":
        diagnostics = ModelDiagnosticsRenderer(str(db_path))
        diagnostics._step_memory.compute_dashboard = lambda: (
            StepMemoryCombinedResult(
                metrics=[],
                status_message="No GPU detected",
            )
        )

        def operation() -> object:
            return diagnostics.get_dashboard_renderable(session.refresh())

    else:
        operation = session.refresh

    recorder = SQLiteSelectRecorder()
    with patch.object(sqlite3, "connect", recorder.connect):
        operation()

    assert recorder.count == 2, recorder.statements
