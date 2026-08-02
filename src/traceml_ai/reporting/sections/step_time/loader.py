# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""SQLite loader for the final-report step-time section."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from traceml_ai.reporting.config import normalize_summary_window_rows
from traceml_ai.reporting.sections.step_time.model import (
    MAX_SUMMARY_WINDOW_ROWS,
    GlobalRankIdentity,
    RankStepSummary,
    rank_summaries_from_window,
)
from traceml_ai.step_time.model import StepTimeLoadRequest, StepTimeWindow
from traceml_ai.step_time.sqlite import SQLiteStepTimeRepository
from traceml_ai.utils.step_time_sqlite import (
    load_step_time_summary_from_sqlite,
)


@dataclass(frozen=True)
class StepTimeSectionData:
    """Loaded inputs for the step-time final-report section."""

    training_steps: int
    latest_step_observed: Optional[int]
    step_time_window: StepTimeWindow
    per_global_rank_summary: Dict[int, RankStepSummary]
    identities: Dict[int, GlobalRankIdentity]
    max_rows: int
    training_strategy: str = "ddp"


def load_global_rank_identities(
    conn: sqlite3.Connection,
    global_ranks: Iterable[int],
) -> Dict[int, GlobalRankIdentity]:
    """Compatibility adapter returning repository-owned rank identities."""
    requested = tuple(sorted({int(rank) for rank in global_ranks}))
    if not requested:
        return {}
    snapshot = SQLiteStepTimeRepository(conn).load_summary(
        StepTimeLoadRequest(window_size=1, rank_filter=requested)
    )
    return dict(snapshot.identities)


def load_step_time_section_data(
    db_path: str,
    *,
    max_rows: int = MAX_SUMMARY_WINDOW_ROWS,
) -> StepTimeSectionData:
    """
    Load final-report Step Time data from SQLite.

    The repository-backed window loader returns timing rows, identities,
    progress, and run context from one consistent read snapshot. Reporting
    only projects those source facts into its stable public schema.
    """
    row_limit = normalize_summary_window_rows(max_rows)
    conn = sqlite3.connect(db_path)

    try:
        loaded = load_step_time_summary_from_sqlite(
            conn,
            max_rows=row_limit,
        )
        latest_step_observed = loaded.cursor.latest_step
        training_steps = (
            latest_step_observed + 1 if latest_step_observed is not None else 0
        )
        identities = dict(loaded.identities)
        step_time_window = loaded.window
        selected_summary = rank_summaries_from_window(step_time_window)
    finally:
        conn.close()

    return StepTimeSectionData(
        training_steps=training_steps,
        latest_step_observed=latest_step_observed,
        step_time_window=step_time_window,
        per_global_rank_summary=selected_summary,
        identities=identities,
        max_rows=row_limit,
        training_strategy=loaded.training_strategy,
    )


__all__ = [
    "StepTimeSectionData",
    "load_global_rank_identities",
    "load_step_time_section_data",
]
