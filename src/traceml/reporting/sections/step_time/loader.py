# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""SQLite loader for the final-report step-time section."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Optional

from traceml.reporting.sections.step_time.alignment import (
    AlignedStepWindow,
    build_aligned_step_summary,
)
from traceml.reporting.sections.step_time.model import (
    MAX_SUMMARY_WINDOW_ROWS,
    GlobalRankIdentity,
    RankStepSummary,
    _build_rank_summary,
    _load_global_rank_identities,
    _load_global_rank_step_rows,
)


@dataclass(frozen=True)
class StepTimeSectionData:
    """Loaded inputs for the step-time final-report section."""

    training_steps: int
    latest_step_observed: Optional[int]
    aligned_summary: Dict[int, RankStepSummary]
    aligned_step_metrics: Dict[int, Dict[int, Dict[str, float]]]
    aligned_window: AlignedStepWindow
    per_global_rank_summary: Dict[int, RankStepSummary]
    per_global_rank_step_metrics: Dict[int, Dict[int, Dict[str, float]]]
    identities: Dict[int, GlobalRankIdentity]
    max_rows: int


def load_step_time_section_data(
    db_path: str,
    *,
    max_rows: int = MAX_SUMMARY_WINDOW_ROWS,
) -> StepTimeSectionData:
    """
    Load bounded step-time section data from the SQLite history database.
    """
    row_limit = min(max(1, int(max_rows)), MAX_SUMMARY_WINDOW_ROWS)
    conn = sqlite3.connect(db_path)

    try:
        latest_step_observed_row = conn.execute(
            "SELECT MAX(step) FROM step_time_samples;"
        ).fetchone()
        latest_step_observed = (
            int(latest_step_observed_row[0])
            if latest_step_observed_row[0] is not None
            else None
        )
        training_steps = (
            latest_step_observed + 1 if latest_step_observed is not None else 0
        )

        global_rank_rows = conn.execute(
            """
            SELECT DISTINCT global_rank
            FROM step_time_samples
            WHERE global_rank IS NOT NULL
            ORDER BY global_rank ASC;
            """
        ).fetchall()
        global_ranks_present = [
            int(row[0]) for row in global_rank_rows if row[0] is not None
        ]
        identities = _load_global_rank_identities(conn, global_ranks_present)

        per_global_rank_summary: Dict[int, RankStepSummary] = {}
        per_global_rank_step_metrics: Dict[
            int,
            Dict[int, Dict[str, float]],
        ] = {}

        for global_rank in global_ranks_present:
            step_rows = _load_global_rank_step_rows(
                conn,
                global_rank=global_rank,
                max_rows=row_limit,
            )
            analysis = _build_rank_summary(step_rows)
            if analysis is not None:
                per_global_rank_summary[global_rank] = analysis.summary
                per_global_rank_step_metrics[global_rank] = (
                    analysis.per_step_metrics
                )

        aligned_summary, aligned_step_metrics, aligned_window = (
            build_aligned_step_summary(
                per_global_rank_step_metrics=per_global_rank_step_metrics,
                max_rows=row_limit,
            )
        )
    finally:
        conn.close()

    return StepTimeSectionData(
        training_steps=training_steps,
        latest_step_observed=latest_step_observed,
        aligned_summary=aligned_summary,
        aligned_step_metrics=aligned_step_metrics,
        aligned_window=aligned_window,
        per_global_rank_summary=per_global_rank_summary,
        per_global_rank_step_metrics=per_global_rank_step_metrics,
        identities=identities,
        max_rows=row_limit,
    )


__all__ = [
    "StepTimeSectionData",
    "load_step_time_section_data",
]
