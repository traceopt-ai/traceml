"""SQLite loader for the final-report step-time section."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Optional

from traceml.reporting.summaries.step_time import (
    MAX_SUMMARY_WINDOW_ROWS,
    RankStepSummary,
    _build_rank_summary,
    _load_rank_step_rows,
)


@dataclass(frozen=True)
class StepTimeSectionData:
    """Loaded inputs for the step-time final-report section."""

    training_steps: int
    latest_step_observed: Optional[int]
    per_rank_summary: Dict[int, RankStepSummary]
    per_rank_step_metrics: Dict[int, Dict[int, Dict[str, float]]]
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

        rank_rows = conn.execute(
            "SELECT DISTINCT rank FROM step_time_samples ORDER BY rank ASC;"
        ).fetchall()
        ranks_present = [int(r[0]) for r in rank_rows if r[0] is not None]

        per_rank_summary: Dict[int, RankStepSummary] = {}
        per_rank_step_metrics: Dict[int, Dict[int, Dict[str, float]]] = {}

        for rank in ranks_present:
            step_rows = _load_rank_step_rows(
                conn,
                rank=rank,
                max_rows=row_limit,
            )
            analysis = _build_rank_summary(step_rows)
            if analysis is not None:
                per_rank_summary[rank] = analysis.summary
                per_rank_step_metrics[rank] = analysis.per_step_metrics
    finally:
        conn.close()

    return StepTimeSectionData(
        training_steps=training_steps,
        latest_step_observed=latest_step_observed,
        per_rank_summary=per_rank_summary,
        per_rank_step_metrics=per_rank_step_metrics,
        max_rows=row_limit,
    )


__all__ = [
    "StepTimeSectionData",
    "load_step_time_section_data",
]
