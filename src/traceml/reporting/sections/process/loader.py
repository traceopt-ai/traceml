"""SQLite loader for the final-report process section."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Optional

from traceml.reporting.summaries.process import (
    MAX_SUMMARY_ROWS,
    PerRankProcessSummary,
    ProcessSummaryAgg,
    _load_per_rank_process_summary,
    _load_process_summary_agg,
)


@dataclass(frozen=True)
class ProcessSectionData:
    """Loaded inputs for the process final-report section."""

    aggregate: ProcessSummaryAgg
    per_rank: Dict[int, PerRankProcessSummary]


def load_process_section_data(
    db_path: str,
    *,
    rank: Optional[int] = None,
    max_process_rows: int = MAX_SUMMARY_ROWS,
) -> ProcessSectionData:
    """
    Load bounded process-section data from the SQLite history database.
    """
    row_limit = min(max(1, int(max_process_rows)), MAX_SUMMARY_ROWS)
    conn = sqlite3.connect(db_path)
    try:
        aggregate = _load_process_summary_agg(
            conn,
            rank=rank,
            max_process_rows=row_limit,
        )
        per_rank = _load_per_rank_process_summary(
            conn,
            rank=rank,
            max_process_rows=row_limit,
        )
    finally:
        conn.close()

    return ProcessSectionData(aggregate=aggregate, per_rank=per_rank)


__all__ = [
    "ProcessSectionData",
    "load_process_section_data",
]
