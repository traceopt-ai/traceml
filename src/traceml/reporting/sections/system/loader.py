"""SQLite loader for the final-report system section."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Optional

from traceml.reporting.summaries.system import (
    MAX_SUMMARY_ROWS,
    PerGPUSummary,
    SystemSummaryAgg,
    _load_per_gpu_summary,
    _load_system_summary_agg,
)


@dataclass(frozen=True)
class SystemSectionData:
    """Loaded inputs for the system final-report section."""

    aggregate: SystemSummaryAgg
    per_gpu: Dict[int, PerGPUSummary]


def load_system_section_data(
    db_path: str,
    *,
    rank: Optional[int] = None,
    max_system_rows: int = MAX_SUMMARY_ROWS,
) -> SystemSectionData:
    """
    Load bounded system-section data from the SQLite history database.
    """
    row_limit = min(max(1, int(max_system_rows)), MAX_SUMMARY_ROWS)
    conn = sqlite3.connect(db_path)
    try:
        aggregate = _load_system_summary_agg(
            conn,
            rank=rank,
            max_system_rows=row_limit,
        )
        per_gpu = _load_per_gpu_summary(
            conn,
            rank=rank,
            max_system_rows=row_limit,
        )
    finally:
        conn.close()

    return SystemSectionData(aggregate=aggregate, per_gpu=per_gpu)


__all__ = [
    "SystemSectionData",
    "load_system_section_data",
]
