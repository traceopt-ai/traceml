"""
Final-report system section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from traceml.core.summaries import SummaryResult
from traceml.reporting.sections.system.builder import (
    build_system_section_payload,
)
from traceml.reporting.sections.system.formatter import (
    format_system_section_text,
)
from traceml.reporting.sections.system.loader import load_system_section_data
from traceml.reporting.summaries.system import MAX_SUMMARY_ROWS


@dataclass(frozen=True)
class SystemSummarySection:
    """Build TraceML's final-report system section."""

    rank: Optional[int] = None
    max_system_rows: int = MAX_SUMMARY_ROWS
    name: str = "system"

    def build(self, db_path: str) -> SummaryResult:
        data = load_system_section_data(
            db_path,
            rank=self.rank,
            max_system_rows=self.max_system_rows,
        )
        payload = build_system_section_payload(data)
        return SummaryResult(
            section=self.name,
            payload=payload,
            text=format_system_section_text(payload),
        )


__all__ = [
    "SystemSummarySection",
    "build_system_section_payload",
    "format_system_section_text",
    "load_system_section_data",
]
