"""
Final-report step-time section.
"""

from __future__ import annotations

from dataclasses import dataclass

from traceml.core.summaries import SummaryResult
from traceml.reporting.sections.step_time.builder import (
    build_step_time_section_payload,
)
from traceml.reporting.sections.step_time.formatter import (
    format_step_time_section_text,
)
from traceml.reporting.sections.step_time.loader import (
    load_step_time_section_data,
)
from traceml.reporting.sections.step_time.model import MAX_SUMMARY_WINDOW_ROWS


@dataclass(frozen=True)
class StepTimeSummarySection:
    """Build TraceML's final-report step-time section."""

    max_rows: int = MAX_SUMMARY_WINDOW_ROWS
    name: str = "step_time"

    def build(self, db_path: str) -> SummaryResult:
        data = load_step_time_section_data(
            db_path,
            max_rows=self.max_rows,
        )
        payload = build_step_time_section_payload(data)
        return SummaryResult(
            section=self.name,
            payload=payload,
            text=format_step_time_section_text(payload),
        )


__all__ = [
    "StepTimeSummarySection",
    "build_step_time_section_payload",
    "format_step_time_section_text",
    "load_step_time_section_data",
]
