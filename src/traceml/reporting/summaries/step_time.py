"""Compatibility entry point for the step-time final summary."""

from __future__ import annotations

from typing import Any, Dict

from traceml.reporting.sections.output import persist_section_summary
from traceml.reporting.sections.step_time import StepTimeSummarySection
from traceml.reporting.sections.step_time.model import (
    MAX_SUMMARY_WINDOW_ROWS,
    RankStepSummary,
)


def generate_step_time_summary_card(
    db_path: str,
    *,
    max_rows: int = MAX_SUMMARY_WINDOW_ROWS,
    print_to_stdout: bool = True,
) -> Dict[str, Any]:
    """Generate and persist the end-of-run step-time summary."""
    result = StepTimeSummarySection(max_rows=max_rows).build(db_path)
    summary = result.payload

    persist_section_summary(
        db_path,
        section_name="step_time",
        text=result.text,
        payload=summary,
    )

    if print_to_stdout:
        print(result.text)

    return summary


__all__ = [
    "MAX_SUMMARY_WINDOW_ROWS",
    "RankStepSummary",
    "generate_step_time_summary_card",
]
