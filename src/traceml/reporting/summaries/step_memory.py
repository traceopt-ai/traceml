"""Compatibility entry point for the step-memory final summary."""

from __future__ import annotations

from typing import Any, Dict

from traceml.reporting.sections.output import persist_section_summary
from traceml.reporting.sections.step_memory import StepMemorySummarySection
from traceml.reporting.sections.step_memory.model import (
    MAX_SUMMARY_WINDOW_ROWS,
)


def generate_step_memory_summary_card(
    db_path: str,
    *,
    window_size: int = 400,
    print_to_stdout: bool = True,
) -> Dict[str, Any]:
    """Generate and persist the end-of-run step-memory summary."""
    result = StepMemorySummarySection(window_size=window_size).build(db_path)
    summary = result.payload

    persist_section_summary(
        db_path,
        section_name="step_memory",
        text=result.text,
        payload=summary,
    )

    if print_to_stdout:
        print(result.text)

    return summary


__all__ = [
    "MAX_SUMMARY_WINDOW_ROWS",
    "generate_step_memory_summary_card",
]
