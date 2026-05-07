"""Compatibility entry point for the process final summary."""

from __future__ import annotations

from typing import Any, Dict, Optional

from traceml.reporting.sections.output import persist_section_summary
from traceml.reporting.sections.process import ProcessSummarySection
from traceml.reporting.sections.process.model import MAX_SUMMARY_ROWS


def generate_process_summary_card(
    db_path: str,
    *,
    rank: Optional[int] = None,
    print_to_stdout: bool = True,
    max_process_rows: int = MAX_SUMMARY_ROWS,
) -> Dict[str, Any]:
    """Generate and persist the end-of-run process summary."""
    result = ProcessSummarySection(
        rank=rank,
        max_process_rows=max_process_rows,
    ).build(db_path)
    summary = result.payload

    persist_section_summary(
        db_path,
        section_name="process",
        text=result.text,
        payload=summary,
    )

    if print_to_stdout:
        print(result.text)

    return summary


__all__ = [
    "MAX_SUMMARY_ROWS",
    "generate_process_summary_card",
]
