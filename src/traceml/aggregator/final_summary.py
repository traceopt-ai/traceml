"""
Final end-of-run summary orchestration for TraceML.

This module is responsible for building the user-facing end-of-run summary from
individual summary generators.

Design goals
------------
- Print exactly once at shutdown
- Keep section layout consistent across runs
- Let per-domain summary modules compute and serialize their own payloads
- Keep the printed summary compact and shareable
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from traceml.aggregator.summaries.process import generate_process_summary_card
from traceml.aggregator.summaries.step_memory import (
    generate_step_memory_summary_card,
)
from traceml.aggregator.summaries.step_time import (
    generate_step_time_summary_card,
)
from traceml.aggregator.summaries.summary_layout import (
    border,
    indented_block,
    row,
    wrap_lines,
)
from traceml.aggregator.summaries.system import generate_system_summary_card

SUMMARY_WIDTH = 78
SUMMARY_INNER_TEXT_WIDTH = SUMMARY_WIDTH - 4


def _summary_duration_s(*sections: Dict[str, Any]) -> Optional[float]:
    """
    Pick the first valid duration from the available summary sections.
    """
    for section in sections:
        if not isinstance(section, dict):
            continue
        value = section.get("duration_s")
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _append_wrapped_card_lines(
    lines: List[str],
    card_text: str,
    *,
    section_title: str,
    card_header_prefix: str,
) -> None:
    """
    Append wrapped summary card lines into the final combined summary.

    Parameters
    ----------
    lines:
        Mutable list of already-rendered boxed rows.
    card_text:
        Raw multiline card text returned by a summary builder.
    section_title:
        Section heading used in the final combined summary. Matching inner
        headings are skipped to avoid duplication.
    card_header_prefix:
        Prefix of the inner card header line to skip, for example
        'TraceML Step Timing Summary'.
    """
    for line in indented_block(card_text):
        if line.startswith(card_header_prefix):
            continue
        if line == section_title:
            continue

        for wrapped in wrap_lines(line, SUMMARY_INNER_TEXT_WIDTH):
            lines.append(row(wrapped, width=SUMMARY_WIDTH))


def _build_final_summary_text(
    *,
    system_summary: Dict[str, Any],
    process_summary: Dict[str, Any],
    step_time_summary: Dict[str, Any],
    step_memory_summary: Dict[str, Any],
) -> str:
    """
    Build the single printed end-of-run summary.

    This intentionally uses one outer boundary and compact inner sections to
    keep the output easy to scan and easy to paste into issues or chat.
    """
    duration_s = _summary_duration_s(
        step_time_summary,
        process_summary,
        system_summary,
    )

    lines: List[str] = [
        border(width=SUMMARY_WIDTH),
        row(
            "TraceML Run Summary"
            + (
                f" | duration {duration_s:.1f}s"
                if duration_s is not None
                else ""
            ),
            width=SUMMARY_WIDTH,
        ),
        border(width=SUMMARY_WIDTH),
        row(width=SUMMARY_WIDTH),
        row("System", width=SUMMARY_WIDTH),
    ]

    _append_wrapped_card_lines(
        lines,
        system_summary.get("card", ""),
        section_title="System",
        card_header_prefix="TraceML System Summary",
    )

    lines.extend(
        [
            row(width=SUMMARY_WIDTH),
            row("Process", width=SUMMARY_WIDTH),
        ]
    )

    _append_wrapped_card_lines(
        lines,
        process_summary.get("card", ""),
        section_title="Process",
        card_header_prefix="TraceML Process Summary",
    )

    lines.extend(
        [
            row(width=SUMMARY_WIDTH),
            row("Step Time", width=SUMMARY_WIDTH),
        ]
    )

    _append_wrapped_card_lines(
        lines,
        step_time_summary.get("card", ""),
        section_title="Step Time",
        card_header_prefix="TraceML Step Timing Summary",
    )

    lines.extend(
        [
            row(width=SUMMARY_WIDTH),
            row("Step Memory", width=SUMMARY_WIDTH),
        ]
    )

    _append_wrapped_card_lines(
        lines,
        step_memory_summary.get("card", ""),
        section_title="Step Memory",
        card_header_prefix="TraceML Step Memory Summary",
    )

    lines.append(border(width=SUMMARY_WIDTH))
    return "\n".join(lines)


def generate_summary(db_path: str) -> None:
    """
    Generate and print the final end-of-run summary.

    Parameters
    ----------
    db_path:
        Path to the session SQLite database file.

    Notes
    -----
    Individual summary modules still write their own JSON/text artifacts.
    This function is responsible only for orchestrating and printing the final
    compact combined summary once.
    """
    system_summary = generate_system_summary_card(
        db_path,
        print_to_stdout=False,
    )
    process_summary = generate_process_summary_card(
        db_path,
        print_to_stdout=False,
    )
    step_time_summary = generate_step_time_summary_card(
        db_path,
        print_to_stdout=False,
    )
    step_memory_summary = generate_step_memory_summary_card(
        db_path,
        print_to_stdout=False,
    )

    final_text = _build_final_summary_text(
        system_summary=system_summary,
        process_summary=process_summary,
        step_time_summary=step_time_summary,
        step_memory_summary=step_memory_summary,
    )
    print(final_text)
