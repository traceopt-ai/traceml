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

from typing import Any, Dict, List, Optional

from traceml.aggregator.summaries.process import generate_process_summary_card
from traceml.aggregator.summaries.step_time import (
    generate_step_time_summary_card,
)
from traceml.aggregator.summaries.system import generate_system_summary_card


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


def _border(width: int = 78) -> str:
    """Return a fixed-width outer border."""
    return "+" + "-" * (width - 2) + "+"


def _row(text: str = "", width: int = 78) -> str:
    """Return one padded row inside the outer border."""
    inner_width = width - 4
    return f"|  {text:<{inner_width}}|"


def _indented_block(text: str) -> List[str]:
    """
    Convert a multiline text block into rows suitable for the final summary.

    Empty lines are preserved as blank rows.
    """
    return [line.rstrip() for line in str(text).splitlines()]


def _build_final_summary_text(
    *,
    system_summary: Dict[str, Any],
    process_summary: Dict[str, Any],
    step_time_summary: Dict[str, Any],
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
        _border(),
        _row(
            "TraceML Run Summary"
            + (
                f" | duration {duration_s:.1f}s"
                if duration_s is not None
                else ""
            )
        ),
        _border(),
        _row(),
        _row("System"),
    ]

    for line in _indented_block(system_summary.get("card", "")):
        if line.startswith("TraceML System Summary"):
            continue
        if line == "System":
            continue
        lines.append(_row(line))

    lines.extend(
        [
            _row(),
            _row("Process"),
        ]
    )

    for line in _indented_block(process_summary.get("card", "")):
        if line.startswith("TraceML Process Summary"):
            continue
        if line == "Process":
            continue
        lines.append(_row(line))

    lines.extend(
        [
            _row(),
            _row("Step Time"),
        ]
    )

    for line in _indented_block(step_time_summary.get("card", "")):
        if line.startswith("TraceML Step Timing Summary"):
            continue
        if line == "Step Time":
            continue
        lines.append(_row(line))

    lines.append(_border())
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

    final_text = _build_final_summary_text(
        system_summary=system_summary,
        process_summary=process_summary,
        step_time_summary=step_time_summary,
    )
    print(final_text)
