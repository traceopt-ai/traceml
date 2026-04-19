"""
Final end-of-run summary orchestration for TraceML.

This module builds the user-facing end-of-run summary from individual summary
generators and can also write canonical summary artifacts for programmatic use.

Design goals
------------
- Print exactly once at shutdown when requested
- Keep section layout consistent across runs
- Return a clean structured payload for programmatic callers
- Write canonical final summary artifacts for user code and integrations
"""

from pathlib import Path
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
from traceml.final_summary_protocol import (
    get_final_summary_json_path,
    get_final_summary_txt_path,
    utc_now_iso,
    write_json_atomic,
    write_text_atomic,
)

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
    """
    for line in indented_block(card_text):
        if line.startswith(card_header_prefix):
            continue
        if line == section_title:
            continue

        for wrapped in wrap_lines(line, SUMMARY_INNER_TEXT_WIDTH):
            lines.append(row(wrapped, width=SUMMARY_WIDTH))


def _build_final_summary_text_from_sections(
    *,
    system_summary: Dict[str, Any],
    process_summary: Dict[str, Any],
    step_time_summary: Dict[str, Any],
    step_memory_summary: Dict[str, Any],
) -> str:
    """
    Build the single printed end-of-run summary from per-domain sections.
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


def build_summary_payload(db_path: str) -> Dict[str, Any]:
    """
    Build the structured final summary payload for one session database.

    Parameters
    ----------
    db_path:
        Path to the session SQLite database file.

    Returns
    -------
    Dict[str, Any]
        Structured summary payload suitable for writing to disk or returning to
        programmatic callers.
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

    final_text = _build_final_summary_text_from_sections(
        system_summary=system_summary,
        process_summary=process_summary,
        step_time_summary=step_time_summary,
        step_memory_summary=step_memory_summary,
    )

    return {
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "duration_s": _summary_duration_s(
            step_time_summary,
            process_summary,
            system_summary,
        ),
        "system": system_summary,
        "process": process_summary,
        "step_time": step_time_summary,
        "step_memory": step_memory_summary,
        "text": final_text,
    }


def write_summary_artifacts(
    *,
    db_path: str,
    payload: Dict[str, Any],
    session_root: Optional[str] = None,
) -> None:
    """
    Write final summary artifacts to disk.

    Artifacts written
    -----------------
    - legacy DB-adjacent artifacts for compatibility
    - canonical session-root artifacts for public API consumers
    """
    final_text = str(payload.get("text", ""))

    legacy_json_path = Path(str(db_path) + "_summary_card.json").resolve()
    legacy_txt_path = Path(str(db_path) + "_summary_card.txt").resolve()

    write_json_atomic(legacy_json_path, payload)
    write_text_atomic(legacy_txt_path, final_text + "\n")

    if session_root:
        session_root_path = Path(session_root).resolve()
        write_json_atomic(
            get_final_summary_json_path(session_root_path),
            payload,
        )
        write_text_atomic(
            get_final_summary_txt_path(session_root_path),
            final_text + "\n",
        )


def generate_summary(
    db_path: str,
    *,
    session_root: Optional[str] = None,
    print_to_stdout: bool = True,
) -> Dict[str, Any]:
    """
    Generate, write, and optionally print the final end-of-run summary.

    Parameters
    ----------
    db_path:
        Path to the session SQLite database file.
    session_root:
        Optional session root used for canonical final summary artifacts.
    print_to_stdout:
        If True, print the final combined summary text.

    Returns
    -------
    Dict[str, Any]
        Structured final summary payload.
    """
    payload = build_summary_payload(db_path)
    write_summary_artifacts(
        db_path=db_path,
        payload=payload,
        session_root=session_root,
    )

    if print_to_stdout:
        print(payload["text"])

    return payload
