"""Final end-of-run report orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from traceml.core.summaries import SummaryResult
from traceml.loggers.error_log import get_error_logger
from traceml.reporting.sections.base import SummarySection
from traceml.reporting.sections.process import ProcessSummarySection
from traceml.reporting.sections.step_memory import StepMemorySummarySection
from traceml.reporting.sections.step_time import StepTimeSummarySection
from traceml.reporting.sections.system import SystemSummarySection
from traceml.reporting.summaries.summary_layout import (
    border,
    indented_block,
    row,
    wrap_lines,
)
from traceml.sdk.protocol import (
    get_final_summary_json_path,
    get_final_summary_txt_path,
    utc_now_iso,
    write_json_atomic,
    write_text_atomic,
)

SUMMARY_WIDTH = 78
SUMMARY_INNER_TEXT_WIDTH = SUMMARY_WIDTH - 4


def _log_final_report_error(message: str, exc: Exception) -> None:
    """Log final-report failures without raising from cleanup paths."""
    try:
        get_error_logger("FinalReport").exception("[TraceML] %s", message)
    except Exception:
        pass


@dataclass(frozen=True)
class FunctionSummarySection:
    """Summary section backed by a function."""

    name: str
    builder: Callable[..., Dict[str, Any]]

    def build(self, db_path: str) -> SummaryResult:
        """Build one summary section from the SQLite database path."""
        payload = self.builder(db_path, print_to_stdout=False)
        return SummaryResult(
            section=self.name,
            payload=dict(payload or {}),
            text=str((payload or {}).get("card", "")),
        )


def _summary_duration_s(*sections: Dict[str, Any]) -> Optional[float]:
    """Pick the first valid duration from the available sections."""
    for section in sections:
        if not isinstance(section, dict):
            continue
        value = section.get("duration_s")
        if value is None:
            value = section.get("overview", {}).get("duration_s")
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


def _fallback_section_result(section: SummarySection) -> SummaryResult:
    """Return a stable section payload when one section fails."""
    name = str(getattr(section, "name", "unknown"))
    return SummaryResult(
        section=name,
        payload={
            "status": "NO DATA",
            "error": "Section summary unavailable.",
            "card": f"TraceML {name.replace('_', ' ').title()} Summary\n- Status: unavailable",
        },
        text=f"TraceML {name.replace('_', ' ').title()} Summary\n- Status: unavailable",
    )


@dataclass(frozen=True)
class FinalReportGenerator:
    """
    Build TraceML's final summary payload from registered sections.

    ``sections`` is ordered because final-summary text and JSON key order are
    intentionally stable for users and compare tooling.
    """

    sections: Sequence[SummarySection]

    def generate(self, db_path: str) -> Dict[str, Any]:
        """
        Generate the structured final summary payload.

        Section failures are isolated and logged. A failed section contributes a
        small ``NO DATA`` payload so one broken report domain does not prevent
        artifact generation during aggregator shutdown.
        """
        results: Dict[str, SummaryResult] = {}
        for section in self.sections:
            try:
                result = section.build(db_path)
            except Exception as exc:
                _log_final_report_error(
                    f"Final summary section failed: {getattr(section, 'name', section)}",
                    exc,
                )
                result = _fallback_section_result(section)
            results[result.section] = result

        system_summary = dict(
            results.get("system", SummaryResult("system")).payload
        )
        process_summary = dict(
            results.get("process", SummaryResult("process")).payload
        )
        step_time_summary = dict(
            results.get("step_time", SummaryResult("step_time")).payload
        )
        step_memory_summary = dict(
            results.get("step_memory", SummaryResult("step_memory")).payload
        )

        final_text = _build_final_summary_text_from_sections(
            system_summary=system_summary,
            process_summary=process_summary,
            step_time_summary=step_time_summary,
            step_memory_summary=step_memory_summary,
        )

        return {
            "schema_version": 1.2,
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


DEFAULT_FINAL_REPORT_GENERATOR = FinalReportGenerator(
    sections=(
        SystemSummarySection(),
        ProcessSummarySection(),
        StepTimeSummarySection(),
        StepMemorySummarySection(),
    )
)


def build_summary_payload(
    db_path: str,
    *,
    generator: FinalReportGenerator = DEFAULT_FINAL_REPORT_GENERATOR,
) -> Dict[str, Any]:
    """
    Build the structured final summary payload for one session database.
    """
    return generator.generate(db_path)


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


__all__ = [
    "DEFAULT_FINAL_REPORT_GENERATOR",
    "FinalReportGenerator",
    "FunctionSummarySection",
    "ProcessSummarySection",
    "StepMemorySummarySection",
    "StepTimeSummarySection",
    "SystemSummarySection",
    "build_summary_payload",
    "generate_summary",
    "write_summary_artifacts",
]
