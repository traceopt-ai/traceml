"""Compatibility entry point for the step-time final summary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from traceml_ai.reporting.config import DEFAULT_SUMMARY_WINDOW_ROWS
from traceml_ai.reporting.sections.output import persist_section_summary
from traceml_ai.reporting.sections.step_time import StepTimeSummarySection

MAX_SUMMARY_WINDOW_ROWS = DEFAULT_SUMMARY_WINDOW_ROWS


@dataclass
class RankStepSummary:
    """Deprecated public container retained for one compatibility release.

    The built-in summary no longer constructs this intermediate; it projects
    canonical ``StepTimeRankFacts`` directly. External integrations can keep
    importing the released type while they migrate to ``StepTimeValues``.
    """

    steps_analyzed: int
    avg_dataloader_ms: Optional[float]
    avg_input_wait_ms: Optional[float]
    avg_step_time_ms: Optional[float]
    avg_h2d_ms: Optional[float]
    avg_forward_ms: Optional[float]
    avg_backward_ms: Optional[float]
    avg_optimizer_ms: Optional[float]
    avg_traced_step_ms: Optional[float]
    avg_compute_ms: Optional[float]
    avg_residual_ms: Optional[float]
    avg_total_step_ms: Optional[float]


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
