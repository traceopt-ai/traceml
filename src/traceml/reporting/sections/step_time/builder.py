"""
Payload builder for the final-report step-time section.
"""

from __future__ import annotations

from typing import Any, Dict

from traceml.reporting.sections.step_time.loader import StepTimeSectionData
from traceml.reporting.summaries.step_time import _build_step_time_card


def build_step_time_section_payload(
    data: StepTimeSectionData,
) -> Dict[str, Any]:
    """
    Build the JSON-safe step-time section payload from loaded data.
    """
    _, payload = _build_step_time_card(
        training_steps=data.training_steps,
        latest_step_observed=data.latest_step_observed,
        per_rank_summary=data.per_rank_summary,
        per_rank_step_metrics=data.per_rank_step_metrics,
        max_rows=data.max_rows,
    )
    return payload


__all__ = [
    "build_step_time_section_payload",
]
