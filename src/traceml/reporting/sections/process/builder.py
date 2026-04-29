"""
Payload builder for the final-report process section.
"""

from __future__ import annotations

from typing import Any, Dict

from traceml.reporting.sections.process.loader import ProcessSectionData
from traceml.reporting.summaries.process import _build_process_card


def build_process_section_payload(
    data: ProcessSectionData,
) -> Dict[str, Any]:
    """
    Build the JSON-safe process-section payload from loaded data.
    """
    _, payload = _build_process_card(
        data.aggregate,
        per_rank=data.per_rank,
    )
    return payload


__all__ = [
    "build_process_section_payload",
]
