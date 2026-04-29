"""
Payload builder for the final-report system section.
"""

from __future__ import annotations

from typing import Any, Dict

from traceml.reporting.sections.system.loader import SystemSectionData
from traceml.reporting.summaries.system import _build_system_card


def build_system_section_payload(
    data: SystemSectionData,
) -> Dict[str, Any]:
    """
    Build the JSON-safe system-section payload from loaded data.
    """
    _, payload = _build_system_card(
        data.aggregate,
        per_gpu=data.per_gpu,
    )
    return payload


__all__ = [
    "build_system_section_payload",
]
