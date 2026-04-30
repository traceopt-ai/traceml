"""
Payload builder for the final-report step-memory section.
"""

from __future__ import annotations

from typing import Any, Dict

from traceml.reporting.sections.step_memory.loader import StepMemorySectionData
from traceml.reporting.summaries.step_memory import _build_step_memory_card


def build_step_memory_section_payload(
    data: StepMemorySectionData,
) -> Dict[str, Any]:
    """
    Build the JSON-safe step-memory section payload from loaded data.
    """
    _, payload = _build_step_memory_card(
        training_steps=data.training_steps,
        latest_step_observed=data.latest_step_observed,
        metrics=data.metrics,
        diagnosis=data.diagnosis,
        diagnosis_result=data.diagnosis_result,
        no_gpu_detected=data.no_gpu_detected,
        per_rank=data.per_rank,
    )
    return payload


__all__ = [
    "build_step_memory_section_payload",
]
