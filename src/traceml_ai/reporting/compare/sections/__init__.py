"""Registered compare section builders."""

from __future__ import annotations

from traceml_ai.reporting.compare.sections.base import SectionComparer
from traceml_ai.reporting.compare.sections.process import ProcessComparer
from traceml_ai.reporting.compare.sections.step_memory import (
    StepMemoryComparer,
)
from traceml_ai.reporting.compare.sections.step_time import StepTimeComparer
from traceml_ai.reporting.compare.sections.system import SystemComparer

SECTION_COMPARERS: tuple[SectionComparer, ...] = (
    StepTimeComparer(),
    StepMemoryComparer(),
    ProcessComparer(),
    SystemComparer(),
)

__all__ = [
    "ProcessComparer",
    "SECTION_COMPARERS",
    "StepMemoryComparer",
    "StepTimeComparer",
    "SystemComparer",
]
