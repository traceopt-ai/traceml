# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Final-report step-memory section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from traceml.core.summaries import SummaryResult
from traceml.reporting.sections.step_memory.builder import (
    build_step_memory_section_payload,
)
from traceml.reporting.sections.step_memory.formatter import (
    format_step_memory_section_text,
)
from traceml.reporting.sections.step_memory.loader import (
    load_step_memory_section_data,
)


@dataclass(frozen=True)
class StepMemorySummarySection:
    """Build TraceML's final-report step-memory section."""

    name: ClassVar[str] = "step_memory"
    window_size: int = 400

    def build(self, db_path: str) -> SummaryResult:
        """Build the Step Memory summary section for a TraceML SQLite database."""
        data = load_step_memory_section_data(
            db_path,
            window_size=self.window_size,
        )
        payload = build_step_memory_section_payload(data)
        return SummaryResult(
            section=self.name,
            payload=payload,
            text=format_step_memory_section_text(payload),
        )


__all__ = [
    "StepMemorySummarySection",
    "build_step_memory_section_payload",
    "format_step_memory_section_text",
    "load_step_memory_section_data",
]
