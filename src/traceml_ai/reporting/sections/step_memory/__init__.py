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

from traceml_ai.core.summaries import SummaryResult
from traceml_ai.diagnostics.common import DiagnosticResult
from traceml_ai.diagnostics.step_memory import (
    SUMMARY_STEP_MEMORY_POLICY,
    StepMemoryDiagnosis,
    StepMemoryDiagnosisInput,
    diagnose_step_memory_summary,
)
from traceml_ai.reporting.sections.base import BaseSummarySection
from traceml_ai.reporting.sections.step_memory.builder import (
    build_step_memory_section_payload,
)
from traceml_ai.reporting.sections.step_memory.formatter import (
    format_step_memory_section_text,
)
from traceml_ai.reporting.sections.step_memory.loader import (
    StepMemorySectionData,
    load_step_memory_section_data,
)
from traceml_ai.reporting.sections.step_memory.model import (
    MAX_SUMMARY_WINDOW_ROWS,
)


@dataclass(frozen=True)
class StepMemorySummarySection(
    BaseSummarySection[
        StepMemorySectionData,
        StepMemoryDiagnosisInput,
        DiagnosticResult[StepMemoryDiagnosis],
    ],
):
    """Build TraceML's final-report step-memory section."""

    name: ClassVar[str] = "step_memory"
    window_size: int = MAX_SUMMARY_WINDOW_ROWS

    def load(self, db_path: str) -> StepMemorySectionData:
        """Load the bounded, aligned Step Memory telemetry window."""
        return load_step_memory_section_data(
            db_path,
            window_size=self.window_size,
        )

    def to_diagnosis_input(
        self,
        data: StepMemorySectionData,
    ) -> StepMemoryDiagnosisInput:
        """Adapt aligned Step Memory metrics to the diagnosis contract."""
        return StepMemoryDiagnosisInput(
            metrics=tuple(data.metrics),
            gpu_total_bytes=data.gpu_total_bytes,
            no_gpu_detected=data.no_gpu_detected,
            thresholds=SUMMARY_STEP_MEMORY_POLICY.thresholds,
        )

    def diagnose(
        self,
        diagnosis_input: StepMemoryDiagnosisInput,
    ) -> DiagnosticResult[StepMemoryDiagnosis]:
        """Run Step Memory diagnosis for the aligned telemetry window."""
        return diagnose_step_memory_summary(diagnosis_input)

    def build_payload(
        self,
        data: StepMemorySectionData,
        diagnosis_result: DiagnosticResult[StepMemoryDiagnosis],
    ) -> SummaryResult:
        """Assemble the Step Memory summary payload and display text."""
        payload = build_step_memory_section_payload(data, diagnosis_result)
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
