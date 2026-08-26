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
from traceml_ai.reporting.analysis_window import AnalysisWindow
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
    analysis_window: AnalysisWindow | None = None

    def load(self, db_path: str) -> StepMemorySectionData:
        """Load aligned Step Memory telemetry for the final report."""
        return load_step_memory_section_data(
            db_path,
            start_step=(
                self.analysis_window.start_step
                if self.analysis_window is not None
                else None
            ),
            end_step=(
                self.analysis_window.end_step
                if self.analysis_window is not None
                else None
            ),
            start_ts_s=(
                self.analysis_window.start_ts_s
                if self.analysis_window is not None
                else None
            ),
            end_ts_s=(
                self.analysis_window.end_ts_s
                if self.analysis_window is not None
                else None
            ),
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
        if self.analysis_window is not None:
            payload["metadata"].update(self.analysis_window.metadata())
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
