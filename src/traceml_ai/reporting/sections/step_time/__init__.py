# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Final-report step-time section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from traceml_ai.core.summaries import SummaryResult
from traceml_ai.diagnostics.common import DiagnosticResult
from traceml_ai.diagnostics.step_time.adapters import (
    StepTimeDiagnosisInput,
    diagnose_step_time_summary,
)
from traceml_ai.diagnostics.step_time.api import StepDiagnosis
from traceml_ai.reporting.sections.base import BaseSummarySection
from traceml_ai.reporting.sections.step_time.builder import (
    build_step_time_payload,
)
from traceml_ai.reporting.sections.step_time.formatter import (
    format_step_time_section_text,
)
from traceml_ai.reporting.sections.step_time.loader import (
    StepTimeSectionData,
    load_step_time_section_data,
)
from traceml_ai.reporting.sections.step_time.model import (
    MAX_SUMMARY_WINDOW_ROWS,
    to_rank_signals,
)


@dataclass(frozen=True)
class StepTimeSummarySection(
    BaseSummarySection[
        StepTimeSectionData,
        StepTimeDiagnosisInput,
        Optional[DiagnosticResult[StepDiagnosis]],
    ],
):
    """Build TraceML's final-report step-time section."""

    name: ClassVar[str] = "step_time"
    max_rows: int = MAX_SUMMARY_WINDOW_ROWS

    def load(self, db_path: str) -> StepTimeSectionData:
        """Load the bounded, step-aligned Step Time telemetry window."""
        return load_step_time_section_data(db_path, max_rows=self.max_rows)

    def to_diagnosis_input(
        self,
        data: StepTimeSectionData,
    ) -> StepTimeDiagnosisInput:
        """Adapt aligned Step Time summaries to the diagnosis contract."""
        return StepTimeDiagnosisInput(
            rank_signals=to_rank_signals(data.aligned_summary),
            per_rank_step_metrics=data.aligned_step_metrics,
            max_rows=data.max_rows,
        )

    def diagnose(
        self,
        diagnosis_input: StepTimeDiagnosisInput,
    ) -> Optional[DiagnosticResult[StepDiagnosis]]:
        """Run Step Time diagnosis for the aligned telemetry window."""
        return diagnose_step_time_summary(diagnosis_input)

    def build_payload(
        self,
        data: StepTimeSectionData,
        diagnosis_result: Optional[DiagnosticResult[StepDiagnosis]],
    ) -> SummaryResult:
        """Assemble the Step Time summary payload and display text."""
        payload = build_step_time_payload(data, diagnosis_result)
        return SummaryResult(
            section=self.name,
            payload=payload,
            text=format_step_time_section_text(payload),
        )


__all__ = [
    "StepTimeSummarySection",
    "build_step_time_payload",
    "format_step_time_section_text",
    "load_step_time_section_data",
]
