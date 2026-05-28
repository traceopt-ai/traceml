# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Final-report process section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict

from traceml_ai.core.summaries import SummaryResult
from traceml_ai.diagnostics.common import DiagnosticResult
from traceml_ai.diagnostics.process import ProcessDiagnosis, diagnose_process
from traceml_ai.diagnostics.process.context import (
    ProcessDiagnosisInput,
    ProcessRankDiagnosisInput,
)
from traceml_ai.reporting.sections.base import BaseSummarySection
from traceml_ai.reporting.sections.process.builder import build_process_payload
from traceml_ai.reporting.sections.process.formatter import (
    format_process_section_text,
)
from traceml_ai.reporting.sections.process.loader import (
    ProcessSectionData,
    load_process_section_data,
)
from traceml_ai.reporting.sections.process.model import (
    MAX_SUMMARY_ROWS,
    PerRankProcessSummary,
)
from traceml_ai.reporting.summaries.summary_formatting import (
    duration_from_bounds,
)


def _rank_diagnosis_inputs(
    per_global_rank: Dict[int, PerRankProcessSummary],
) -> Dict[int, ProcessRankDiagnosisInput]:
    """Convert per-rank summaries into the diagnosis input contract."""
    return {
        int(rank_id): ProcessRankDiagnosisInput(
            cpu_avg_percent=item.cpu_avg_percent,
            cpu_peak_percent=item.cpu_peak_percent,
            ram_avg_bytes=item.ram_avg_bytes,
            ram_peak_bytes=item.ram_peak_bytes,
            ram_total_bytes=item.ram_total_bytes,
            gpu_mem_used_avg_bytes=item.gpu_mem_used_avg_bytes,
            gpu_mem_used_peak_bytes=item.gpu_mem_used_peak_bytes,
            gpu_mem_reserved_avg_bytes=item.gpu_mem_reserved_avg_bytes,
            gpu_mem_reserved_peak_bytes=item.gpu_mem_reserved_peak_bytes,
            gpu_mem_total_bytes=item.gpu_mem_total_bytes,
            gpu_mem_reserved_overhang_ratio=(
                item.gpu_mem_reserved_overhang_ratio
            ),
        )
        for rank_id, item in sorted(per_global_rank.items())
    }


@dataclass(frozen=True)
class ProcessSummarySection(
    BaseSummarySection[
        ProcessSectionData,
        ProcessDiagnosisInput,
        DiagnosticResult[ProcessDiagnosis],
    ],
):
    """Build TraceML's final-report process section."""

    name: ClassVar[str] = "process"
    max_process_rows: int = MAX_SUMMARY_ROWS

    def load(self, db_path: str) -> ProcessSectionData:
        """Load the bounded process telemetry window."""
        return load_process_section_data(
            db_path,
            max_process_rows=self.max_process_rows,
        )

    def to_diagnosis_input(
        self,
        data: ProcessSectionData,
    ) -> ProcessDiagnosisInput:
        """Adapt loaded process data to the diagnosis contract."""
        agg = data.aggregate
        return ProcessDiagnosisInput(
            duration_s=duration_from_bounds(agg.first_ts, agg.last_ts),
            samples=agg.process_samples,
            distinct_ranks=agg.distinct_global_ranks,
            cpu_avg_percent=agg.cpu_avg_percent,
            cpu_peak_percent=agg.cpu_peak_percent,
            cpu_logical_core_count=agg.cpu_logical_core_count,
            ram_avg_bytes=agg.ram_avg_bytes,
            ram_peak_bytes=agg.ram_peak_bytes,
            ram_total_bytes=agg.ram_total_bytes,
            gpu_available=agg.gpu_available,
            gpu_count=agg.gpu_count,
            gpu_mem_used_avg_bytes=agg.gpu_mem_used_avg_bytes,
            gpu_mem_used_peak_bytes=agg.gpu_mem_used_peak_bytes,
            gpu_mem_reserved_avg_bytes=agg.gpu_mem_reserved_avg_bytes,
            gpu_mem_reserved_peak_bytes=agg.gpu_mem_reserved_peak_bytes,
            gpu_mem_total_bytes=agg.gpu_mem_total_bytes,
            per_rank=_rank_diagnosis_inputs(data.per_global_rank),
        )

    def diagnose(
        self,
        diagnosis_input: ProcessDiagnosisInput,
    ) -> DiagnosticResult[ProcessDiagnosis]:
        """Run Process diagnosis for the loaded telemetry window."""
        return diagnose_process(diagnosis_input)

    def build_payload(
        self,
        data: ProcessSectionData,
        diagnosis_result: DiagnosticResult[ProcessDiagnosis],
    ) -> SummaryResult:
        """Assemble the Process summary payload and display text."""
        payload = build_process_payload(data, diagnosis_result)
        return SummaryResult(
            section=self.name,
            payload=payload,
            text=format_process_section_text(payload),
        )


__all__ = [
    "ProcessSummarySection",
    "build_process_payload",
    "format_process_section_text",
    "load_process_section_data",
]
