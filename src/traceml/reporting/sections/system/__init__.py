# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Final-report system section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional

from traceml.core.summaries import SummaryResult
from traceml.diagnostics.common import DiagnosticResult
from traceml.diagnostics.system import SystemDiagnosis, diagnose_system
from traceml.diagnostics.system.context import (
    SystemDiagnosisInput,
    SystemGpuDiagnosisInput,
    SystemNodeDiagnosisInput,
)
from traceml.reporting.sections.base import BaseSummarySection
from traceml.reporting.sections.system.builder import build_system_payload
from traceml.reporting.sections.system.formatter import (
    format_system_section_text,
)
from traceml.reporting.sections.system.loader import (
    SystemSectionData,
    load_system_section_data,
)
from traceml.reporting.sections.system.model import (
    MAX_SUMMARY_ROWS,
    PerGPUSummary,
    SystemNodeSummary,
)
from traceml.reporting.summaries.summary_formatting import duration_from_bounds


def _gpu_diagnosis_inputs(
    per_gpu: Dict[int, PerGPUSummary],
) -> Dict[int, SystemGpuDiagnosisInput]:
    """Convert GPU summaries into the diagnosis input contract."""
    return {
        int(gpu_idx): SystemGpuDiagnosisInput(
            util_avg_percent=item.util_avg_percent,
            util_peak_percent=item.util_peak_percent,
            mem_avg_bytes=item.mem_avg_bytes,
            mem_peak_bytes=item.mem_peak_bytes,
            mem_total_bytes=item.mem_total_bytes,
            temp_avg_c=item.temp_avg_c,
            temp_peak_c=item.temp_peak_c,
            power_avg_w=item.power_avg_w,
            power_peak_w=item.power_peak_w,
            power_limit_w=item.power_limit_w,
        )
        for gpu_idx, item in sorted(per_gpu.items())
    }


def _node_diagnosis_input(
    node: SystemNodeSummary,
) -> SystemNodeDiagnosisInput:
    """Convert one node summary into diagnosis input."""
    agg = node.aggregate
    return SystemNodeDiagnosisInput(
        node_label=node.identity.label,
        node_rank=node.identity.node_rank,
        duration_s=duration_from_bounds(agg.first_ts, agg.last_ts),
        samples=agg.system_samples,
        cpu_avg_percent=agg.cpu_avg_percent,
        cpu_peak_percent=agg.cpu_peak_percent,
        ram_avg_bytes=agg.ram_avg_bytes,
        ram_peak_bytes=agg.ram_peak_bytes,
        ram_total_bytes=agg.ram_total_bytes,
        gpu_available=agg.gpu_available,
        gpu_count=agg.gpu_count,
        gpu_util_avg_percent=agg.gpu_util_avg_percent,
        gpu_util_peak_percent=agg.gpu_util_peak_percent,
        gpu_mem_avg_bytes=agg.gpu_mem_avg_bytes,
        gpu_mem_peak_bytes=agg.gpu_mem_peak_bytes,
        gpu_temp_avg_c=agg.gpu_temp_avg_c,
        gpu_temp_peak_c=agg.gpu_temp_peak_c,
        gpu_power_avg_w=agg.gpu_power_avg_w,
        gpu_power_peak_w=agg.gpu_power_peak_w,
        per_gpu=_gpu_diagnosis_inputs(node.per_gpu),
    )


@dataclass(frozen=True)
class SystemSummarySection(
    BaseSummarySection[
        SystemSectionData,
        SystemDiagnosisInput,
        DiagnosticResult[SystemDiagnosis],
    ],
):
    """Build TraceML's final-report system section."""

    name: ClassVar[str] = "system"
    node_rank: Optional[int] = None
    max_system_rows: int = MAX_SUMMARY_ROWS

    def load(self, db_path: str) -> SystemSectionData:
        """Load the bounded system telemetry window."""
        return load_system_section_data(
            db_path,
            node_rank=self.node_rank,
            max_system_rows=self.max_system_rows,
        )

    def to_diagnosis_input(
        self,
        data: SystemSectionData,
    ) -> SystemDiagnosisInput:
        """Adapt loaded system data to the diagnosis contract."""
        cluster = data.cluster
        agg = cluster.aggregate
        return SystemDiagnosisInput(
            duration_s=duration_from_bounds(agg.first_ts, agg.last_ts),
            samples=agg.system_samples,
            nodes_seen=cluster.observed_nodes,
            cpu_avg_percent=agg.cpu_avg_percent,
            cpu_peak_percent=agg.cpu_peak_percent,
            ram_avg_bytes=agg.ram_avg_bytes,
            ram_peak_bytes=agg.ram_peak_bytes,
            ram_total_bytes=agg.ram_total_bytes,
            gpu_available=agg.gpu_available,
            gpu_count=agg.gpu_count,
            gpu_util_avg_percent=agg.gpu_util_avg_percent,
            gpu_util_peak_percent=agg.gpu_util_peak_percent,
            gpu_mem_avg_bytes=agg.gpu_mem_avg_bytes,
            gpu_mem_peak_bytes=agg.gpu_mem_peak_bytes,
            gpu_temp_avg_c=agg.gpu_temp_avg_c,
            gpu_temp_peak_c=agg.gpu_temp_peak_c,
            gpu_power_avg_w=agg.gpu_power_avg_w,
            gpu_power_peak_w=agg.gpu_power_peak_w,
            per_node={
                label: _node_diagnosis_input(node)
                for label, node in sorted(cluster.nodes.items())
            },
        )

    def diagnose(
        self,
        diagnosis_input: SystemDiagnosisInput,
    ) -> DiagnosticResult[SystemDiagnosis]:
        """Run System diagnosis for the loaded telemetry window."""
        return diagnose_system(diagnosis_input)

    def build_payload(
        self,
        data: SystemSectionData,
        diagnosis_result: DiagnosticResult[SystemDiagnosis],
    ) -> SummaryResult:
        """Assemble the System summary payload and display text."""
        payload = build_system_payload(data, diagnosis_result)
        return SummaryResult(
            section=self.name,
            payload=payload,
            text=format_system_section_text(payload),
        )


__all__ = [
    "SystemSummarySection",
    "build_system_payload",
    "format_system_section_text",
    "load_system_section_data",
]
