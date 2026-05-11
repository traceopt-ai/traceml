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
from typing import Optional

from traceml.core.summaries import SummaryResult
from traceml.reporting.sections.process.builder import (
    build_process_section_payload,
)
from traceml.reporting.sections.process.formatter import (
    format_process_section_text,
)
from traceml.reporting.sections.process.loader import load_process_section_data
from traceml.reporting.sections.process.model import MAX_SUMMARY_ROWS


@dataclass(frozen=True)
class ProcessSummarySection:
    """Build TraceML's final-report process section."""

    global_rank: Optional[int] = None
    max_process_rows: int = MAX_SUMMARY_ROWS
    name: str = "process"

    def build(self, db_path: str) -> SummaryResult:
        data = load_process_section_data(
            db_path,
            global_rank=self.global_rank,
            max_process_rows=self.max_process_rows,
        )
        payload = build_process_section_payload(data)
        return SummaryResult(
            section=self.name,
            payload=payload,
            text=format_process_section_text(payload),
        )


__all__ = [
    "ProcessSummarySection",
    "build_process_section_payload",
    "format_process_section_text",
    "load_process_section_data",
]
