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
from typing import ClassVar, Optional

from traceml.core.summaries import SummaryResult
from traceml.reporting.sections.system.builder import (
    build_system_section_payload,
)
from traceml.reporting.sections.system.formatter import (
    format_system_section_text,
)
from traceml.reporting.sections.system.loader import load_system_section_data
from traceml.reporting.sections.system.model import MAX_SUMMARY_ROWS


@dataclass(frozen=True)
class SystemSummarySection:
    """Build TraceML's final-report system section."""

    name: ClassVar[str] = "system"
    node_rank: Optional[int] = None
    max_system_rows: int = MAX_SUMMARY_ROWS

    def build(self, db_path: str) -> SummaryResult:
        """Build the System summary section for a TraceML SQLite database."""
        data = load_system_section_data(
            db_path,
            node_rank=self.node_rank,
            max_system_rows=self.max_system_rows,
        )
        payload = build_system_section_payload(data)
        return SummaryResult(
            section=self.name,
            payload=payload,
            text=format_system_section_text(payload),
        )


__all__ = [
    "SystemSummarySection",
    "build_system_section_payload",
    "format_system_section_text",
    "load_system_section_data",
]
