# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Final-report Step Time section.

The section has one readable application path: load the metadata-complete
repository profile, run canonical analysis and diagnosis once, then project
the immutable result to the public report schema.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from typing import ClassVar

from traceml_ai.core.summaries import SummaryResult
from traceml_ai.reporting.config import (
    DEFAULT_SUMMARY_WINDOW_ROWS,
    normalize_summary_window_rows,
)
from traceml_ai.reporting.sections.step_time.builder import (
    STEP_TIME_METRIC_NAMES,
    project_step_time_summary,
)
from traceml_ai.step_time.model import StepTimeLoadRequest
from traceml_ai.step_time.pipeline import StepTimePipeline
from traceml_ai.step_time.sqlite import SQLiteStepTimeRepository


@dataclass(frozen=True)
class StepTimeSummarySection:
    """Load, analyze, and project TraceML's final Step Time section once."""

    name: ClassVar[str] = "step_time"
    max_rows: int = DEFAULT_SUMMARY_WINDOW_ROWS

    def build(self, db_path: str) -> SummaryResult:
        """Build one schema-stable section from persisted telemetry."""
        row_limit = normalize_summary_window_rows(self.max_rows)
        with closing(sqlite3.connect(db_path)) as connection:
            analysis = StepTimePipeline(
                repository=SQLiteStepTimeRepository(connection),
                profile="summary",
            ).run(StepTimeLoadRequest(window_size=row_limit))

        payload = project_step_time_summary(analysis)
        return SummaryResult(
            section=self.name,
            payload=payload,
            text=str(payload.get("card", "")),
        )


__all__ = [
    "STEP_TIME_METRIC_NAMES",
    "StepTimeSummarySection",
    "project_step_time_summary",
]
