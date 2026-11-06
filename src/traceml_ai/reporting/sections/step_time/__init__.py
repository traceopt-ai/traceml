# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Final-report Step Time section.

The section has one readable application path: load the metadata-complete
repository profile, run canonical analysis and diagnosis once, then project
the analyzed result to the public report schema.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from typing import ClassVar

from traceml_ai.core.summaries import SummaryResult
from traceml_ai.diagnostics.step_time.policy import SUMMARY_STEP_TIME_POLICY
from traceml_ai.measurements.window import AnalysisWindow
from traceml_ai.reporting.analysis_window import analysis_window_metadata
from traceml_ai.reporting.sections.step_time.builder import (
    STEP_TIME_METRIC_NAMES,
    project_step_time_summary,
)
from traceml_ai.step_time.analysis import StepTimeAnalyzer
from traceml_ai.step_time.model import (
    StepTimeLoadRequest,
    StepTimeRepositorySnapshot,
)
from traceml_ai.step_time.pipeline import analyze_step_time_snapshot
from traceml_ai.step_time.sqlite import SQLiteStepTimeRepository


@dataclass(frozen=True)
class StepTimeSummarySection:
    """Load, analyze, and project TraceML's final Step Time section once."""

    name: ClassVar[str] = "step_time"
    analysis_window: AnalysisWindow | None = None

    def build(self, db_path: str) -> SummaryResult:
        """Build one schema-stable section from persisted telemetry."""
        with closing(sqlite3.connect(db_path)) as connection:
            snapshot = SQLiteStepTimeRepository(connection).load_summary(
                self._load_request()
            )
        return self.build_from_data(snapshot)

    def build_from_data(
        self,
        snapshot: StepTimeRepositorySnapshot,
    ) -> SummaryResult:
        """Analyze and project a repository snapshot read by the caller."""
        analysis = analyze_step_time_snapshot(
            request=self._load_request(),
            snapshot=snapshot,
            analyzer=StepTimeAnalyzer(),
            policy=SUMMARY_STEP_TIME_POLICY,
        )

        payload = project_step_time_summary(analysis)
        if self.analysis_window is not None:
            payload["metadata"].update(
                analysis_window_metadata(self.analysis_window)
            )
        return SummaryResult(
            section=self.name,
            payload=payload,
            text=str(payload.get("card", "")),
        )

    def _load_request(self) -> StepTimeLoadRequest:
        """Return the repository bounds for this summary section."""
        window = self.analysis_window
        return StepTimeLoadRequest(
            start_step=window.start_step if window is not None else None,
            end_step=window.end_step if window is not None else None,
        )


__all__ = [
    "STEP_TIME_METRIC_NAMES",
    "StepTimeSummarySection",
    "project_step_time_summary",
]
