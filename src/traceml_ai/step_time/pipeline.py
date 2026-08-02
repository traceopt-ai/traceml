# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""One application-level path through Step Time telemetry.

The facade deliberately performs only orchestration. SQLite selection,
analysis semantics, diagnosis rules, and presentation remain independently
owned and testable. Live and summary are data profiles, not UI surfaces:
terminal and dashboard both use the bounded live profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from traceml_ai.diagnostics.common import DiagnosticResult
from traceml_ai.diagnostics.step_time.api import (
    StepDiagnosis,
    diagnose_step_time_window,
)
from traceml_ai.diagnostics.step_time.policy import (
    LIVE_STEP_TIME_POLICY,
    SUMMARY_STEP_TIME_POLICY,
    StepTimeDiagnosisPolicy,
)
from traceml_ai.step_time.analysis import StepTimeAnalyzer
from traceml_ai.step_time.model import (
    StepTimeLoadRequest,
    StepTimeRepositorySnapshot,
    StepTimeWindow,
)
from traceml_ai.step_time.sqlite import SQLiteStepTimeRepository

StepTimeProfile = Literal["live", "summary"]


@dataclass(frozen=True, slots=True)
class StepTimeAnalysis:
    """Immutable result shared by future Step Time presenters."""

    snapshot: StepTimeRepositorySnapshot
    window: StepTimeWindow
    diagnosis: DiagnosticResult[StepDiagnosis]


@dataclass(slots=True)
class StepTimePipeline:
    """Load, analyze, and diagnose exactly once for one data profile.

    Args:
        repository: SQLite source boundary owned by the caller.
        profile: ``live`` for terminal/dashboard bounded-tail reads or
            ``summary`` for metadata-complete final-summary reads.
        policy: Optional diagnosis override. The profile's standard policy is
            used when omitted.
        analyzer: Injectable canonical analyzer, primarily for focused tests.
    """

    repository: SQLiteStepTimeRepository
    profile: StepTimeProfile = "live"
    policy: Optional[StepTimeDiagnosisPolicy] = None
    analyzer: StepTimeAnalyzer = field(default_factory=StepTimeAnalyzer)

    def __post_init__(self) -> None:
        if self.profile not in ("live", "summary"):
            raise ValueError(
                "Step Time profile must be 'live' or 'summary', "
                f"got {self.profile!r}"
            )

    def run(self, request: StepTimeLoadRequest) -> StepTimeAnalysis:
        """Return one canonical analysis for ``request``."""
        if self.profile == "live":
            snapshot = self.repository.load_live(request)
            default_policy = LIVE_STEP_TIME_POLICY
        else:
            snapshot = self.repository.load_summary(request)
            default_policy = SUMMARY_STEP_TIME_POLICY

        window = self.analyzer.analyze(
            snapshot,
            window_size=request.window_size,
        )
        diagnosis = diagnose_step_time_window(
            window,
            policy=self.policy or default_policy,
            include_attribution=False,
        )
        return StepTimeAnalysis(
            snapshot=snapshot,
            window=window,
            diagnosis=diagnosis,
        )


__all__ = ["StepTimeAnalysis", "StepTimePipeline", "StepTimeProfile"]
