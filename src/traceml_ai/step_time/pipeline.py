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

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Literal, Optional

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
from traceml_ai.loggers.error_log import get_error_logger
from traceml_ai.step_time.analysis import StepTimeAnalyzer
from traceml_ai.step_time.model import (
    StepTimeLoadRequest,
    StepTimeRepositorySnapshot,
    StepTimeWindow,
)
from traceml_ai.step_time.sqlite import SQLiteStepTimeRepository

StepTimeProfile = Literal["live", "summary"]
LiveStepTimeFreshness = Literal["cold", "live", "bridged", "expired"]


@dataclass(frozen=True, slots=True)
class StepTimeAnalysis:
    """Facts and diagnosis shared by Step Time presenters."""

    request: StepTimeLoadRequest
    snapshot: StepTimeRepositorySnapshot
    window: StepTimeWindow
    diagnosis: DiagnosticResult[StepDiagnosis]


@dataclass(frozen=True, slots=True)
class LiveStepTimeResult:
    """One presentation-ready live analysis and its freshness state.

    ``freshness`` describes the read, not training-process liveness. Repeated
    reads of an unchanged persisted window remain ``live``. Only empty or
    failed reads enter the last-good bridge and eventually expire it.
    """

    freshness: LiveStepTimeFreshness
    analysis: StepTimeAnalysis


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

    def run(
        self,
        request: StepTimeLoadRequest,
        *,
        previous: Optional[StepTimeAnalysis] = None,
    ) -> StepTimeAnalysis:
        """Return one canonical analysis, reusing an unchanged live result."""
        if self.profile == "live":
            reusable = (
                previous
                if previous is not None and previous.request == request
                else None
            )
            snapshot = (
                self.repository.load_live(
                    request,
                    previous=reusable.snapshot,
                )
                if reusable is not None
                else self.repository.load_live(request)
            )
            if reusable is not None and snapshot is reusable.snapshot:
                return reusable
            default_policy = LIVE_STEP_TIME_POLICY
        else:
            if previous is not None:
                raise ValueError(
                    "Previous analysis reuse is supported only for live "
                    "Step Time reads"
                )
            snapshot = self.repository.load_summary(request)
            default_policy = SUMMARY_STEP_TIME_POLICY

        return _analyze_snapshot(
            request=request,
            snapshot=snapshot,
            analyzer=self.analyzer,
            policy=self.policy or default_policy,
        )


def _analyze_snapshot(
    *,
    request: StepTimeLoadRequest,
    snapshot: StepTimeRepositorySnapshot,
    analyzer: StepTimeAnalyzer,
    policy: StepTimeDiagnosisPolicy,
) -> StepTimeAnalysis:
    """Build one analysis from source facts without presentation concerns."""
    window = analyzer.analyze(
        snapshot,
        window_size=request.window_size,
    )
    diagnosis = diagnose_step_time_window(
        window,
        policy=policy,
    )
    return StepTimeAnalysis(
        request=request,
        snapshot=snapshot,
        window=window,
        diagnosis=diagnosis,
    )


class LiveStepTimeSession:
    """Own one live Step Time cache, bridge, and monotonic expiry clock.

    The session keeps SQLite connections refresh-local, making it safe to
    move between display threads. A lock serializes refreshes so a future
    dashboard fan-out cannot duplicate work for the same session.
    """

    def __init__(
        self,
        db_path: str,
        *,
        request: Optional[StepTimeLoadRequest] = None,
        stale_ttl_s: Optional[float] = 30.0,
        table: str = "step_time_samples",
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        selected = request or StepTimeLoadRequest(
            window_size=100,
            lookback_factor=4,
        )
        self.db_path = str(db_path)
        self.request = StepTimeLoadRequest(
            window_size=max(1, int(selected.window_size)),
            lookback_factor=max(1, int(selected.lookback_factor)),
            rank_filter=(
                tuple(int(rank) for rank in selected.rank_filter)
                if selected.rank_filter is not None
                else None
            ),
        )
        self.table = str(table)
        self._stale_ttl_s = (
            float(stale_ttl_s) if stale_ttl_s is not None else None
        )
        self._monotonic = monotonic
        self._logger = get_error_logger("LiveStepTimeSession")
        self._analyzer = StepTimeAnalyzer()
        self._lock = Lock()
        self._last_observed: Optional[StepTimeAnalysis] = None
        self._last_good: Optional[StepTimeAnalysis] = None
        self._last_good_at: Optional[float] = None
        self._empty_analysis = _analyze_snapshot(
            request=self.request,
            snapshot=StepTimeRepositorySnapshot(),
            analyzer=self._analyzer,
            policy=LIVE_STEP_TIME_POLICY,
        )

    def refresh(self) -> LiveStepTimeResult:
        """Read once and return a live, bridged, cold, or expired result."""
        with self._lock:
            try:
                analysis = self._run_pipeline()
            except Exception:
                self._logger.exception("Live Step Time refresh failed")
                return self._missing_result(
                    empty_analysis=self._empty_analysis,
                )

            self._last_observed = analysis
            if not analysis.window.metrics:
                return self._missing_result(
                    empty_analysis=analysis,
                )

            self._last_good = analysis
            self._last_good_at = float(self._monotonic())
            return LiveStepTimeResult(
                freshness="live",
                analysis=analysis,
            )

    def _run_pipeline(self) -> StepTimeAnalysis:
        """Run the canonical live pipeline on one short-lived connection."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            repository = SQLiteStepTimeRepository(conn, table=self.table)
            return StepTimePipeline(
                repository=repository,
                profile="live",
                analyzer=self._analyzer,
            ).run(
                self.request,
                previous=self._last_observed,
            )

    def _missing_result(
        self,
        *,
        empty_analysis: StepTimeAnalysis,
    ) -> LiveStepTimeResult:
        """Bridge a transient gap, or expose a cold/expired empty result."""
        if self._last_good is not None and self._last_good_at is not None:
            age = float(self._monotonic()) - self._last_good_at
            if self._stale_ttl_s is None or age <= self._stale_ttl_s:
                return LiveStepTimeResult(
                    freshness="bridged",
                    analysis=self._last_good,
                )

        freshness: LiveStepTimeFreshness = (
            "cold" if self._last_good is None else "expired"
        )
        return LiveStepTimeResult(
            freshness=freshness,
            analysis=empty_analysis,
        )


__all__ = [
    "LiveStepTimeFreshness",
    "LiveStepTimeResult",
    "LiveStepTimeSession",
    "StepTimeAnalysis",
    "StepTimePipeline",
    "StepTimeProfile",
]
