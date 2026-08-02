# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Temporary dashboard adapter for the canonical live Step Time session."""

from __future__ import annotations

from typing import Optional, Sequence

from traceml_ai.aggregator.display_drivers.layout import MODEL_COMBINED_LAYOUT
from traceml_ai.renderers.base_renderer import BaseRenderer
from traceml_ai.step_time.model import StepTimeLoadRequest, StepTimeResult
from traceml_ai.step_time.pipeline import (
    LiveStepTimeResult,
    LiveStepTimeSession,
)

STEP_TIME_TABLE = "step_time_samples"


def _legacy_result(result: LiveStepTimeResult) -> StepTimeResult:
    """Project live state into the dashboard contract retained for PR7."""
    visible = result.freshness in {"live", "bridged"}
    window = result.analysis.window if visible else None
    return StepTimeResult(
        status_message=result.status_message,
        window=window,
        training_strategy=result.analysis.window.training_strategy,
        had_ok=result.freshness != "cold",
    )


class StepCombinedComputer:
    """Keep current dashboard callers stable while they migrate in PR7.

    The adapter owns no SQL, freshness, cache, or diagnosis behavior. It will
    be removed after both dashboard Step Time consumers share one live session.
    """

    def __init__(
        self,
        db_path: str,
        window_size: int = 100,
        stale_ttl_s: Optional[float] = 30.0,
        table: str = STEP_TIME_TABLE,
        lookback_factor: int = 4,
        rank_filter: Optional[Sequence[int]] = None,
    ) -> None:
        self._session = LiveStepTimeSession(
            db_path,
            request=StepTimeLoadRequest(
                window_size=window_size,
                lookback_factor=lookback_factor,
                rank_filter=(
                    tuple(rank_filter) if rank_filter is not None else None
                ),
            ),
            stale_ttl_s=stale_ttl_s,
            table=table,
        )

    def compute_dashboard(self) -> StepTimeResult:
        """Return the historical dashboard payload from one live refresh."""
        return _legacy_result(self._session.refresh())


class StepTimeDashboardAdapter(BaseRenderer):
    """Temporary dashboard provider retained until the PR7 shared fan-out."""

    def __init__(self, db_path: str) -> None:
        super().__init__(
            name="Model Step Summary",
            layout_section_name=MODEL_COMBINED_LAYOUT,
        )
        self._computer = StepCombinedComputer(db_path=db_path)

    def get_dashboard_renderable(self) -> StepTimeResult:
        """Return the compatibility payload expected by the current UI."""
        return self._computer.compute_dashboard()


__all__ = [
    "STEP_TIME_TABLE",
    "StepCombinedComputer",
    "StepTimeDashboardAdapter",
]
