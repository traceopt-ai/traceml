# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Application-boundary contracts for the Step Time pipeline facade."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

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
from traceml_ai.step_time.pipeline import StepTimePipeline
from traceml_ai.step_time.sqlite import SQLiteStepTimeRepository


@pytest.mark.parametrize(
    ("profile", "loader_name", "expected_policy"),
    (
        ("live", "load_live", LIVE_STEP_TIME_POLICY),
        ("summary", "load_summary", SUMMARY_STEP_TIME_POLICY),
    ),
)
def test_run_calls_each_pipeline_stage_once(
    profile: str,
    loader_name: str,
    expected_policy: StepTimeDiagnosisPolicy,
) -> None:
    request = StepTimeLoadRequest(window_size=100, lookback_factor=4)
    snapshot = StepTimeRepositorySnapshot(training_strategy="fsdp")
    window = StepTimeWindow(training_strategy="fsdp")
    diagnosis = Mock(name="diagnosis")
    repository = Mock(spec=SQLiteStepTimeRepository)
    analyzer = Mock(spec=StepTimeAnalyzer)
    getattr(repository, loader_name).return_value = snapshot
    analyzer.analyze.return_value = window

    with patch(
        "traceml_ai.step_time.pipeline.diagnose_step_time_window",
        return_value=diagnosis,
    ) as diagnose:
        result = StepTimePipeline(
            repository=repository,
            profile=profile,
            analyzer=analyzer,
        ).run(request)

    getattr(repository, loader_name).assert_called_once_with(request)
    other_loader = "load_summary" if profile == "live" else "load_live"
    getattr(repository, other_loader).assert_not_called()
    analyzer.analyze.assert_called_once_with(snapshot, window_size=100)
    diagnose.assert_called_once_with(
        window,
        policy=expected_policy,
        include_attribution=False,
    )
    assert result.snapshot is snapshot
    assert result.window is window
    assert result.diagnosis is diagnosis


def test_run_honors_an_explicit_diagnosis_policy() -> None:
    request = StepTimeLoadRequest(window_size=20)
    snapshot = StepTimeRepositorySnapshot()
    window = StepTimeWindow()
    custom_policy = StepTimeDiagnosisPolicy(name="custom")
    repository = Mock(spec=SQLiteStepTimeRepository)
    analyzer = Mock(spec=StepTimeAnalyzer)
    repository.load_live.return_value = snapshot
    analyzer.analyze.return_value = window

    with patch(
        "traceml_ai.step_time.pipeline.diagnose_step_time_window"
    ) as diagnose:
        StepTimePipeline(
            repository=repository,
            policy=custom_policy,
            analyzer=analyzer,
        ).run(request)

    assert diagnose.call_args.kwargs["policy"] is custom_policy


def test_invalid_data_profile_is_rejected_at_construction() -> None:
    repository = Mock(spec=SQLiteStepTimeRepository)

    with pytest.raises(ValueError, match="profile"):
        StepTimePipeline(repository=repository, profile="dashboard")
