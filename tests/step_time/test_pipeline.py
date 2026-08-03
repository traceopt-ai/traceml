# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Application-boundary contracts for the Step Time pipeline facade."""

from __future__ import annotations

from unittest.mock import Mock, call, patch

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
from traceml_ai.step_time.pipeline import StepTimeAnalysis, StepTimePipeline
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
    )
    assert result.snapshot is snapshot
    assert result.window is window
    assert result.diagnosis is diagnosis
    assert result.request is request


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


def test_live_run_reuses_the_exact_previous_analysis() -> None:
    request = StepTimeLoadRequest(window_size=100, lookback_factor=4)
    snapshot = StepTimeRepositorySnapshot()
    window = StepTimeWindow()
    repository = Mock(spec=SQLiteStepTimeRepository)
    analyzer = Mock(spec=StepTimeAnalyzer)
    repository.load_live.return_value = snapshot
    analyzer.analyze.return_value = window
    pipeline = StepTimePipeline(repository=repository, analyzer=analyzer)

    with patch(
        "traceml_ai.step_time.pipeline.diagnose_step_time_window"
    ) as diagnose:
        first = pipeline.run(request)
        second = pipeline.run(request, previous=first)

    assert second is first
    assert repository.load_live.call_args_list == [
        call(request),
        call(request, previous=snapshot),
    ]
    assert analyzer.analyze.call_count == 1
    assert diagnose.call_count == 1


def test_live_run_does_not_reuse_an_analysis_for_another_request() -> None:
    first_request = StepTimeLoadRequest(window_size=100, lookback_factor=4)
    second_request = StepTimeLoadRequest(window_size=20, lookback_factor=2)
    first_snapshot = StepTimeRepositorySnapshot()
    second_snapshot = StepTimeRepositorySnapshot()
    repository = Mock(spec=SQLiteStepTimeRepository)
    analyzer = Mock(spec=StepTimeAnalyzer)
    repository.load_live.side_effect = (first_snapshot, second_snapshot)
    analyzer.analyze.return_value = StepTimeWindow()
    pipeline = StepTimePipeline(repository=repository, analyzer=analyzer)

    first = pipeline.run(first_request)
    pipeline.run(second_request, previous=first)

    assert repository.load_live.call_args_list == [
        call(first_request),
        call(second_request),
    ]
    assert analyzer.analyze.call_count == 2


def test_summary_rejects_live_cache_input() -> None:
    request = StepTimeLoadRequest(window_size=10)
    previous = Mock(spec=StepTimeAnalysis)
    repository = Mock(spec=SQLiteStepTimeRepository)

    with pytest.raises(ValueError, match="only for live"):
        StepTimePipeline(repository, profile="summary").run(
            request,
            previous=previous,
        )
