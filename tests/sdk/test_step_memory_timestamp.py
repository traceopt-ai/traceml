# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

import traceml_ai.utils.step_memory as step_memory_module
from traceml_ai.samplers.schema.step_memory import StepMemorySample
from traceml_ai.samplers.step_memory_sampler import StepMemorySampler
from traceml_ai.utils.step_memory import StepMemoryEvent, StepMemoryTracker


def test_tracker_captures_timestamp_when_memory_is_measured(
    monkeypatch,
) -> None:
    monkeypatch.delenv("TRACEML_DISABLED", raising=False)
    monkeypatch.setattr(
        step_memory_module,
        "should_record_trace_events",
        lambda: True,
    )
    monkeypatch.setattr(step_memory_module.time, "time", lambda: 12.25)

    tracker = StepMemoryTracker.__new__(StepMemoryTracker)
    tracker.model_id = 7
    tracker.device = torch.device("cpu")

    try:
        tracker.record()
        event = step_memory_module._temp_step_memory_buffer.pop(7)
        assert event.timestamp == 12.25
    finally:
        step_memory_module._temp_step_memory_buffer.pop(7, None)


def test_sampler_preserves_event_timestamp_through_wire_schema() -> None:
    sampler = StepMemorySampler.__new__(StepMemorySampler)
    sampler.sample_idx = 3
    event = StepMemoryEvent(
        step=7,
        model_id=1,
        device="cuda:0",
        timestamp=12.25,
        peak_allocated=100.0,
        peak_reserved=200.0,
    )

    sample = sampler._event_to_sample(event)

    assert sample.timestamp == 12.25
    assert StepMemorySample.from_wire(sample.to_wire()) == sample


def test_wire_schema_requires_measurement_timestamp() -> None:
    with pytest.raises(KeyError):
        StepMemorySample.from_wire({"seq": 1})
