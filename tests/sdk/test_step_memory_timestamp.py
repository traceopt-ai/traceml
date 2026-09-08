# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from queue import Queue

import pytest
import torch

import traceml_ai.utils.step_memory as step_memory_module
from traceml_ai.instrumentation import step_events
from traceml_ai.instrumentation.step_events import StepCapture, StepMemoryEvent
from traceml_ai.runtime.state import configure_trace_recording
from traceml_ai.samplers.schema.step_memory import StepMemorySample
from traceml_ai.samplers.step_memory_sampler import StepMemorySampler
from traceml_ai.utils.step_memory import StepMemoryTracker


@pytest.fixture(autouse=True)
def isolated_memory_recording(monkeypatch):
    monkeypatch.delenv("TRACEML_DISABLED", raising=False)
    monkeypatch.setattr(step_events, "_STEP_MEMORY_QUEUE", Queue(maxsize=2048))
    monkeypatch.setattr(step_events, "_ACTIVE_STEP_CAPTURE", StepCapture())
    configure_trace_recording()
    yield
    configure_trace_recording()


def test_tracker_captures_timestamp_when_memory_is_measured(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(step_memory_module.time, "time", lambda: 12.25)

    tracker = StepMemoryTracker(torch.nn.Identity())
    capture = step_events.begin_step_capture()
    tracker.reset()
    tracker.record()
    assert step_events.complete_step_capture(capture, 7)
    monkeypatch.setattr(step_memory_module.time, "time", lambda: 99.0)
    (event,) = step_events.drain_step_memory_events()
    assert event.timestamp == 12.25
    assert event.device == "cpu"
    assert event.peak_allocated is event.peak_reserved is None


def test_sampler_preserves_event_timestamp_through_wire_schema() -> None:
    sampler = StepMemorySampler.__new__(StepMemorySampler)
    sampler.sample_idx = 3
    event = StepMemoryEvent(
        step=7,
        device="cuda:0",
        timestamp=12.25,
        peak_allocated=100.0,
        peak_reserved=200.0,
    )

    sample = sampler._event_to_sample(event)

    assert sample.timestamp == 12.25
    assert sample.to_wire() == {
        "seq": 3,
        "ts": 12.25,
        "device": "cuda:0",
        "step": 7,
        "peak_alloc": 100.0,
        "peak_resv": 200.0,
    }
    assert StepMemorySample.from_wire(sample.to_wire()) == sample


def test_wire_schema_requires_measurement_timestamp() -> None:
    with pytest.raises(KeyError):
        StepMemorySample.from_wire({"seq": 1})


def test_pending_snapshot_is_cleared_between_steps(
    monkeypatch,
) -> None:
    tracker = StepMemoryTracker(torch.nn.Linear(1, 1))
    # Exercise allocator calls without requiring a CUDA-equipped test machine.
    tracker.device = torch.device("cuda:0")
    calls = []
    allocated = iter([100, 150, 30])
    reserved = iter([200, 250, 60])

    def read_peak(name, values, device):
        calls.append((name, str(device)))
        return next(values)

    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset", str(device))),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda device: read_peak("allocated", allocated, device),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        lambda device: read_peak("reserved", reserved, device),
    )
    timestamps = iter([10.0, 11.0, 12.0])
    monkeypatch.setattr(
        step_memory_module.time, "time", lambda: next(timestamps)
    )

    first_capture = step_events.begin_step_capture()
    tracker.reset()
    tracker.record()
    tracker.record()  # Only the final snapshot belongs to this capture.
    assert step_events.complete_step_capture(first_capture, 10)
    # Repeated finalization must not duplicate the event.
    assert not step_events.complete_step_capture(first_capture, 10)
    second_capture = step_events.begin_step_capture()
    tracker.reset()
    tracker.record()
    assert step_events.complete_step_capture(second_capture, 11)
    assert first_capture.memory_event is None
    assert second_capture.memory_event is None
    assert calls == [
        (name, "cuda:0")
        for name in (
            "reset",
            "allocated",
            "reserved",
            "allocated",
            "reserved",
            "reset",
            "allocated",
            "reserved",
        )
    ]

    # Both steps wait in the queue, including after recording is disabled.
    configure_trace_recording(max_steps=11).mark_step_flushed(11)
    monkeypatch.setenv("TRACEML_DISABLED", "1")
    monkeypatch.setattr(step_memory_module.time, "time", lambda: 99.0)
    sampler = StepMemorySampler()
    rows = []
    monkeypatch.setattr(sampler, "_add_record", rows.append)
    sampler.sample()
    assert [
        (r["step"], r["peak_alloc"], r["peak_resv"], r["ts"]) for r in rows
    ] == [
        (10, 150.0, 250.0, 11.0),
        (11, 30.0, 60.0, 12.0),
    ]
    assert all(r["device"] == "cuda:0" for r in rows)
    sampler.sample()
    assert len(rows) == 2


def test_full_memory_queue_drops_only_incoming_event(capsys) -> None:
    events = [
        StepMemoryEvent(step, "cpu", float(step), None, None)
        for step in range(2048)
    ]
    for event in events:
        step_events.publish_step_memory_event(event)
    step_events.publish_step_memory_event(
        StepMemoryEvent(2048, "cpu", 2048.0, None, None)
    )

    drained = step_events.drain_step_memory_events()
    assert len(drained) == 2048
    assert all(actual is expected for actual, expected in zip(drained, events))
    assert "dropping event for step 2048 on cpu" in capsys.readouterr().err
    assert step_events.drain_step_memory_events() == []
