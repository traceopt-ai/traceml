# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3
from queue import Queue

import pytest
import torch

import traceml_ai.utils.step_memory as step_memory_module
from traceml_ai.aggregator.sqlite_writers import step_memory as projection
from traceml_ai.instrumentation import step_events
from traceml_ai.instrumentation.step_events import StepCapture, StepMemoryEvent
from traceml_ai.runtime.state import configure_trace_recording
from traceml_ai.samplers.schema.step_memory import StepMemorySample
from traceml_ai.samplers.step_memory_sampler import StepMemorySampler
from traceml_ai.telemetry.envelope import normalize_telemetry_envelope
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
    monkeypatch.setattr(
        step_memory_module.time, "time_ns", lambda: 12_250_000_001
    )

    tracker = StepMemoryTracker(torch.nn.Identity())
    capture = step_events.begin_step_capture()
    tracker.reset()
    tracker.record()
    assert step_events.complete_step_capture(capture, 7)
    monkeypatch.setattr(
        step_memory_module.time, "time_ns", lambda: 99_000_000_000
    )
    (event,) = step_events.drain_step_memory_events()
    assert event.timestamp_unix_ns == 12_250_000_001
    assert event.device == "cpu"
    assert event.peak_allocated is event.peak_reserved is None


def test_sampler_preserves_event_timestamp_through_wire_schema() -> None:
    sampler = StepMemorySampler.__new__(StepMemorySampler)
    sampler.sample_idx = 3
    event = StepMemoryEvent(
        step=7,
        device="cuda:0",
        timestamp_unix_ns=1_700_000_000_000_000_001,
        peak_allocated=100.0,
        peak_reserved=200.0,
    )

    sample = sampler._event_to_sample(event)

    assert sample.timestamp_unix_ns == 1_700_000_000_000_000_001
    assert sample.to_wire() == {
        "seq": 3,
        "timestamp_unix_ns": 1_700_000_000_000_000_001,
        "device": "cuda:0",
        "step": 7,
        "peak_alloc": 100.0,
        "peak_resv": 200.0,
    }
    assert StepMemorySample.from_wire(sample.to_wire()) == sample

    envelope = normalize_telemetry_envelope(
        {
            "meta": {"sampler": "StepMemorySampler"},
            "body": {"tables": {"StepMemoryTable": [sample.to_wire()]}},
        }
    )
    with sqlite3.connect(":memory:") as conn:
        projection.init_schema(conn)
        projection.insert_rows(
            conn, projection.build_rows(envelope, observed_timestamp_unix_ns=1)
        )
        assert conn.execute(
            "SELECT timestamp_unix_ns, step, peak_alloc_bytes, "
            "peak_reserved_bytes FROM step_memory_samples;"
        ).fetchone() == (sample.timestamp_unix_ns, 7, 100.0, 200.0)


def test_projection_accepts_only_canonical_source_time():
    timestamp_unix_ns = 1_700_000_000_000_000_001
    inputs = [
        {"timestamp_unix_ns": timestamp_unix_ns},
        {"ts": 1.25},
        {"timestamp_unix_ns": None},
        {"timestamp_unix_ns": True},
        {"timestamp_unix_ns": 1.5},
        {"timestamp_unix_ns": 2**63},
        {},
    ]
    envelope = normalize_telemetry_envelope(
        {
            "meta": {"sampler": "StepMemorySampler"},
            "body": {"tables": {"StepMemoryTable": inputs}},
        }
    )
    with sqlite3.connect(":memory:") as conn:
        projection.init_schema(conn)
        projection.insert_rows(
            conn, projection.build_rows(envelope, observed_timestamp_unix_ns=1)
        )
        columns = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(step_memory_samples);")
        }
        assert columns["timestamp_unix_ns"] == "INTEGER"
        indexes = conn.execute(
            "PRAGMA index_list(step_memory_samples);"
        ).fetchall()
        assert len(indexes) == 4
        assert conn.execute(
            "SELECT timestamp_unix_ns FROM step_memory_samples ORDER BY id;"
        ).fetchall() == [
            (timestamp_unix_ns,),
            *[(None,)] * (len(inputs) - 1),
        ]


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
    timestamps = iter([10_000_000_000, 11_000_000_001, 12_000_000_002])
    monkeypatch.setattr(
        step_memory_module.time, "time_ns", lambda: next(timestamps)
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
    monkeypatch.setattr(
        step_memory_module.time, "time_ns", lambda: 99_000_000_000
    )
    sampler = StepMemorySampler()
    rows = []
    monkeypatch.setattr(sampler, "_add_record", rows.append)
    sampler.sample()
    assert [
        (r["step"], r["peak_alloc"], r["peak_resv"], r["timestamp_unix_ns"])
        for r in rows
    ] == [
        (10, 150.0, 250.0, 11_000_000_001),
        (11, 30.0, 60.0, 12_000_000_002),
    ]
    assert all(r["device"] == "cuda:0" for r in rows)
    sampler.sample()
    assert len(rows) == 2


def test_full_memory_queue_drops_only_incoming_event(capsys) -> None:
    events = [
        StepMemoryEvent(step, "cpu", step * 1_000_000_000, None, None)
        for step in range(2048)
    ]
    for event in events:
        step_events.publish_step_memory_event(event)
    step_events.publish_step_memory_event(
        StepMemoryEvent(2048, "cpu", 2_048_000_000_000, None, None)
    )

    drained = step_events.drain_step_memory_events()
    assert len(drained) == 2048
    assert all(actual is expected for actual, expected in zip(drained, events))
    assert "dropping event for step 2048 on cpu" in capsys.readouterr().err
    assert step_events.drain_step_memory_events() == []
