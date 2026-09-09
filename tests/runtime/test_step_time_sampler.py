from queue import Queue

import pytest

from traceml_ai.instrumentation import step_events
from traceml_ai.instrumentation.step_events import (
    StepCapture,
    StepTimeBatch,
    TimeEvent,
)
from traceml_ai.runtime.state import configure_trace_recording
from traceml_ai.samplers.step_time_sampler import StepTimeSampler
from traceml_ai.utils import timing


@pytest.fixture
def isolated_timing_queue(monkeypatch):
    monkeypatch.delenv("TRACEML_DISABLED", raising=False)
    monkeypatch.setattr(step_events, "_STEP_TIME_QUEUE", Queue(maxsize=2048))
    monkeypatch.setattr(step_events, "_ACTIVE_STEP_CAPTURE", StepCapture())
    configure_trace_recording()
    yield
    configure_trace_recording()


def _payload_for(*events: TimeEvent):
    sampler = StepTimeSampler()
    _timestamp, payload = sampler._build_step_payload(
        StepTimeBatch(step=1, events=list(events))
    )
    return payload


def test_step_time_sampler_emits_cpu_clock_for_cpu_only_event() -> None:
    payload = _payload_for(
        TimeEvent(
            name="_test_cpu",
            device="cpu",
            cpu_start=10.0,
            cpu_end=10.025,
        )
    )

    stats = payload["_test_cpu"]["cpu"]

    assert stats["is_gpu"] is False
    assert stats["duration_ms"] == pytest.approx(25.0)
    assert stats["cpu_ms"] == pytest.approx(25.0)
    assert stats["gpu_ms"] is None
    assert stats["n_calls"] == 1


def test_step_time_sampler_emits_cpu_and_gpu_clocks_for_gpu_event() -> None:
    payload = _payload_for(
        TimeEvent(
            name="_test_gpu",
            device="cuda:0",
            cpu_start=20.0,
            cpu_end=20.004,
            gpu_time_ms=12.5,
            resolved=True,
        )
    )

    stats = payload["_test_gpu"]["cuda:0"]

    assert stats["is_gpu"] is True
    assert stats["duration_ms"] == pytest.approx(12.5)
    assert stats["cpu_ms"] == pytest.approx(4.0)
    assert stats["gpu_ms"] == pytest.approx(12.5)
    assert stats["n_calls"] == 1


def test_step_time_sampler_keeps_dataloader_duration_cpu_with_gpu_event() -> (
    None
):
    payload = _payload_for(
        TimeEvent(
            name="_traceml_internal:dataloader_next",
            device="cuda:0",
            cpu_start=30.0,
            cpu_end=30.006,
            gpu_time_ms=1.5,
            resolved=True,
        )
    )

    stats = payload["_traceml_internal:dataloader_next"]["cuda:0"]

    assert stats["is_gpu"] is False
    assert stats["duration_ms"] == pytest.approx(6.0)
    assert stats["cpu_ms"] == pytest.approx(6.0)
    assert stats["gpu_ms"] == pytest.approx(1.5)
    assert stats["n_calls"] == 1


def test_step_time_sampler_keeps_step_envelope_duration_cpu_with_gpu_event() -> (
    None
):
    payload = _payload_for(
        TimeEvent(
            name="_traceml_internal:step_time",
            device="cuda:0",
            cpu_start=40.0,
            cpu_end=40.025,
            gpu_time_ms=12.0,
            resolved=True,
        )
    )

    stats = payload["_traceml_internal:step_time"]["cuda:0"]

    assert stats["is_gpu"] is False
    assert stats["duration_ms"] == pytest.approx(25.0)
    assert stats["cpu_ms"] == pytest.approx(25.0)
    assert stats["gpu_ms"] == pytest.approx(12.0)
    assert stats["n_calls"] == 1


def test_step_time_sampler_aggregates_repeated_event_clocks() -> None:
    payload = _payload_for(
        TimeEvent(
            name="_test_gpu",
            device="cuda:0",
            cpu_start=1.0,
            cpu_end=1.002,
            gpu_time_ms=5.0,
            resolved=True,
        ),
        TimeEvent(
            name="_test_gpu",
            device="cuda:0",
            cpu_start=2.0,
            cpu_end=2.003,
            gpu_time_ms=7.0,
            resolved=True,
        ),
    )

    stats = payload["_test_gpu"]["cuda:0"]

    assert stats["is_gpu"] is True
    assert stats["duration_ms"] == pytest.approx(12.0)
    assert stats["cpu_ms"] == pytest.approx(5.0)
    assert stats["gpu_ms"] == pytest.approx(12.0)
    assert stats["n_calls"] == 2


def test_completed_captures_wait_until_sampler_drains(
    isolated_timing_queue, monkeypatch
):
    first = TimeEvent("forward", "cpu", 1.0, 1.002)
    repeated = TimeEvent("forward", "cpu", 2.0, 2.003)
    second = TimeEvent("forward", "cpu", 3.0, 3.004)
    first_capture = step_events.begin_step_capture()
    timing.record_event(first)
    timing.record_event(repeated)
    assert step_events.complete_step_capture(first_capture, 10)
    second_capture = step_events.begin_step_capture()
    timing.record_event(second)
    assert step_events.complete_step_capture(second_capture, 11)
    assert not first_capture.timing_events
    assert not second_capture.timing_events

    # Recording may stop before the next sampler tick; queued work still drains.
    configure_trace_recording(max_steps=11).mark_step_flushed(11)
    monkeypatch.setenv("TRACEML_DISABLED", "1")
    sampler = StepTimeSampler()
    rows = []
    monkeypatch.setattr(sampler, "_add_record", rows.append)
    sampler.sample()

    assert [row["step"] for row in rows] == [10, 11]
    assert [first.step, repeated.step, second.step] == [10, 10, 11]
    assert rows[0]["events"]["forward"]["cpu"]["cpu_ms"] == pytest.approx(5.0)
    assert rows[0]["events"]["forward"]["cpu"]["n_calls"] == 2
    assert rows[1]["events"]["forward"]["cpu"]["cpu_ms"] == pytest.approx(4.0)
    assert [row["timestamp"] for row in rows] == [2.003, 3.004]
    assert step_events.drain_step_time_batches() == []
    sampler.sample()
    assert len(rows) == 2


def test_unresolved_cuda_batch_holds_back_later_batches(
    isolated_timing_queue, monkeypatch
):
    class CUDAEvent:
        ready = False
        queries = 0

        def query(self):
            self.queries += 1
            return self.ready

        def elapsed_time(self, end):
            assert end.ready
            return 7.0

        def synchronize(self):
            pytest.fail("Timing handoff must not synchronize CUDA")

    start, end = CUDAEvent(), CUDAEvent()
    event = TimeEvent("gpu", "cuda:0", 1.0, 1.002, start, end, step=10)
    first = StepTimeBatch(10, [event])
    second = StepTimeBatch(11, [TimeEvent("cpu", "cpu", 2.0, 2.001, step=11)])
    returned = []
    monkeypatch.setattr(step_events, "return_cuda_event", returned.append)
    step_events.publish_step_time_batch(first)
    step_events.publish_step_time_batch(second)
    assert end.queries == 0

    sampler = StepTimeSampler()
    rows = []
    monkeypatch.setattr(sampler, "_add_record", rows.append)
    sampler.sample()
    assert rows == []
    assert sampler._pending[0] is first
    assert sampler._pending[1] is second
    assert sampler.has_pending_recording_data()
    assert returned == []

    end.ready = True
    sampler.sample()
    assert [row["step"] for row in rows] == [10, 11]
    assert rows[0]["events"]["gpu"]["cuda:0"]["gpu_ms"] == 7.0
    assert returned == [start, end]
    assert event.gpu_start is event.gpu_end is None
    assert not sampler.has_pending_recording_data()
    sampler.sample()
    assert len(rows) == 2
    assert returned == [start, end]


def test_full_timing_queue_drops_only_new_batch(isolated_timing_queue, capsys):
    # Exercise the unchanged production capacity and real nonblocking queue.
    batches = [StepTimeBatch(step) for step in range(2048)]
    for batch in batches:
        step_events.publish_step_time_batch(batch)
    step_events.publish_step_time_batch(StepTimeBatch(2048))

    drained = step_events.drain_step_time_batches()
    assert len(drained) == 2048
    assert all(
        actual is expected for actual, expected in zip(drained, batches)
    )
    assert "dropping step batch 2048" in capsys.readouterr().err
    assert step_events.drain_step_time_batches() == []
