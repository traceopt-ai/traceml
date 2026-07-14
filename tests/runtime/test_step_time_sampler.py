import pytest

from traceml_ai.samplers.step_time_sampler import StepTimeSampler
from traceml_ai.utils.timing import StepTimeBatch, TimeEvent


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
