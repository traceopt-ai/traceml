import threading
import time

from traceml_ai.core import Registry
from traceml_ai.runtime.runtime import TraceMLRuntime
from traceml_ai.runtime.sampler_registry import (
    DEFAULT_SAMPLER_REGISTRY,
    SamplerSpec,
    build_samplers,
    select_sampler_specs,
)
from traceml_ai.runtime.settings import TraceMLSettings
from traceml_ai.runtime.state import (
    TraceMLRecordingStatus,
    configure_trace_recording,
    get_trace_recording_state,
    mark_trace_step_flushed,
)
from traceml_ai.samplers import process_sampler
from traceml_ai.samplers.base_sampler import BaseSampler
from traceml_ai.samplers.process_sampler import ProcessSampler


class _FakeSampler(BaseSampler):
    def __init__(self) -> None:
        super().__init__(
            sampler_name="FakeSampler",
            table_name="fake_samples",
        )
        self.sample_idx = 0

    def sample(self) -> None:
        self.sample_idx += 1
        return None


class _FakeLogger:
    def __init__(self) -> None:
        self.exceptions: list[tuple[str, tuple[object, ...]]] = []

    def exception(self, message: str, *args: object) -> None:
        self.exceptions.append((message, args))


class _FakePublisher:
    def __init__(self) -> None:
        self.published: list[tuple[str, ...]] = []

    def publish(self, samplers) -> None:
        self.published.append(
            tuple(getattr(s, "sampler_name", "") for s in samplers)
        )


class _PendingDrainSampler(_FakeSampler):
    def __init__(self) -> None:
        super().__init__()
        self.sampler_name = "PendingDrainSampler"
        self.drain_on_recording_stop = True
        self.pending = True

    def has_pending_recording_data(self) -> bool:
        return self.pending


def _selected_keys(
    *,
    profile: str,
    mode: str,
    local_rank: int = 0,
    world_size: int = 1,
) -> tuple[str, ...]:
    return tuple(
        spec.key
        for spec in select_sampler_specs(
            profile=profile,
            mode=mode,
            local_rank=local_rank,
            world_size=world_size,
        )
    )


def test_watch_cli_selects_host_and_process_samplers() -> None:
    assert _selected_keys(profile="watch", mode="cli") == (
        "system",
        "runtime_environment",
        "process",
    )


def test_watch_dashboard_omits_stdout_sampler() -> None:
    assert _selected_keys(profile="watch", mode="dashboard") == (
        "system",
        "runtime_environment",
        "process",
    )


def test_run_cli_selects_step_samplers() -> None:
    assert _selected_keys(profile="run", mode="cli") == (
        "system",
        "runtime_environment",
        "process",
        "step_time",
        "step_memory",
    )


def test_default_registry_has_no_deep_or_layer_samplers() -> None:
    specs = DEFAULT_SAMPLER_REGISTRY.all()
    all_keys = {spec.key for spec in specs}
    assert "deep" not in {
        profile for spec in specs for profile in (spec.profiles or ())
    }
    assert not any(key.startswith("layer_") for key in all_keys)


def test_nonzero_local_rank_skips_system_only_when_distributed() -> None:
    for world_size, system_enabled in ((1, True), (4, False)):
        selected = _selected_keys(
            profile="run",
            mode="cli",
            local_rank=1,
            world_size=world_size,
        )
        assert ("system" in selected) is system_enabled
        assert tuple(key for key in selected if key != "system") == (
            "runtime_environment",
            "process",
            "step_time",
            "step_memory",
        )


def test_local_rank_zero_keeps_node_leader_system_sampler() -> None:
    assert _selected_keys(
        profile="run",
        mode="cli",
        local_rank=0,
    ) == (
        "system",
        "runtime_environment",
        "process",
        "step_time",
        "step_memory",
    )


def test_unknown_profile_keeps_only_profile_agnostic_samplers() -> None:
    assert _selected_keys(profile="unknown", mode="cli") == (
        "system",
        "runtime_environment",
        "process",
    )


def test_custom_registry_filters_by_profile_mode_and_preserves_order() -> None:
    registry: Registry[SamplerSpec] = Registry(
        [
            (
                "all",
                SamplerSpec(key="all", factory=_FakeSampler),
            ),
            (
                "run_only",
                SamplerSpec(
                    key="run_only",
                    factory=_FakeSampler,
                    profiles=("run",),
                ),
            ),
            (
                "dashboard_only",
                SamplerSpec(
                    key="dashboard_only",
                    factory=_FakeSampler,
                    modes=("dashboard",),
                ),
            ),
            (
                "run_summary",
                SamplerSpec(
                    key="run_summary",
                    factory=_FakeSampler,
                    profiles=("run",),
                    modes=("summary",),
                ),
            ),
        ]
    )

    selected = select_sampler_specs(
        profile="run",
        mode="dashboard",
        local_rank=0,
        registry=registry,
    )

    assert tuple(spec.key for spec in selected) == (
        "all",
        "run_only",
        "dashboard_only",
    )


def test_build_samplers_logs_and_skips_constructor_failures() -> None:
    def _raise() -> _FakeSampler:
        raise RuntimeError("constructor failed")

    registry: Registry[SamplerSpec] = Registry(
        [
            ("ok", SamplerSpec(key="ok", factory=_FakeSampler)),
            ("bad", SamplerSpec(key="bad", factory=_raise)),
        ]
    )
    logger = _FakeLogger()

    samplers = build_samplers(
        profile="run",
        mode="cli",
        local_rank=0,
        registry=registry,
        logger=logger,
    )

    assert len(samplers) == 1
    assert isinstance(samplers[0], _FakeSampler)
    assert len(logger.exceptions) == 1
    assert "Failed to initialize sampler" in logger.exceptions[0][0]


def test_queue_backed_samplers_are_marked_for_final_recording_drain() -> None:
    specs = {
        spec.key: spec
        for spec in select_sampler_specs(
            profile="run",
            mode="summary",
            local_rank=0,
        )
    }

    assert specs["system"].drain_on_recording_stop is False
    assert specs["runtime_environment"].drain_on_recording_stop is True
    assert specs["process"].drain_on_recording_stop is False
    assert specs["step_time"].drain_on_recording_stop is True
    assert specs["step_memory"].drain_on_recording_stop is True


def test_runtime_final_recording_drain_skips_periodic_samplers() -> None:
    periodic = _FakeSampler()
    periodic.sampler_name = "PeriodicSampler"
    drain = _FakeSampler()
    drain.sampler_name = "DrainSampler"
    drain.drain_on_recording_stop = True

    publisher = _FakePublisher()
    runtime = TraceMLRuntime.__new__(TraceMLRuntime)
    runtime._samplers = [periodic, drain]
    runtime._publisher = publisher
    runtime._logger = _FakeLogger()

    configure_trace_recording(max_steps=1)
    mark_trace_step_flushed(1)
    runtime._drain_recording_samplers()

    assert periodic.sample_idx == 0
    assert drain.sample_idx == 1
    assert publisher.published == [("DrainSampler",)]
    assert (
        get_trace_recording_state().status == TraceMLRecordingStatus.COMPLETE
    )

    configure_trace_recording()


def test_runtime_recording_drain_waits_for_pending_timing_data() -> None:
    drain = _PendingDrainSampler()
    publisher = _FakePublisher()
    runtime = TraceMLRuntime.__new__(TraceMLRuntime)
    runtime._samplers = [drain]
    runtime._publisher = publisher
    runtime._logger = _FakeLogger()

    configure_trace_recording(max_steps=1)
    mark_trace_step_flushed(1)
    runtime._drain_recording_samplers()

    assert drain.sample_idx == 1
    assert (
        get_trace_recording_state().status == TraceMLRecordingStatus.DRAINING
    )

    drain.pending = False
    runtime._drain_recording_samplers()

    assert drain.sample_idx == 2
    assert (
        get_trace_recording_state().status == TraceMLRecordingStatus.COMPLETE
    )

    configure_trace_recording()


class _ClearsOnSecondSampleSampler(_PendingDrainSampler):
    def sample(self) -> None:
        super().sample()
        if self.sample_idx >= 2:
            self.pending = False


def _stopped_runtime(samplers, interval_sec: float) -> TraceMLRuntime:
    runtime = TraceMLRuntime.__new__(TraceMLRuntime)
    runtime._samplers = samplers
    runtime._publisher = _FakePublisher()
    runtime._logger = _FakeLogger()
    runtime._settings = TraceMLSettings(sampler_interval_sec=interval_sec)
    runtime._stop_event = threading.Event()
    runtime._stop_event.set()
    return runtime


def test_normal_stop_ticks_pending_drain_sampler_until_clear() -> None:
    periodic = _FakeSampler()
    periodic.sampler_name = "PeriodicSampler"
    drain = _ClearsOnSecondSampleSampler()
    runtime = _stopped_runtime([periodic, drain], interval_sec=0.05)

    configure_trace_recording()
    assert (
        get_trace_recording_state().status == TraceMLRecordingStatus.RECORDING
    )
    runtime._sampler_loop()

    assert drain.sample_idx == 2
    assert not drain.has_pending_recording_data()
    assert periodic.sample_idx == 1
    assert runtime._publisher.published == [
        ("PeriodicSampler", "PendingDrainSampler"),
        ("PendingDrainSampler",),
    ]

    configure_trace_recording()


def test_normal_stop_drain_is_bounded_and_never_synchronizes(
    monkeypatch,
) -> None:
    sync_calls: list[object] = []
    monkeypatch.setattr(
        process_sampler.torch.cuda,
        "synchronize",
        lambda *args, **kwargs: sync_calls.append(args),
    )
    interval_sec = 0.05
    bound_sec = min(2.0, 2 * interval_sec)
    drain = _PendingDrainSampler()
    runtime = _stopped_runtime([drain], interval_sec=interval_sec)

    configure_trace_recording()
    started = time.monotonic()
    runtime._sampler_loop()
    elapsed = time.monotonic() - started

    assert bound_sec <= elapsed < bound_sec + 0.25
    assert drain.sample_idx > 1
    assert drain.has_pending_recording_data()
    assert sync_calls == []

    configure_trace_recording()


def test_process_sampler_does_not_set_cuda_device_when_unavailable(
    monkeypatch,
) -> None:
    set_device_calls: list[int] = []

    monkeypatch.setattr(
        process_sampler.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        process_sampler.torch.cuda,
        "device_count",
        lambda: 0,
    )
    monkeypatch.setattr(
        process_sampler.torch.cuda,
        "set_device",
        lambda device_index: set_device_calls.append(device_index),
    )

    sampler = ProcessSampler()

    assert sampler._sample_gpu() is None
    assert sampler.gpu_available is False
    assert sampler.gpu_count == 0
    assert set_device_calls == []


def test_process_sampler_does_not_touch_cuda_before_ddp_init(
    monkeypatch,
) -> None:
    cuda_calls: list[str] = []

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(process_sampler.dist, "is_available", lambda: True)
    monkeypatch.setattr(process_sampler.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        process_sampler.torch.cuda,
        "is_available",
        lambda: cuda_calls.append("is_available") or True,
    )
    monkeypatch.setattr(
        process_sampler.torch.cuda,
        "device_count",
        lambda: cuda_calls.append("device_count") or 1,
    )
    monkeypatch.setattr(
        process_sampler.torch.cuda,
        "set_device",
        lambda device_index: cuda_calls.append(f"set_device:{device_index}"),
    )

    sampler = ProcessSampler()

    assert sampler._sample_gpu() is None
    assert cuda_calls == []


def test_build_samplers_continues_after_constructor_failure() -> None:
    def _raise() -> _FakeSampler:
        raise RuntimeError("constructor failed")

    registry: Registry[SamplerSpec] = Registry(
        [
            ("before", SamplerSpec(key="before", factory=_FakeSampler)),
            ("bad", SamplerSpec(key="bad", factory=_raise)),
            ("after", SamplerSpec(key="after", factory=_FakeSampler)),
        ]
    )
    logger = _FakeLogger()

    samplers = build_samplers(
        profile="run",
        mode="cli",
        local_rank=0,
        registry=registry,
        logger=logger,
    )

    assert [sampler.sampler_name for sampler in samplers] == [
        "FakeSampler",
        "FakeSampler",
    ]
    assert len(logger.exceptions) == 1
