from traceml_ai.core import Registry
from traceml_ai.runtime.sampler_registry import (
    SamplerSpec,
    build_samplers,
    select_sampler_specs,
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

    def sample(self) -> None:
        return None


class _FakeLogger:
    def __init__(self) -> None:
        self.exceptions: list[tuple[str, tuple[object, ...]]] = []

    def exception(self, message: str, *args: object) -> None:
        self.exceptions.append((message, args))


def _selected_keys(
    *,
    profile: str,
    mode: str,
    is_ddp: bool = False,
    local_rank: int = 0,
) -> tuple[str, ...]:
    return tuple(
        spec.key
        for spec in select_sampler_specs(
            profile=profile,
            mode=mode,
            is_ddp=is_ddp,
            local_rank=local_rank,
        )
    )


def test_watch_cli_selects_host_process_and_stdout_samplers() -> None:
    assert _selected_keys(profile="watch", mode="cli") == (
        "system",
        "process",
        "stdout_stderr",
    )


def test_watch_dashboard_omits_stdout_sampler() -> None:
    assert _selected_keys(profile="watch", mode="dashboard") == (
        "system",
        "process",
    )


def test_run_cli_selects_step_samplers() -> None:
    assert _selected_keys(profile="run", mode="cli") == (
        "system",
        "process",
        "stdout_stderr",
        "step_time",
        "step_memory",
    )


def test_deep_profile_selects_layer_samplers_after_step_samplers() -> None:
    assert _selected_keys(profile="deep", mode="summary") == (
        "system",
        "process",
        "step_time",
        "step_memory",
        "layer_memory",
        "layer_forward_memory",
        "layer_backward_memory",
        "layer_forward_time",
        "layer_backward_time",
    )


def test_ddp_nonzero_rank_skips_rank_zero_only_system_sampler() -> None:
    assert _selected_keys(
        profile="run",
        mode="cli",
        is_ddp=True,
        local_rank=1,
    ) == (
        "process",
        "stdout_stderr",
        "step_time",
        "step_memory",
    )


def test_ddp_rank_zero_keeps_rank_zero_only_system_sampler() -> None:
    assert _selected_keys(
        profile="run",
        mode="cli",
        is_ddp=True,
        local_rank=0,
    ) == (
        "system",
        "process",
        "stdout_stderr",
        "step_time",
        "step_memory",
    )


def test_unknown_profile_keeps_only_profile_agnostic_samplers() -> None:
    assert _selected_keys(profile="unknown", mode="cli") == (
        "system",
        "process",
        "stdout_stderr",
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
                "deep_cli",
                SamplerSpec(
                    key="deep_cli",
                    factory=_FakeSampler,
                    profiles=("deep",),
                    modes=("cli",),
                ),
            ),
        ]
    )

    selected = select_sampler_specs(
        profile="run",
        mode="dashboard",
        is_ddp=False,
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
        is_ddp=False,
        local_rank=0,
        registry=registry,
        logger=logger,
    )

    assert len(samplers) == 1
    assert isinstance(samplers[0], _FakeSampler)
    assert len(logger.exceptions) == 1
    assert "Failed to initialize sampler" in logger.exceptions[0][0]


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
        is_ddp=False,
        local_rank=0,
        registry=registry,
        logger=logger,
    )

    assert [sampler.sampler_name for sampler in samplers] == [
        "FakeSampler",
        "FakeSampler",
    ]
    assert len(logger.exceptions) == 1
