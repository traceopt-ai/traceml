from traceml.core import Registry
from traceml.runtime.sampler_registry import (
    SamplerSpec,
    build_samplers,
    select_sampler_specs,
)
from traceml.samplers.base_sampler import BaseSampler


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


def test_unknown_profile_keeps_only_profile_agnostic_samplers() -> None:
    assert _selected_keys(profile="unknown", mode="cli") == (
        "system",
        "process",
        "stdout_stderr",
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
