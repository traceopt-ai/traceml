"""Runtime sampler registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from traceml_ai.core import Registry
from traceml_ai.loggers.error_log import get_error_logger
from traceml_ai.samplers.base_sampler import BaseSampler
from traceml_ai.samplers.process_sampler import ProcessSampler
from traceml_ai.samplers.runtime_environment_sampler import (
    RuntimeEnvironmentSampler,
)
from traceml_ai.samplers.stdout_stderr_sampler import StdoutStderrSampler
from traceml_ai.samplers.step_memory_sampler import StepMemorySampler
from traceml_ai.samplers.step_time_sampler import StepTimeSampler
from traceml_ai.samplers.system_sampler import SystemSampler

SamplerFactory = Callable[[], BaseSampler]


@dataclass(frozen=True)
class SamplerSpec:
    """Declarative sampler registration."""

    key: str
    factory: SamplerFactory
    profiles: Optional[tuple[str, ...]] = None
    modes: Optional[tuple[str, ...]] = None
    rank_zero_only: bool = False
    drain_on_recording_stop: bool = False

    def enabled_for(
        self,
        *,
        profile: str,
        mode: str,
        is_ddp: bool,
        local_rank: int,
    ) -> bool:
        """Return True when this sampler should run for the rank context."""
        if self.profiles is not None and profile not in self.profiles:
            return False
        if self.modes is not None and mode not in self.modes:
            return False
        if self.rank_zero_only and is_ddp and int(local_rank) != 0:
            return False
        return True


def _spec(
    key: str,
    factory: SamplerFactory,
    *,
    profiles: Optional[Iterable[str]] = None,
    modes: Optional[Iterable[str]] = None,
    rank_zero_only: bool = False,
    drain_on_recording_stop: bool = False,
) -> tuple[str, SamplerSpec]:
    """Build one registry item with the key stored inside the spec."""
    spec = SamplerSpec(
        key=key,
        factory=factory,
        profiles=tuple(profiles) if profiles is not None else None,
        modes=tuple(modes) if modes is not None else None,
        rank_zero_only=rank_zero_only,
        drain_on_recording_stop=bool(drain_on_recording_stop),
    )
    return key, spec


DEFAULT_SAMPLER_REGISTRY: Registry[SamplerSpec] = Registry(
    [
        _spec("system", SystemSampler, rank_zero_only=True),
        _spec(
            "runtime_environment",
            RuntimeEnvironmentSampler,
            drain_on_recording_stop=True,
        ),
        _spec("process", ProcessSampler),
        _spec("stdout_stderr", StdoutStderrSampler, modes=("cli",)),
        _spec(
            "step_time",
            StepTimeSampler,
            profiles=("run",),
            drain_on_recording_stop=True,
        ),
        _spec(
            "step_memory",
            StepMemorySampler,
            profiles=("run",),
            drain_on_recording_stop=True,
        ),
    ]
)


def select_sampler_specs(
    *,
    profile: str,
    mode: str,
    is_ddp: bool,
    local_rank: int,
    registry: Registry[SamplerSpec] = DEFAULT_SAMPLER_REGISTRY,
) -> tuple[SamplerSpec, ...]:
    """Select sampler specs without instantiating sampler classes."""
    normalized_profile = str(profile or "run")
    normalized_mode = str(mode or "cli")
    return tuple(
        spec
        for spec in registry.all()
        if spec.enabled_for(
            profile=normalized_profile,
            mode=normalized_mode,
            is_ddp=bool(is_ddp),
            local_rank=int(local_rank),
        )
    )


def build_samplers(
    *,
    profile: str,
    mode: str,
    is_ddp: bool,
    local_rank: int,
    registry: Registry[SamplerSpec] = DEFAULT_SAMPLER_REGISTRY,
    logger=None,
) -> list[BaseSampler]:
    """Instantiate samplers for the rank."""
    log = logger or get_error_logger("TraceMLSamplerRegistry")
    samplers: list[BaseSampler] = []
    for spec in select_sampler_specs(
        profile=profile,
        mode=mode,
        is_ddp=is_ddp,
        local_rank=local_rank,
        registry=registry,
    ):
        try:
            sampler = spec.factory()
        except Exception as exc:
            log.exception(
                "[TraceML] Failed to initialize sampler %s: %s",
                spec.key,
                exc,
            )
            continue
        sampler.drain_on_recording_stop = bool(spec.drain_on_recording_stop)
        samplers.append(sampler)
    return samplers


__all__ = [
    "DEFAULT_SAMPLER_REGISTRY",
    "SamplerFactory",
    "SamplerSpec",
    "build_samplers",
    "select_sampler_specs",
]
