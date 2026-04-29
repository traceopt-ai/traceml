"""
Runtime sampler registry.

This module owns the mapping from TraceML profiles/modes to sampler classes.
The runtime agent asks for sampler instances; it does not need to know which
sampler classes belong to each profile.

Extension notes
---------------
To add a sampler:
1. Add a ``SamplerSpec`` to ``DEFAULT_SAMPLER_REGISTRY``.
2. Add a SQLite projection and renderer/summary support only if the sampler is
   user-facing.
3. Add selection tests for profile, mode, and DDP rank behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from traceml.core import Registry
from traceml.loggers.error_log import get_error_logger
from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.layer_backward_memory_sampler import (
    LayerBackwardMemorySampler,
)
from traceml.samplers.layer_backward_time_sampler import (
    LayerBackwardTimeSampler,
)
from traceml.samplers.layer_forward_memory_sampler import (
    LayerForwardMemorySampler,
)
from traceml.samplers.layer_forward_time_sampler import LayerForwardTimeSampler
from traceml.samplers.layer_memory_sampler import LayerMemorySampler
from traceml.samplers.process_sampler import ProcessSampler
from traceml.samplers.stdout_stderr_sampler import StdoutStderrSampler
from traceml.samplers.step_memory_sampler import StepMemorySampler
from traceml.samplers.step_time_sampler import StepTimeSampler
from traceml.samplers.system_sampler import SystemSampler

SamplerFactory = Callable[[], BaseSampler]


@dataclass(frozen=True)
class SamplerSpec:
    """
    Declarative sampler registration.

    Attributes
    ----------
    key:
        Stable internal registry key.
    factory:
        Zero-argument callable that builds the sampler instance.
    profiles:
        Profiles where this sampler is enabled. ``None`` means all profiles.
    modes:
        Display/runtime modes where this sampler is enabled. ``None`` means all
        modes.
    rank_zero_only:
        If True, run the sampler only on rank 0 in DDP. This is used for host
        telemetry where duplicate per-rank sampling adds noise without value.
    """

    key: str
    factory: SamplerFactory
    profiles: Optional[tuple[str, ...]] = None
    modes: Optional[tuple[str, ...]] = None
    rank_zero_only: bool = False

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
) -> tuple[str, SamplerSpec]:
    """
    Build one registry item with the key stored inside the spec as well.
    """
    spec = SamplerSpec(
        key=key,
        factory=factory,
        profiles=tuple(profiles) if profiles is not None else None,
        modes=tuple(modes) if modes is not None else None,
        rank_zero_only=rank_zero_only,
    )
    return key, spec


DEFAULT_SAMPLER_REGISTRY: Registry[SamplerSpec] = Registry(
    [
        _spec("system", SystemSampler, rank_zero_only=True),
        _spec("process", ProcessSampler),
        _spec("stdout_stderr", StdoutStderrSampler, modes=("cli",)),
        _spec("step_time", StepTimeSampler, profiles=("run", "deep")),
        _spec("step_memory", StepMemorySampler, profiles=("run", "deep")),
        _spec("layer_memory", LayerMemorySampler, profiles=("deep",)),
        _spec(
            "layer_forward_memory",
            LayerForwardMemorySampler,
            profiles=("deep",),
        ),
        _spec(
            "layer_backward_memory",
            LayerBackwardMemorySampler,
            profiles=("deep",),
        ),
        _spec(
            "layer_forward_time", LayerForwardTimeSampler, profiles=("deep",)
        ),
        _spec(
            "layer_backward_time",
            LayerBackwardTimeSampler,
            profiles=("deep",),
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
    """
    Select sampler specs for the rank without instantiating samplers.

    Keeping selection pure gives contributors a fast unit-test target and keeps
    profile/mode semantics independent from sampler constructor side effects.
    """
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
    """
    Instantiate samplers for the rank.

    Sampler construction is fail-open. If an optional sampler cannot be created
    because a host library, device API, or filesystem path is unavailable, the
    error is logged and the runtime continues with the remaining samplers. This
    preserves TraceML's core guarantee: telemetry must not break user training.
    """
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
        samplers.append(sampler)
    return samplers


__all__ = [
    "DEFAULT_SAMPLER_REGISTRY",
    "SamplerFactory",
    "SamplerSpec",
    "build_samplers",
    "select_sampler_specs",
]
