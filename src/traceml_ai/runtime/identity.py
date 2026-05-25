# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Canonical runtime identity for TraceML training processes.

This module owns the answer to "which training process am I?". Keeping that
logic here avoids scattering launcher-specific environment parsing through the
runtime, samplers, and telemetry layers.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

EnvMapping = Mapping[str, str]
TorchLoader = Callable[[], Optional[Any]]


def _load_torch() -> Optional[Any]:
    """Import torch only when distributed state needs it."""
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


def _env_int(
    env: EnvMapping,
    name: str,
    default: Optional[int] = None,
) -> Optional[int]:
    """Read an integer environment variable best-effort."""
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _torch_distributed_ready(torch: Optional[Any]) -> bool:
    """Return True when torch.distributed is initialized."""
    if torch is None:
        return False
    try:
        return bool(
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
    except Exception:
        return False


def _torch_distributed_int(
    torch: Optional[Any],
    method_name: str,
) -> Optional[int]:
    """Read an integer from torch.distributed when initialized."""
    if not _torch_distributed_ready(torch):
        return None
    try:
        method = getattr(torch.distributed, method_name)
        return int(method())
    except Exception:
        return None


def _default_node_rank(
    *,
    global_rank: int,
    local_world_size: int,
) -> int:
    """Infer node rank from global rank when no launcher value is present."""
    if local_world_size <= 0:
        return 0
    return max(0, int(global_rank) // int(local_world_size))


@dataclass(frozen=True)
class RuntimeIdentity:
    """Process identity resolved from distributed launcher metadata."""

    global_rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    node_rank: int
    hostname: str
    pid: int

    @property
    def rank(self) -> int:
        """
        Backward-compatible TraceML rank.

        Current TraceML runtime paths historically use local rank for sampler
        selection, sender metadata, and rank-local directories. Multi-node
        storage/reporting will migrate to ``global_rank`` in follow-up work.
        """
        return self.local_rank

    @property
    def is_distributed(self) -> bool:
        """Return True when the identity describes a multi-process job."""
        return self.world_size > 1

    @property
    def is_multinode(self) -> bool:
        """Return True when world size exceeds local world size."""
        return self.world_size > self.local_world_size

    def to_meta(self) -> dict[str, Any]:
        """Serialize identity for telemetry metadata envelopes."""
        return {
            "rank": self.rank,
            "global_rank": self.global_rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_rank": self.node_rank,
            "hostname": self.hostname,
            "pid": self.pid,
        }


@dataclass(frozen=True)
class RuntimeIdentityResolver:
    """
    Resolve process identity from launcher and distributed runtime metadata.

    The resolver is dependency-injected so tests and future launchers can supply
    their own environment/runtime source without changing TraceMLRuntime.
    """

    env: EnvMapping
    torch_loader: TorchLoader = _load_torch

    def resolve(self) -> RuntimeIdentity:
        """
        Build a RuntimeIdentity with single-process-safe defaults.

        Resolution order is intentionally conservative:
        1. launcher environment variables
        2. initialized torch.distributed state
        3. safe local defaults
        """
        torch = self.torch_loader()

        global_rank = self._global_rank(torch)
        world_size = self._world_size(torch)
        local_world_size = self._local_world_size(world_size)
        local_rank = self._local_rank(
            torch=torch,
            global_rank=global_rank,
            world_size=world_size,
            local_world_size=local_world_size,
        )
        node_rank = self._node_rank(
            global_rank=global_rank,
            local_world_size=local_world_size,
        )

        return RuntimeIdentity(
            global_rank=int(global_rank),
            local_rank=int(local_rank),
            world_size=int(world_size),
            local_world_size=int(local_world_size),
            node_rank=int(node_rank),
            hostname=socket.gethostname(),
            pid=os.getpid(),
        )

    def _global_rank(self, torch: Optional[Any]) -> int:
        value = _env_int(self.env, "RANK", None)
        if value is None:
            value = _torch_distributed_int(torch, "get_rank")
        return int(value) if value is not None else 0

    def _world_size(self, torch: Optional[Any]) -> int:
        value = _env_int(self.env, "WORLD_SIZE", None)
        if value is None:
            value = _torch_distributed_int(torch, "get_world_size")
        if value is None:
            value = 1
        return max(1, int(value))

    def _local_world_size(
        self,
        world_size: int,
    ) -> int:
        value = _env_int(self.env, "LOCAL_WORLD_SIZE", None)
        if value is None:
            value = int(world_size) if int(world_size) == 1 else 1
        return max(1, int(value))

    def _local_rank(
        self,
        *,
        torch: Optional[Any],
        global_rank: int,
        world_size: int,
        local_world_size: int,
    ) -> int:
        value = _env_int(self.env, "LOCAL_RANK", None)
        if value is None:
            if world_size > 1 or _torch_distributed_ready(torch):
                value = int(global_rank) % int(local_world_size)
            else:
                value = 0
        return max(0, int(value))

    def _node_rank(
        self,
        *,
        global_rank: int,
        local_world_size: int,
    ) -> int:
        value = _env_int(self.env, "GROUP_RANK", None)
        if value is None:
            value = _env_int(self.env, "NODE_RANK", None)
        if value is None:
            value = _default_node_rank(
                global_rank=global_rank,
                local_world_size=local_world_size,
            )
        return int(value)


def resolve_runtime_identity(
    *,
    env: Optional[EnvMapping] = None,
    torch_loader: Optional[TorchLoader] = None,
) -> RuntimeIdentity:
    """Resolve the current process identity using the default resolver."""
    resolver = RuntimeIdentityResolver(
        env=env if env is not None else os.environ,
        torch_loader=torch_loader if torch_loader is not None else _load_torch,
    )
    return resolver.resolve()


__all__ = [
    "EnvMapping",
    "RuntimeIdentity",
    "RuntimeIdentityResolver",
    "resolve_runtime_identity",
]
