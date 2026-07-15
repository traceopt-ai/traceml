# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Runtime environment classification for a TraceML training rank.

This module classifies the local runtime environment so a future sampler can
persist rank-scoped context such as topology, torch.distributed state, and the
observed training strategy. It is not transport metadata: TCP envelope `meta`
continues to describe the sender, while this data belongs in a sampler body row.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from traceml_ai.runtime.identity import (
    RuntimeIdentity,
    resolve_runtime_identity,
)

EnvMapping = Mapping[str, str]
TorchLoader = Callable[[], Optional[Any]]


def _load_torch() -> Optional[Any]:
    """Import torch lazily so importing this module has no torch side effects."""
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


@dataclass(frozen=True)
class RuntimeEnvironmentInfo:
    """Rank-local runtime environment classification.

    `topology` describes process placement: single process, single-node
    multi-process, or multi-node. `training_strategy` describes how the model is
    parallelized when TraceML can observe it, for example DDP or FSDP.
    """

    topology: str
    distributed_initialized: bool
    distributed_backend: Optional[str]
    training_strategy: str
    strategy_source: str
    strategy_confidence: str

    def to_record(self) -> dict[str, Any]:
        """Return a JSON/msgpack-safe body row for future sampler transport."""
        return {
            "topology": self.topology,
            "distributed_initialized": bool(self.distributed_initialized),
            "distributed_backend": self.distributed_backend,
            "training_strategy": self.training_strategy,
            "strategy_source": self.strategy_source,
            "strategy_confidence": self.strategy_confidence,
        }


def detect_runtime_environment(
    model: Optional[Any] = None,
    *,
    env: Optional[EnvMapping] = None,
    torch_loader: Optional[TorchLoader] = None,
) -> RuntimeEnvironmentInfo:
    """Detect runtime topology and training strategy best-effort.

    The detector is safe to call from `trace_step(model)`: it does not probe
    CUDA, does not synchronize, imports torch lazily, and degrades to
    conservative unknown values if torch or distributed state cannot be read.
    """
    torch = _safe_load_torch(torch_loader)
    identity = _safe_resolve_identity(env=env, torch=torch)
    topology = _topology(identity)
    distributed_initialized, distributed_backend = _distributed_state(torch)

    model_strategy = _training_strategy_from_model(torch, model)
    if model_strategy is not None:
        training_strategy, strategy_source, strategy_confidence = (
            model_strategy
        )
    elif topology == "single_process":
        training_strategy = "single_process"
        strategy_source = "topology"
        strategy_confidence = "high"
    elif distributed_initialized:
        training_strategy = "distributed_unknown"
        strategy_source = "runtime_distributed"
        strategy_confidence = "low"
    elif identity.world_size > 1:
        training_strategy = "distributed_unknown"
        strategy_source = "topology"
        strategy_confidence = "low"
    else:
        training_strategy = "unknown"
        strategy_source = "unknown"
        strategy_confidence = "low"

    return RuntimeEnvironmentInfo(
        topology=topology,
        distributed_initialized=distributed_initialized,
        distributed_backend=distributed_backend,
        training_strategy=training_strategy,
        strategy_source=strategy_source,
        strategy_confidence=strategy_confidence,
    )


def _safe_load_torch(torch_loader: Optional[TorchLoader]) -> Optional[Any]:
    loader = torch_loader if torch_loader is not None else _load_torch
    try:
        return loader()
    except Exception:
        return None


def _safe_resolve_identity(
    *,
    env: Optional[EnvMapping],
    torch: Optional[Any],
) -> RuntimeIdentity:
    try:
        return resolve_runtime_identity(
            env=env if env is not None else os.environ,
            torch_loader=lambda: torch,
        )
    except Exception:
        return RuntimeIdentity(
            global_rank=0,
            local_rank=0,
            world_size=1,
            local_world_size=1,
            node_rank=0,
            hostname=socket.gethostname(),
            pid=os.getpid(),
        )


def _topology(identity: RuntimeIdentity) -> str:
    if identity.world_size <= 1:
        return "single_process"
    if identity.is_multinode:
        return "multi_node"
    return "single_node_multi_process"


def _distributed_state(torch: Optional[Any]) -> tuple[bool, Optional[str]]:
    dist = getattr(torch, "distributed", None) if torch is not None else None
    if dist is None:
        return False, None

    try:
        initialized = bool(dist.is_available() and dist.is_initialized())
    except Exception:
        return False, None
    if not initialized:
        return False, None

    try:
        backend = dist.get_backend()
    except Exception:
        backend = None

    return True, str(backend) if backend is not None else None


def _training_strategy_from_model(
    torch: Optional[Any],
    model: Optional[Any],
) -> Optional[tuple[str, str, str]]:
    if model is None or torch is None:
        return None

    if _isinstance_path(
        model,
        torch,
        "nn.parallel.DistributedDataParallel",
    ):
        return "ddp", "runtime_model", "high"

    if _isinstance_path(
        model,
        torch,
        "distributed.fsdp.FullyShardedDataParallel",
    ) or _isinstance_path(
        model,
        torch,
        "distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel",
    ):
        return "fsdp", "runtime_model", "high"

    return None


def _isinstance_path(model: Any, root: Any, dotted_path: str) -> bool:
    cls = _attr_path(root, dotted_path)
    if cls is None:
        return False
    try:
        return isinstance(model, cls)
    except TypeError:
        return False


def _attr_path(root: Any, dotted_path: str) -> Optional[Any]:
    current = root
    for part in dotted_path.split("."):
        try:
            current = getattr(current, part)
        except Exception:
            return None
    return current


__all__ = [
    "RuntimeEnvironmentInfo",
    "detect_runtime_environment",
]
