# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Typed launch configuration for TraceML CLI runs.

The launcher has two jobs that are easy to mix up:
- start torchrun with the right distributed arguments
- decide where the TraceML aggregator lives

Keeping those decisions in small value objects keeps the command handler
focused on orchestration and makes multi-node behavior easier to test.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any


def _positive_int(value: Any, name: str) -> int:
    """Return a positive integer or raise a user-facing ValueError."""
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1")
    return parsed


def _non_negative_int(value: Any, name: str) -> int:
    """Return a non-negative integer or raise a user-facing ValueError."""
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


@dataclass(frozen=True)
class TorchrunLaunchConfig:
    """Distributed launch arguments passed to ``torchrun``."""

    nnodes: int = 1
    nproc_per_node: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    @classmethod
    def from_args(cls, args: Any) -> "TorchrunLaunchConfig":
        """Build and validate torchrun launch settings from argparse args."""
        nnodes = _positive_int(getattr(args, "nnodes", 1), "--nnodes")
        nproc_per_node = _positive_int(
            getattr(args, "nproc_per_node", 1),
            "--nproc-per-node",
        )
        node_rank = _non_negative_int(
            getattr(args, "node_rank", 0),
            "--node-rank",
        )
        if node_rank >= nnodes:
            raise ValueError("--node-rank must be less than --nnodes")

        master_addr = str(getattr(args, "master_addr", "127.0.0.1") or "")
        if not master_addr:
            raise ValueError("--master-addr cannot be empty")

        master_port = _positive_int(
            getattr(args, "master_port", 29500),
            "--master-port",
        )

        return cls(
            nnodes=nnodes,
            nproc_per_node=nproc_per_node,
            node_rank=node_rank,
            master_addr=master_addr,
            master_port=master_port,
        )

    def to_command(self) -> list[str]:
        """Return the Python-module form of the torchrun command."""
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={self.nnodes}",
            f"--nproc_per_node={self.nproc_per_node}",
            f"--node_rank={self.node_rank}",
            f"--master_addr={self.master_addr}",
            f"--master_port={self.master_port}",
        ]


@dataclass(frozen=True)
class AggregatorLaunchConfig:
    """TraceML aggregator address and ownership policy for a launch."""

    connect_host: str
    bind_host: str
    port: int
    owner_node_rank: int = 0

    @classmethod
    def from_args(
        cls,
        args: Any,
        *,
        torchrun: TorchrunLaunchConfig,
    ) -> "AggregatorLaunchConfig":
        """Build aggregator settings from CLI args and torchrun defaults."""
        connect_host = str(
            getattr(args, "aggregator_host", None) or torchrun.master_addr
        )
        default_bind_host = "0.0.0.0" if torchrun.nnodes > 1 else "127.0.0.1"
        bind_host = str(
            getattr(args, "aggregator_bind_host", None) or default_bind_host
        )
        if not connect_host:
            raise ValueError("--aggregator-host cannot be empty")
        if not bind_host:
            raise ValueError("--aggregator-bind-host cannot be empty")

        return cls(
            connect_host=connect_host,
            bind_host=bind_host,
            port=_positive_int(getattr(args, "tcp_port", 29765), "--tcp-port"),
        )

    def is_owner(self, *, node_rank: int) -> bool:
        """Return True when this launcher should start the aggregator."""
        return int(node_rank) == int(self.owner_node_rank)


@dataclass(frozen=True)
class DistributedLaunchConfig:
    """Complete distributed launch configuration used by the CLI handler."""

    torchrun: TorchrunLaunchConfig
    aggregator: AggregatorLaunchConfig

    @classmethod
    def from_args(cls, args: Any) -> "DistributedLaunchConfig":
        """Build a complete launch config from argparse args."""
        torchrun = TorchrunLaunchConfig.from_args(args)
        session_id = str(getattr(args, "session_id", "") or "").strip()
        if torchrun.nnodes > 1 and not session_id:
            raise ValueError(
                "--session-id is required when --nnodes > 1 so all nodes "
                "write into the same TraceML session."
            )
        aggregator = AggregatorLaunchConfig.from_args(
            args,
            torchrun=torchrun,
        )
        return cls(torchrun=torchrun, aggregator=aggregator)


__all__ = [
    "AggregatorLaunchConfig",
    "DistributedLaunchConfig",
    "TorchrunLaunchConfig",
]
