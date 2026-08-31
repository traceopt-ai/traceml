# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
TraceML settings (shared configuration schema).

This module defines the shared configuration dataclasses used by:
- CLI launcher (sets env vars)
- executor (reads env vars, constructs settings)
- runtime (per-rank agent)
- aggregator (out-of-process server + UI)

"""

import os
from dataclasses import dataclass
from typing import Optional

from traceml_ai.telemetry.retention import DEFAULT_HISTORY_RETENTION_S

DEFAULT_FINALIZE_TIMEOUT_SEC = 300.0
# The public ``interval`` setting uses this value unless explicitly overridden
# through the CLI, environment, YAML, or an integration-specific config.
DEFAULT_INTERVAL_SEC = 2.0
# Summary is the safe, topology-independent display default. Live CLI and
# dashboard rendering remain explicit opt-ins.
DEFAULT_UI_MODE = "summary"
MISSING_AGGREGATOR_POLICIES = ("raise", "warn")


def resolve_on_missing_aggregator(
    value: Optional[str],
    *,
    default: str,
) -> str:
    """Resolve explicit value, environment, then the frontend default."""
    resolved = value
    if resolved is None:
        resolved = os.environ.get("TRACEML_ON_MISSING_AGGREGATOR")
    if resolved is None:
        resolved = default

    normalized = str(resolved).strip().lower()
    if normalized not in MISSING_AGGREGATOR_POLICIES:
        raise ValueError(
            "on_missing_aggregator must be 'raise' or 'warn', got "
            f"{normalized!r}."
        )
    return normalized


@dataclass(frozen=True)
class AggregatorEndpoint:
    """Reachable TraceML aggregator endpoint for worker runtimes."""

    host: str
    port: int
    session_id: str


@dataclass(frozen=True)
class AggregatorTransportSettings:
    """
    Aggregator endpoint used by workers and the aggregator process.

    ``connect_host`` is used by training workers. ``bind_host`` is used by the
    aggregator process. They are the same on simple local runs, but different
    on multi-node runs where the aggregator binds ``0.0.0.0`` and workers
    connect to node 0's reachable address.
    """

    connect_host: str = "127.0.0.1"
    bind_host: str = "127.0.0.1"
    port: int = 29765


@dataclass(frozen=True)
class TraceMLSettings:
    """
    High-level TraceML settings shared across runtime and aggregator.

    Notes:
    - `sampler_interval_sec` controls worker sampling cadence (all ranks).
    - `render_interval_sec` controls aggregator UI cadence only; TCP telemetry
      is drained as soon as data arrives.
    - `mode` selects display backend and capture behavior ("cli" | "summary" | "dashboard").
    - `summary` mode disables live rendering and prints only the final
      end-of-run summary.
    - Aggregator transport is used for telemetry, including rank0 -> rank0
      loopback on local runs.
    """

    profile: str = "run"
    mode: str = DEFAULT_UI_MODE
    sampler_interval_sec: float = DEFAULT_INTERVAL_SEC
    render_interval_sec: float = DEFAULT_INTERVAL_SEC
    logs_dir: str = "./logs"
    enable_logging: bool = False
    dashboard_port: int = 8765
    dashboard_auto_open: bool = True
    aggregator: AggregatorTransportSettings = AggregatorTransportSettings()
    session_id: str = ""
    history_enabled: bool = True
    history_retention_s: float = DEFAULT_HISTORY_RETENTION_S
    db_path: str = ""
    trace_max_steps: Optional[int] = None
    html_report: bool = False
    finalize_timeout_sec: float = DEFAULT_FINALIZE_TIMEOUT_SEC
    expected_world_size: int = 1
