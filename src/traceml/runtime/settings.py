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

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceMLTCPSettings:
    """
    TCP configuration for TraceML telemetry transport.

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
    - `sampler_interval_sec` controls sampling cadence (all ranks).
    - `render_interval_sec` controls UI cadence (aggregator only).
    - `mode` selects display backend and capture behavior ("cli" | "summary" | "dashboard").
    - `summary` mode disables live rendering and prints only the final
      end-of-run summary.
    - TCP is used for telemetry transport (including rank0 -> rank0 loopback).
    """

    profile: str = "run"  # "deep"
    mode: str = "cli"
    sampler_interval_sec: float = 1.0
    render_interval_sec: float = 1.0
    num_display_layers: int = 20
    logs_dir: str = "./logs"
    enable_logging: bool = False
    remote_max_rows: int = 200
    tcp: TraceMLTCPSettings = TraceMLTCPSettings()
    session_id: str = ""
    history_enabled: bool = True
    db_path: str = ""
