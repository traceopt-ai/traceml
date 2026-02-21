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
    """TCP configuration for TraceML telemetry transport."""

    host: str = "127.0.0.1"
    port: int = 29765


@dataclass(frozen=True)
class TraceMLSettings:
    """
    High-level TraceML settings shared across runtime and aggregator.

    Notes:
    - `sampler_interval_sec` controls sampling cadence (all ranks).
    - `render_interval_sec` controls UI cadence (aggregator only).
    - `mode` selects display backend and capture behavior ("cli" | "notebook" | "dashboard").
    - TCP is used for telemetry transport (including rank0 -> rank0 loopback).
    """

    mode: str = "cli"
    sampler_interval_sec: float = 1.0
    render_interval_sec: float = 1.0
    num_display_layers: int = 20
    logs_dir: str = "./logs"
    enable_logging: bool = False
    remote_max_rows: int = 200
    tcp: TraceMLTCPSettings = TraceMLTCPSettings()
    session_id: str = ""

