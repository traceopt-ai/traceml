# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Process domain objects and pure helpers for the final-report section."""

from dataclasses import dataclass
from typing import Optional

MAX_SUMMARY_ROWS = 10_000

PROCESS_METRIC_NAMES = [
    "cpu_percent",
    "cpu_capacity_percent",
    "ram_bytes",
    "ram_percent",
    "gpu_mem_used_bytes",
    "gpu_mem_reserved_bytes",
    "gpu_mem_reserved_percent",
    "gpu_mem_headroom_bytes",
]


def cpu_capacity_percent(
    cpu_avg_percent: Optional[float],
    cpu_logical_core_count: Optional[int],
) -> Optional[float]:
    """Return average CPU usage normalized by logical host core capacity."""
    if (
        cpu_avg_percent is None
        or cpu_logical_core_count is None
        or cpu_logical_core_count <= 0
    ):
        return None
    return max(
        0.0,
        float(cpu_avg_percent)
        / (100.0 * float(cpu_logical_core_count))
        * 100.0,
    )


def _format_percent_stat(
    label: str,
    value: Optional[float],
) -> Optional[str]:
    if value is None:
        return None
    return f"{label} {float(value):.0f}%"


def _format_memory_stat(
    label: str,
    value_gb: Optional[float],
    total_gb: Optional[float],
) -> Optional[str]:
    if value_gb is None:
        return None
    if total_gb is None:
        return f"{label} {float(value_gb):.1f} GB"
    return f"{label} {float(value_gb):.1f} / {float(total_gb):.1f} GB"


@dataclass
class ProcessSummaryAgg:
    """
    Aggregated process metrics loaded from `process_samples`.

    Notes
    -----
    - Memory values remain in raw bytes during aggregation and are converted
      only at formatting / serialization time.
    - Fields are intentionally broad enough to support both single-process and
      distributed runs.
    """

    # Sample timestamps are Unix seconds from the traced process.
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    process_samples: int = 0

    distinct_global_ranks: int = 0

    cpu_avg_percent: Optional[float] = None
    cpu_peak_percent: Optional[float] = None
    cpu_logical_core_count: Optional[int] = None

    ram_avg_bytes: Optional[float] = None
    ram_peak_bytes: Optional[float] = None
    ram_total_bytes: Optional[float] = None

    gpu_available: Optional[bool] = None
    gpu_count: Optional[int] = None
    gpu_device_index: Optional[int] = None

    gpu_mem_used_avg_bytes: Optional[float] = None
    gpu_mem_used_peak_bytes: Optional[float] = None
    gpu_mem_reserved_avg_bytes: Optional[float] = None
    gpu_mem_reserved_peak_bytes: Optional[float] = None
    gpu_mem_total_bytes: Optional[float] = None
    # Max reserved/used CUDA memory ratio observed in the selected window.
    gpu_mem_reserved_overhang_ratio: Optional[float] = None


@dataclass
class PerRankProcessSummary:
    """
    Aggregated traced-process metrics for one global rank.

    Notes
    -----
    - Values are aggregated across all selected `process_samples` rows for the
      globally unique worker rank.
    - Memory values remain in raw bytes while aggregating and are converted only
      during final summary serialization.
    """

    global_rank: int
    local_rank: Optional[int] = None
    world_size: Optional[int] = None
    local_world_size: Optional[int] = None
    node_rank: Optional[int] = None
    hostname: Optional[str] = None

    cpu_avg_percent: Optional[float] = None
    cpu_peak_percent: Optional[float] = None
    cpu_logical_core_count: Optional[int] = None

    ram_avg_bytes: Optional[float] = None
    ram_peak_bytes: Optional[float] = None
    ram_total_bytes: Optional[float] = None

    gpu_available: Optional[bool] = None
    gpu_count: Optional[int] = None
    gpu_device_index: Optional[int] = None

    gpu_mem_used_avg_bytes: Optional[float] = None
    gpu_mem_used_peak_bytes: Optional[float] = None
    gpu_mem_reserved_avg_bytes: Optional[float] = None
    gpu_mem_reserved_peak_bytes: Optional[float] = None
    gpu_mem_total_bytes: Optional[float] = None
    # Max reserved/used CUDA memory ratio observed for this global rank.
    gpu_mem_reserved_overhang_ratio: Optional[float] = None


def process_cpu_capacity_percent(agg: ProcessSummaryAgg) -> Optional[float]:
    """Return average CPU capacity used by the aggregate process summary."""
    return cpu_capacity_percent(
        agg.cpu_avg_percent,
        agg.cpu_logical_core_count,
    )


def build_process_stats_line(
    agg: ProcessSummaryAgg,
    *,
    ram_peak_gb: Optional[float],
    ram_total_gb: Optional[float],
    gpu_mem_used_peak_pct: Optional[float],
    gpu_mem_reserved_peak_pct: Optional[float],
) -> str:
    """Build the compact Process stats line for the final text card."""
    gpu_pct = gpu_mem_reserved_peak_pct or gpu_mem_used_peak_pct
    gpu_label = (
        "GPU reserved peak"
        if gpu_mem_reserved_peak_pct is not None
        else "GPU used peak"
    )
    parts = [
        f"global ranks {agg.distinct_global_ranks}",
        _format_percent_stat("CPU avg", agg.cpu_avg_percent),
        _format_memory_stat("RSS peak", ram_peak_gb, ram_total_gb),
        _format_percent_stat(gpu_label, gpu_pct),
    ]
    rendered = [part for part in parts if part is not None]
    return " | ".join(rendered) if rendered else "unavailable"


__all__ = [
    "MAX_SUMMARY_ROWS",
    "PROCESS_METRIC_NAMES",
    "PerRankProcessSummary",
    "ProcessSummaryAgg",
    "build_process_stats_line",
    "cpu_capacity_percent",
    "process_cpu_capacity_percent",
]
