# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Payload builder for the final-report process section."""

from __future__ import annotations

from typing import Any, Dict

from traceml.diagnostics.common import diagnosis_to_dict
from traceml.diagnostics.process import build_process_diagnosis_result
from traceml.diagnostics.process.policy import DEFAULT_PROCESS_POLICY
from traceml.reporting.sections.process.loader import ProcessSectionData
from traceml.reporting.sections.process.model import (
    PerRankProcessSummary,
    ProcessSummaryAgg,
    _band_name,
    _build_stats_line,
    _cpu_capacity_percent,
    _global_rank_rollup_to_json,
    _per_rank_to_diagnosis_input,
    _per_rank_to_json,
)
from traceml.reporting.summaries.issue_summary import (
    issues_by_rank_json,
    issues_to_json,
)
from traceml.reporting.summaries.summary_formatting import (
    bytes_to_gb,
    duration_from_bounds,
    format_optional,
    share_percent,
)


def build_process_card(
    agg: ProcessSummaryAgg,
    *,
    per_rank: Dict[int, PerRankProcessSummary],
) -> tuple[str, Dict[str, Any]]:
    """Build the Process section payload and its compact card text."""
    duration_s = duration_from_bounds(agg.first_ts, agg.last_ts)

    ram_avg_gb = bytes_to_gb(agg.ram_avg_bytes)
    ram_peak_gb = bytes_to_gb(agg.ram_peak_bytes)
    ram_total_gb = bytes_to_gb(agg.ram_total_bytes)

    gpu_mem_used_avg_gb = bytes_to_gb(agg.gpu_mem_used_avg_bytes)
    gpu_mem_used_peak_gb = bytes_to_gb(agg.gpu_mem_used_peak_bytes)
    gpu_mem_reserved_avg_gb = bytes_to_gb(agg.gpu_mem_reserved_avg_bytes)
    gpu_mem_reserved_peak_gb = bytes_to_gb(agg.gpu_mem_reserved_peak_bytes)
    gpu_mem_total_gb = bytes_to_gb(agg.gpu_mem_total_bytes)

    gpu_mem_used_peak_pct = share_percent(
        agg.gpu_mem_used_peak_bytes,
        agg.gpu_mem_total_bytes,
    )
    gpu_mem_reserved_peak_pct = share_percent(
        agg.gpu_mem_reserved_peak_bytes,
        agg.gpu_mem_total_bytes,
    )
    cpu_capacity_percent = _cpu_capacity_percent(agg)
    ram_peak_percent = share_percent(agg.ram_peak_bytes, agg.ram_total_bytes)
    per_rank_for_diagnosis = _per_rank_to_diagnosis_input(per_rank)
    global_rank_rollup = _global_rank_rollup_to_json(per_rank)
    gpu_reserved_overhang_ratio = max(
        (
            item.gpu_mem_reserved_overhang_ratio
            for item in per_rank.values()
            if item.gpu_mem_reserved_overhang_ratio is not None
        ),
        default=None,
    )

    diagnosis_result = build_process_diagnosis_result(
        duration_s=duration_s,
        process_samples=agg.process_samples,
        distinct_ranks=agg.distinct_ranks,
        cpu_avg_percent=agg.cpu_avg_percent,
        cpu_peak_percent=agg.cpu_peak_percent,
        cpu_logical_core_count=agg.cpu_logical_core_count,
        ram_avg_bytes=agg.ram_avg_bytes,
        ram_peak_bytes=agg.ram_peak_bytes,
        ram_total_bytes=agg.ram_total_bytes,
        gpu_available=agg.gpu_available,
        gpu_count=agg.gpu_count,
        gpu_device_index=agg.gpu_device_index,
        gpu_mem_used_avg_bytes=agg.gpu_mem_used_avg_bytes,
        gpu_mem_used_peak_bytes=agg.gpu_mem_used_peak_bytes,
        gpu_mem_reserved_avg_bytes=agg.gpu_mem_reserved_avg_bytes,
        gpu_mem_reserved_peak_bytes=agg.gpu_mem_reserved_peak_bytes,
        gpu_mem_total_bytes=agg.gpu_mem_total_bytes,
        per_rank=per_rank_for_diagnosis,
    )
    primary_diagnosis = diagnosis_result.primary
    issues = diagnosis_result.issues
    issues_by_global_rank, _unassigned_issues = issues_by_rank_json(
        issues,
        rank_keys=per_rank.keys(),
    )
    per_global_rank_json = _per_rank_to_json(per_rank)
    for rank_key, entry in per_global_rank_json.items():
        entry["issues"] = issues_by_global_rank.get(rank_key, [])

    lines = [
        (
            "TraceML Process Summary | duration "
            f"{format_optional(duration_s, 's', 1)} | "
            f"samples {agg.process_samples}"
        ),
        "Process",
        f"- Diagnosis: {primary_diagnosis.status}",
        (
            "- Stats: "
            + _build_stats_line(
                agg,
                ram_peak_gb=ram_peak_gb,
                ram_total_gb=ram_total_gb,
                gpu_mem_used_peak_pct=gpu_mem_used_peak_pct,
                gpu_mem_reserved_peak_pct=gpu_mem_reserved_peak_pct,
            )
        ),
        f"- Why: {primary_diagnosis.reason}",
    ]
    card = "\n".join(lines)

    global_summary = {
        "scope": {
            "global_ranks": agg.distinct_ranks,
            "samples": agg.process_samples,
        },
        "cpu": {
            "avg_percent": agg.cpu_avg_percent,
            "peak_percent": agg.cpu_peak_percent,
            "logical_core_count": agg.cpu_logical_core_count,
            "capacity_percent": cpu_capacity_percent,
            "capacity_band": _band_name(
                DEFAULT_PROCESS_POLICY.cpu_capacity_percent.classify(
                    cpu_capacity_percent
                )
            ),
        },
        "ram": {
            "avg_gb": ram_avg_gb,
            "peak_gb": ram_peak_gb,
            "total_gb": ram_total_gb,
            "peak_percent": ram_peak_percent,
            "peak_band": _band_name(
                DEFAULT_PROCESS_POLICY.rss_peak_percent.classify(
                    ram_peak_percent
                )
            ),
        },
        "gpu": {
            "available": agg.gpu_available,
            "processes_with_gpu": sum(
                1 for item in per_rank.values() if item.gpu_available
            ),
            "visible_count_max_per_process": agg.gpu_count,
            "device_total_gb": gpu_mem_total_gb,
            "used_avg_gb": gpu_mem_used_avg_gb,
            "used_peak_gb": gpu_mem_used_peak_gb,
            "reserved_avg_gb": gpu_mem_reserved_avg_gb,
            "reserved_peak_gb": gpu_mem_reserved_peak_gb,
            "used_peak_pct": gpu_mem_used_peak_pct,
            "reserved_peak_pct": gpu_mem_reserved_peak_pct,
            "reserved_overhang_ratio": gpu_reserved_overhang_ratio,
            "reserved_overhang_band": _band_name(
                DEFAULT_PROCESS_POLICY.gpu_reserved_overhang_ratio.classify(
                    gpu_reserved_overhang_ratio
                )
            ),
        },
    }

    summary = {
        "overview": {
            "duration_s": duration_s,
            "samples": agg.process_samples,
            "global_ranks_seen": agg.distinct_ranks,
        },
        "primary_diagnosis": diagnosis_to_dict(
            primary_diagnosis,
            drop_none=True,
            include_action=False,
        ),
        "issues": issues_to_json(issues),
        "issues_by_global_rank": issues_by_global_rank,
        "global": global_summary,
        "global_rank_rollup": global_rank_rollup,
        "per_global_rank": per_global_rank_json,
        "units": {
            "memory": "GB",
            "cpu": "%",
            "skew": "%",
        },
        "card": card,
    }
    return card, summary


def build_process_section_payload(
    data: ProcessSectionData,
) -> Dict[str, Any]:
    """Build the JSON-safe process-section payload from loaded data."""
    _, payload = build_process_card(
        data.aggregate,
        per_rank=data.per_rank,
    )
    return payload


__all__ = [
    "build_process_card",
    "build_process_section_payload",
]
