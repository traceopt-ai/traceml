"""Payload builder for the final-report system section."""

from __future__ import annotations

from typing import Any, Dict

from traceml.diagnostics.common import diagnosis_to_dict
from traceml.diagnostics.system import build_system_diagnosis_result
from traceml.diagnostics.system.policy import DEFAULT_SYSTEM_POLICY
from traceml.reporting.sections.system.loader import SystemSectionData
from traceml.reporting.summaries.issue_summary import (
    issues_by_metric_json,
    issues_by_rank_json,
    issues_to_json,
)
from traceml.reporting.summaries.summary_formatting import (
    bytes_to_gb,
    duration_from_bounds,
    format_optional,
)
from traceml.reporting.summaries.system import (
    PerGPUSummary,
    SystemSummaryAgg,
    _band_name,
    _best_gpu_idx,
    _build_stats_line,
    _highest_gpu_memory_percent,
    _per_gpu_to_diagnosis_input,
    _per_gpu_to_json,
    _percent,
)


def build_system_card(
    agg: SystemSummaryAgg,
    *,
    per_gpu: Dict[int, PerGPUSummary],
) -> tuple[str, Dict[str, Any]]:
    """Build the System section payload and its compact card text."""
    duration_s = duration_from_bounds(agg.first_ts, agg.last_ts)

    ram_avg_gb = bytes_to_gb(agg.ram_avg_bytes)
    ram_peak_gb = bytes_to_gb(agg.ram_peak_bytes)
    ram_total_gb = bytes_to_gb(agg.ram_total_bytes)
    ram_peak_percent = _percent(agg.ram_peak_bytes, agg.ram_total_bytes)

    gpu_mem_avg_gb = bytes_to_gb(agg.gpu_mem_avg_bytes)
    gpu_mem_peak_gb = bytes_to_gb(agg.gpu_mem_peak_bytes)
    gpu_mem_peak_percent = _highest_gpu_memory_percent(per_gpu)
    gpu_power_avg_limit_percent = max(
        (
            value
            for value in (
                _percent(item.power_avg_w, item.power_limit_w)
                for item in per_gpu.values()
            )
            if value is not None
        ),
        default=None,
    )
    per_gpu_for_diagnosis = _per_gpu_to_diagnosis_input(per_gpu)

    hottest_gpu_idx = _best_gpu_idx(per_gpu, "temp_peak_c")
    highest_mem_gpu_idx = _best_gpu_idx(per_gpu, "mem_peak_bytes")
    highest_util_gpu_idx = _best_gpu_idx(per_gpu, "util_peak_percent")

    hottest_gpu_temp_peak_c = (
        per_gpu[hottest_gpu_idx].temp_peak_c
        if hottest_gpu_idx is not None
        else None
    )
    highest_mem_peak_gb = (
        bytes_to_gb(per_gpu[highest_mem_gpu_idx].mem_peak_bytes)
        if highest_mem_gpu_idx is not None
        else None
    )
    highest_util_peak_percent = (
        per_gpu[highest_util_gpu_idx].util_peak_percent
        if highest_util_gpu_idx is not None
        else None
    )
    diagnosis_result = build_system_diagnosis_result(
        duration_s=duration_s,
        system_samples=agg.system_samples,
        cpu_avg_percent=agg.cpu_avg_percent,
        cpu_peak_percent=agg.cpu_peak_percent,
        ram_avg_bytes=agg.ram_avg_bytes,
        ram_peak_bytes=agg.ram_peak_bytes,
        ram_total_bytes=agg.ram_total_bytes,
        gpu_available=agg.gpu_available,
        gpu_count=agg.gpu_count,
        gpu_util_avg_percent=agg.gpu_util_avg_percent,
        gpu_util_peak_percent=agg.gpu_util_peak_percent,
        gpu_mem_avg_bytes=agg.gpu_mem_avg_bytes,
        gpu_mem_peak_bytes=agg.gpu_mem_peak_bytes,
        gpu_temp_avg_c=agg.gpu_temp_avg_c,
        gpu_temp_peak_c=agg.gpu_temp_peak_c,
        gpu_power_avg_w=agg.gpu_power_avg_w,
        gpu_power_peak_w=agg.gpu_power_peak_w,
        per_gpu=per_gpu_for_diagnosis,
    )
    primary_diagnosis = diagnosis_result.primary
    issues = diagnosis_result.issues
    issues_by_rank, unassigned_issues = issues_by_rank_json(
        issues,
        rank_keys=per_gpu.keys(),
    )
    issues_by_metric, metric_unassigned = issues_by_metric_json(issues)
    per_gpu_json = _per_gpu_to_json(per_gpu)
    for gpu_idx, entry in per_gpu_json.items():
        entry["issues"] = issues_by_rank.get(gpu_idx, [])

    lines = [
        (
            "TraceML System Summary | duration "
            f"{format_optional(duration_s, 's', 1)} | "
            f"samples {agg.system_samples}"
        ),
        "System",
        f"- Diagnosis: {primary_diagnosis.status}",
        f"- Stats: {_build_stats_line(agg, per_gpu=per_gpu)}",
        f"- Why: {primary_diagnosis.reason}",
    ]
    card = "\n".join(lines)

    global_summary = {
        "cpu": {
            "avg_percent": agg.cpu_avg_percent,
            "peak_percent": agg.cpu_peak_percent,
            "avg_band": _band_name(
                DEFAULT_SYSTEM_POLICY.cpu_avg_percent.classify(
                    agg.cpu_avg_percent
                )
            ),
        },
        "ram": {
            "avg_gb": ram_avg_gb,
            "peak_gb": ram_peak_gb,
            "total_gb": ram_total_gb,
            "peak_percent": ram_peak_percent,
            "peak_band": _band_name(
                DEFAULT_SYSTEM_POLICY.ram_peak_percent.classify(
                    ram_peak_percent
                )
            ),
        },
        "gpu_rollup": {
            "available": agg.gpu_available,
            "count": agg.gpu_count,
            "util_avg_percent": agg.gpu_util_avg_percent,
            "util_peak_percent": agg.gpu_util_peak_percent,
            "util_avg_band": _band_name(
                DEFAULT_SYSTEM_POLICY.gpu_util_avg_percent.classify(
                    agg.gpu_util_avg_percent
                )
            ),
            "mem_avg_gb": gpu_mem_avg_gb,
            "mem_peak_gb": gpu_mem_peak_gb,
            "mem_peak_percent": gpu_mem_peak_percent,
            "mem_peak_band": _band_name(
                DEFAULT_SYSTEM_POLICY.gpu_memory_peak_percent.classify(
                    gpu_mem_peak_percent
                )
            ),
            "temp_avg_c": agg.gpu_temp_avg_c,
            "temp_peak_c": agg.gpu_temp_peak_c,
            "temp_peak_band": _band_name(
                DEFAULT_SYSTEM_POLICY.gpu_temp_peak_c.classify(
                    agg.gpu_temp_peak_c
                )
            ),
            "power_avg_w": agg.gpu_power_avg_w,
            "power_peak_w": agg.gpu_power_peak_w,
            "power_avg_limit_percent": gpu_power_avg_limit_percent,
            "power_avg_band": _band_name(
                DEFAULT_SYSTEM_POLICY.gpu_power_avg_limit_percent.classify(
                    gpu_power_avg_limit_percent
                )
            ),
            "hottest_gpu_idx": hottest_gpu_idx,
            "hottest_gpu_temp_peak_c": hottest_gpu_temp_peak_c,
            "highest_mem_gpu_idx": highest_mem_gpu_idx,
            "highest_mem_peak_gb": highest_mem_peak_gb,
            "highest_util_gpu_idx": highest_util_gpu_idx,
            "highest_util_peak_percent": highest_util_peak_percent,
        },
    }

    summary = {
        "overview": {
            "duration_s": duration_s,
            "samples": agg.system_samples,
            "gpu_available": agg.gpu_available,
            "gpu_count": agg.gpu_count,
        },
        "primary_diagnosis": diagnosis_to_dict(
            primary_diagnosis,
            drop_none=True,
            include_action=False,
        ),
        "issues": issues_to_json(issues),
        "issues_by_rank": issues_by_rank,
        "issues_by_metric": issues_by_metric,
        "unassigned_issues": unassigned_issues + metric_unassigned,
        "global": global_summary,
        "per_gpu": per_gpu_json,
        "units": {
            "memory": "GB",
            "temperature": "C",
            "power": "W",
            "util": "%",
        },
        "card": card,
    }

    return card, summary


def build_system_section_payload(
    data: SystemSectionData,
) -> Dict[str, Any]:
    """Build the JSON-safe system-section payload from loaded data."""
    _, payload = build_system_card(
        data.aggregate,
        per_gpu=data.per_gpu,
    )
    return payload


__all__ = [
    "build_system_card",
    "build_system_section_payload",
]
