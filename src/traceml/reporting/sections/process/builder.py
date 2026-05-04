"""Payload builder for the final-report process section."""

from __future__ import annotations

from typing import Any, Dict

from traceml.diagnostics.common import diagnosis_to_dict
from traceml.diagnostics.process import build_process_diagnosis_result
from traceml.diagnostics.process.policy import DEFAULT_PROCESS_POLICY
from traceml.reporting.sections.process.loader import ProcessSectionData
from traceml.reporting.summaries.issue_summary import (
    issues_by_metric_json,
    issues_by_rank_json,
    issues_to_json,
)
from traceml.reporting.summaries.process import (
    PerRankProcessSummary,
    ProcessSummaryAgg,
    _band_name,
    _best_rank_idx,
    _build_stats_line,
    _cpu_capacity_percent,
    _per_rank_to_diagnosis_input,
    _per_rank_to_json,
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
    gpu_reserved_overhang_ratio = max(
        (
            item.gpu_mem_reserved_overhang_ratio
            for item in per_rank.values()
            if item.gpu_mem_reserved_overhang_ratio is not None
        ),
        default=None,
    )
    highest_overhang_rank = _best_rank_idx(
        per_rank,
        "gpu_mem_reserved_overhang_ratio",
    )

    highest_used_rank = _best_rank_idx(per_rank, "gpu_mem_used_peak_bytes")
    highest_reserved_rank = _best_rank_idx(
        per_rank, "gpu_mem_reserved_peak_bytes"
    )

    highest_used_peak_gb = (
        bytes_to_gb(per_rank[highest_used_rank].gpu_mem_used_peak_bytes)
        if highest_used_rank is not None
        else None
    )
    highest_reserved_peak_gb = (
        bytes_to_gb(
            per_rank[highest_reserved_rank].gpu_mem_reserved_peak_bytes
        )
        if highest_reserved_rank is not None
        else None
    )

    least_headroom_rank = None
    least_headroom_gb = None
    for rank_id, item in per_rank.items():
        total_gb = bytes_to_gb(item.gpu_mem_total_bytes)
        reserved_gb = bytes_to_gb(item.gpu_mem_reserved_peak_bytes)
        if total_gb is None or reserved_gb is None:
            continue
        headroom_gb = max(total_gb - reserved_gb, 0.0)
        if least_headroom_gb is None or headroom_gb < least_headroom_gb:
            least_headroom_rank = int(rank_id)
            least_headroom_gb = headroom_gb

    diagnosis_result = build_process_diagnosis_result(
        duration_s=duration_s,
        process_samples=agg.process_samples,
        distinct_ranks=agg.distinct_ranks,
        distinct_pids=agg.distinct_pids,
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
    issues_by_rank, unassigned_issues = issues_by_rank_json(
        issues,
        rank_keys=per_rank.keys(),
    )
    issues_by_metric, metric_unassigned = issues_by_metric_json(issues)
    per_rank_json = _per_rank_to_json(per_rank)
    for rank_key, entry in per_rank_json.items():
        entry["issues"] = issues_by_rank.get(rank_key, [])

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
            "ranks": agg.distinct_ranks,
            "pids": agg.distinct_pids,
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
        "gpu_rollup": {
            "available": agg.gpu_available,
            "count": agg.gpu_count,
            "device_index": agg.gpu_device_index,
            "used_avg_gb": gpu_mem_used_avg_gb,
            "used_peak_gb": gpu_mem_used_peak_gb,
            "reserved_avg_gb": gpu_mem_reserved_avg_gb,
            "reserved_peak_gb": gpu_mem_reserved_peak_gb,
            "total_gb": gpu_mem_total_gb,
            "used_peak_pct": gpu_mem_used_peak_pct,
            "used_peak_band": _band_name(
                DEFAULT_PROCESS_POLICY.gpu_memory_peak_percent.classify(
                    gpu_mem_used_peak_pct
                )
            ),
            "reserved_peak_pct": gpu_mem_reserved_peak_pct,
            "reserved_peak_band": _band_name(
                DEFAULT_PROCESS_POLICY.gpu_memory_peak_percent.classify(
                    gpu_mem_reserved_peak_pct
                )
            ),
            "reserved_overhang_ratio": gpu_reserved_overhang_ratio,
            "reserved_overhang_band": _band_name(
                DEFAULT_PROCESS_POLICY.gpu_reserved_overhang_ratio.classify(
                    gpu_reserved_overhang_ratio
                )
            ),
            "highest_overhang_rank": highest_overhang_rank,
            "highest_used_rank": highest_used_rank,
            "highest_used_peak_gb": highest_used_peak_gb,
            "highest_reserved_rank": highest_reserved_rank,
            "highest_reserved_peak_gb": highest_reserved_peak_gb,
            "least_headroom_rank": least_headroom_rank,
            "least_headroom_gb": least_headroom_gb,
        },
    }

    summary = {
        "overview": {
            "duration_s": duration_s,
            "samples": agg.process_samples,
            "ranks_seen": agg.distinct_ranks,
            "pids_seen": agg.distinct_pids,
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
        "per_rank": per_rank_json,
        "units": {
            "memory": "GB",
            "cpu": "%",
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
