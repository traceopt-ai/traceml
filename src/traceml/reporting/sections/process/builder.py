# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Payload builder for the final-report process section."""

from __future__ import annotations

from statistics import median
from typing import Any, Dict, Iterable, Optional

from traceml.diagnostics.common import diagnosis_to_dict
from traceml.diagnostics.process import build_process_diagnosis_result
from traceml.reporting.schema import (
    BaseGlobal,
    BaseGroups,
    BaseSectionPayload,
    GlobalWindow,
    GroupRow,
    RankMetadata,
)
from traceml.reporting.sections.process.loader import ProcessSectionData
from traceml.reporting.sections.process.model import (
    PROCESS_METRIC_NAMES,
    PerRankProcessSummary,
    ProcessSummaryAgg,
    build_process_stats_line,
    cpu_capacity_percent,
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
from traceml.reporting.topology import topology_mode_from_identities


def _rank_point(
    value: Optional[float],
    idx: Optional[str],
) -> Dict[str, Any]:
    """Return a public metric comparison point."""
    return {"value": value, "idx": idx}


def _rank_metric_values(item: PerRankProcessSummary) -> Dict[str, Any]:
    """Return public row metrics for one global rank."""
    return {
        "cpu_percent": item.cpu_avg_percent,
        "cpu_capacity_percent": cpu_capacity_percent(
            item.cpu_avg_percent,
            item.cpu_logical_core_count,
        ),
        "ram_bytes": item.ram_avg_bytes,
        "ram_percent": share_percent(
            item.ram_avg_bytes,
            item.ram_total_bytes,
        ),
        "gpu_mem_used_bytes": item.gpu_mem_used_avg_bytes,
        "gpu_mem_reserved_bytes": item.gpu_mem_reserved_avg_bytes,
        "gpu_mem_reserved_percent": share_percent(
            item.gpu_mem_reserved_avg_bytes,
            item.gpu_mem_total_bytes,
        ),
        "gpu_mem_headroom_bytes": (
            max(
                float(item.gpu_mem_total_bytes)
                - float(item.gpu_mem_reserved_avg_bytes),
                0.0,
            )
            if item.gpu_mem_total_bytes is not None
            and item.gpu_mem_reserved_avg_bytes is not None
            else None
        ),
    }


def _per_global_rank_to_json(
    per_global_rank: Dict[int, PerRankProcessSummary],
) -> Dict[str, Dict[str, Any]]:
    """Convert per-rank process aggregates into grouped JSON rows."""
    out: Dict[str, Dict[str, Any]] = {}
    for rank_id, item in sorted(per_global_rank.items()):
        identity = {
            "global_rank": item.global_rank,
            "local_rank": item.local_rank,
            "node_rank": item.node_rank,
            "hostname": item.hostname,
            "local_world_size": item.local_world_size,
            "world_size": item.world_size,
        }
        out[str(rank_id)] = GroupRow(
            identity=identity,
            metrics=_rank_metric_values(item),
        ).to_json()
    return out


PROCESS_LOWER_IS_WORSE_METRICS = {
    "gpu_mem_headroom_bytes",
}


def _metric_values_by_row(
    row_metrics: Dict[str, Dict[str, Any]],
    metric_names: Iterable[str],
) -> Dict[str, list[tuple[str, float]]]:
    """Collect finite row metric values by metric name."""
    values: Dict[str, list[tuple[str, float]]] = {
        metric: [] for metric in metric_names
    }
    for row_id, metrics in row_metrics.items():
        for metric_name in metric_names:
            raw = metrics.get(metric_name)
            if raw is None:
                continue
            try:
                values[metric_name].append((str(row_id), float(raw)))
            except Exception:
                continue
    return values


def _average_metrics_from_rows(
    row_metrics: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Average each public Process metric across global-rank rows."""
    values = _metric_values_by_row(row_metrics, PROCESS_METRIC_NAMES)
    return {
        metric: (
            sum(value for _row_id, value in pairs) / len(pairs)
            if pairs
            else None
        )
        for metric, pairs in values.items()
    }


def _closest_row_to_median(
    pairs: list[tuple[str, float]]
) -> tuple[str, float]:
    """Return the row whose value best represents the median."""
    median_value = float(median(value for _row_id, value in pairs))
    return min(
        pairs,
        key=lambda item: (abs(item[1] - median_value), item[1], item[0]),
    )


def _row_metric_points(
    row_metrics: Dict[str, Dict[str, Any]],
    *,
    kind: str,
) -> Dict[str, Any]:
    """Build median or worst `{value, idx}` points from rank row metrics."""
    values_by_metric = _metric_values_by_row(row_metrics, PROCESS_METRIC_NAMES)
    points: Dict[str, Any] = {}
    for metric_name, pairs in values_by_metric.items():
        if not pairs:
            points[metric_name] = _rank_point(None, None)
            continue
        if kind == "median":
            row_id, value = _closest_row_to_median(pairs)
        elif kind == "worst":
            lower_is_worse = metric_name in PROCESS_LOWER_IS_WORSE_METRICS
            row_id, value = (
                min(pairs, key=lambda item: (item[1], item[0]))
                if lower_is_worse
                else max(pairs, key=lambda item: (item[1], item[0]))
            )
        else:
            raise ValueError(f"Unsupported Process point kind: {kind}")
        points[metric_name] = _rank_point(value, row_id)
    return points


def _per_global_rank_to_diagnosis_input(
    per_global_rank: Dict[int, PerRankProcessSummary],
) -> Dict[int, Dict[str, Optional[float]]]:
    """Return richer process inputs used only by diagnosis rules."""
    out: Dict[int, Dict[str, Optional[float]]] = {}
    for rank_id, item in sorted(per_global_rank.items()):
        out[int(rank_id)] = {
            "gpu_device_index": (
                float(item.gpu_device_index)
                if item.gpu_device_index is not None
                else None
            ),
            "cpu_avg_percent": item.cpu_avg_percent,
            "cpu_peak_percent": item.cpu_peak_percent,
            "ram_avg_bytes": item.ram_avg_bytes,
            "ram_peak_bytes": item.ram_peak_bytes,
            "ram_total_bytes": item.ram_total_bytes,
            "gpu_mem_used_avg_bytes": item.gpu_mem_used_avg_bytes,
            "gpu_mem_used_peak_bytes": item.gpu_mem_used_peak_bytes,
            "gpu_mem_reserved_avg_bytes": item.gpu_mem_reserved_avg_bytes,
            "gpu_mem_reserved_peak_bytes": item.gpu_mem_reserved_peak_bytes,
            "gpu_mem_total_bytes": item.gpu_mem_total_bytes,
            "gpu_mem_reserved_overhang_ratio": (
                item.gpu_mem_reserved_overhang_ratio
            ),
        }
    return out


def build_process_payload(
    agg: ProcessSummaryAgg,
    *,
    per_global_rank: Dict[int, PerRankProcessSummary],
) -> Dict[str, Any]:
    """Build the Process section payload and its compact card text."""
    duration_s = duration_from_bounds(agg.first_ts, agg.last_ts)

    ram_peak_gb = bytes_to_gb(agg.ram_peak_bytes)
    ram_total_gb = bytes_to_gb(agg.ram_total_bytes)

    gpu_mem_used_peak_pct = share_percent(
        agg.gpu_mem_used_peak_bytes,
        agg.gpu_mem_total_bytes,
    )
    gpu_mem_reserved_peak_pct = share_percent(
        agg.gpu_mem_reserved_peak_bytes,
        agg.gpu_mem_total_bytes,
    )
    per_global_rank_for_diagnosis = _per_global_rank_to_diagnosis_input(
        per_global_rank
    )

    diagnosis_result = build_process_diagnosis_result(
        duration_s=duration_s,
        process_samples=agg.process_samples,
        distinct_ranks=agg.distinct_global_ranks,
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
        per_rank=per_global_rank_for_diagnosis,
    )
    primary_diagnosis = diagnosis_result.primary
    issues = diagnosis_result.issues
    issues_by_global_rank, _unassigned_issues = issues_by_rank_json(
        issues,
        rank_keys=per_global_rank.keys(),
    )
    per_global_rank_json = _per_global_rank_to_json(per_global_rank)
    for rank_key, entry in per_global_rank_json.items():
        entry["issues"] = issues_by_global_rank.get(rank_key, [])
    row_metrics = {
        rank_key: dict(entry.get("metrics", {}))
        for rank_key, entry in per_global_rank_json.items()
    }

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
            + build_process_stats_line(
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

    mode = topology_mode_from_identities(
        per_global_rank.values(),
        has_data=bool(per_global_rank),
    )

    global_summary = BaseGlobal(
        index_by="global_rank",
        window=GlobalWindow(
            kind="sample_window",
            alignment="none",
            samples=agg.process_samples,
        ).to_json(),
        average=_average_metrics_from_rows(row_metrics),
        median=_row_metric_points(row_metrics, kind="median"),
        worst=_row_metric_points(row_metrics, kind="worst"),
    )

    metadata = RankMetadata(
        mode=mode,
        global_ranks_seen=agg.distinct_global_ranks,
        global_ranks_used=agg.distinct_global_ranks,
        duration_s=duration_s,
        samples=agg.process_samples,
        section_metric_names=PROCESS_METRIC_NAMES,
    )
    summary = BaseSectionPayload(
        metadata=metadata.to_json(),
        diagnosis=diagnosis_to_dict(
            primary_diagnosis,
            drop_none=True,
            include_action=False,
        ),
        issues=issues_to_json(issues),
        global_summary=global_summary.to_json(),
        groups=BaseGroups(
            by="global_rank",
            rows=per_global_rank_json,
        ).to_json(),
        units={
            "memory": "bytes",
            "cpu": "%",
        },
        card=card,
    ).to_json()
    return summary


def build_process_section_payload(
    data: ProcessSectionData,
) -> Dict[str, Any]:
    """Build the JSON-safe process-section payload from loaded data."""
    return build_process_payload(
        data.aggregate,
        per_global_rank=data.per_global_rank,
    )


__all__ = [
    "build_process_payload",
    "build_process_section_payload",
]
