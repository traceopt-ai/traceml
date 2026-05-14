# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Payload builder for the final-report process section."""

from __future__ import annotations

from statistics import median
from typing import Any, Callable, Dict, Optional

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
    process_cpu_capacity_percent,
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


def _topology_mode(
    per_global_rank: Dict[int, PerRankProcessSummary],
) -> str:
    """Return run topology from observed process runtime identity."""
    if not per_global_rank:
        return "no_data"

    node_ranks = {
        item.node_rank
        for item in per_global_rank.values()
        if item.node_rank is not None
    }
    if len(node_ranks) > 1:
        return "multi_node"

    for item in per_global_rank.values():
        if (
            item.world_size is not None
            and item.local_world_size is not None
            and item.world_size > item.local_world_size
        ):
            return "multi_node"

    return "single_node"


def _rank_point(
    rollup: Dict[str, Any],
    *,
    kind: str,
) -> Dict[str, Any]:
    """Return a `{value, global_rank}` point from a rank rollup."""
    global_rank = rollup.get(f"{kind}_global_rank")
    return {
        "value": rollup.get(kind),
        "idx": str(global_rank) if global_rank is not None else None,
    }


def _rank_comparison_points(
    global_rank_rollup: Dict[str, Dict[str, Any]],
    *,
    kind: str,
) -> Dict[str, Any]:
    return {
        metric: _rank_point(global_rank_rollup.get(metric, {}), kind=kind)
        for metric in PROCESS_METRIC_NAMES
    }


def _rank_cpu_capacity_percent(
    item: PerRankProcessSummary,
) -> Optional[float]:
    """CPU capacity used by one traced process, normalized by host cores."""
    if (
        item.cpu_avg_percent is None
        or item.cpu_logical_core_count is None
        or item.cpu_logical_core_count <= 0
    ):
        return None
    return max(
        0.0,
        float(item.cpu_avg_percent)
        / (100.0 * float(item.cpu_logical_core_count))
        * 100.0,
    )


def _rank_metric_values(item: PerRankProcessSummary) -> Dict[str, Any]:
    """Return public row metrics for one global rank."""
    return {
        "cpu_percent": item.cpu_avg_percent,
        "cpu_capacity_percent": _rank_cpu_capacity_percent(item),
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


def _global_average_metrics(
    agg: ProcessSummaryAgg,
    *,
    cpu_capacity_percent: Optional[float],
) -> Dict[str, Any]:
    """Return average-valued Process metrics for the global JSON block."""
    return {
        "cpu_percent": agg.cpu_avg_percent,
        "cpu_capacity_percent": cpu_capacity_percent,
        "ram_bytes": agg.ram_avg_bytes,
        "ram_percent": share_percent(
            agg.ram_avg_bytes,
            agg.ram_total_bytes,
        ),
        "gpu_mem_used_bytes": agg.gpu_mem_used_avg_bytes,
        "gpu_mem_reserved_bytes": agg.gpu_mem_reserved_avg_bytes,
        "gpu_mem_reserved_percent": share_percent(
            agg.gpu_mem_reserved_avg_bytes,
            agg.gpu_mem_total_bytes,
        ),
        "gpu_mem_headroom_bytes": (
            max(
                float(agg.gpu_mem_total_bytes)
                - float(agg.gpu_mem_reserved_avg_bytes),
                0.0,
            )
            if agg.gpu_mem_total_bytes is not None
            and agg.gpu_mem_reserved_avg_bytes is not None
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


def _metric_rollup(
    per_global_rank: Dict[int, PerRankProcessSummary],
    value: Callable[[PerRankProcessSummary], Optional[float]],
    *,
    higher_is_worse: bool = True,
) -> Optional[Dict[str, Optional[float]]]:
    """Return median/worst/skew for one metric across global ranks."""
    pairs: list[tuple[float, int]] = []
    for global_rank, item in per_global_rank.items():
        raw = value(item)
        if raw is not None:
            pairs.append((float(raw), int(global_rank)))

    if not pairs:
        return None

    values = [metric for metric, _global_rank in pairs]
    worst_value, worst_global_rank = (
        max(pairs, key=lambda pair: pair[0])
        if higher_is_worse
        else min(pairs, key=lambda pair: pair[0])
    )
    median_value = float(median(values))
    if median_value == 0.0:
        skew_percent = None
    elif higher_is_worse:
        skew_percent = (
            (float(worst_value) - median_value) / abs(median_value) * 100.0
        )
    else:
        skew_percent = (
            (median_value - float(worst_value)) / abs(median_value) * 100.0
        )

    median_rank = min(
        pairs,
        key=lambda pair: (abs(pair[0] - median_value), pair[0], pair[1]),
    )[1]
    return {
        "median": median_value,
        "worst": float(worst_value),
        "median_global_rank": int(median_rank),
        "worst_global_rank": int(worst_global_rank),
        "skew_percent": skew_percent,
    }


def _global_rank_rollup_to_json(
    per_global_rank: Dict[int, PerRankProcessSummary],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Build Process median/worst points across global ranks."""
    metrics = {
        "cpu_percent": _metric_rollup(
            per_global_rank,
            lambda item: item.cpu_avg_percent,
        ),
        "cpu_capacity_percent": _metric_rollup(
            per_global_rank,
            _rank_cpu_capacity_percent,
        ),
        "ram_bytes": _metric_rollup(
            per_global_rank,
            lambda item: item.ram_peak_bytes,
        ),
        "ram_percent": _metric_rollup(
            per_global_rank,
            lambda item: share_percent(
                item.ram_peak_bytes,
                item.ram_total_bytes,
            ),
        ),
        "gpu_mem_used_bytes": _metric_rollup(
            per_global_rank,
            lambda item: item.gpu_mem_used_peak_bytes,
        ),
        "gpu_mem_reserved_bytes": _metric_rollup(
            per_global_rank,
            lambda item: item.gpu_mem_reserved_peak_bytes,
        ),
        "gpu_mem_reserved_percent": _metric_rollup(
            per_global_rank,
            lambda item: share_percent(
                item.gpu_mem_reserved_peak_bytes,
                item.gpu_mem_total_bytes,
            ),
        ),
        "gpu_mem_headroom_bytes": _metric_rollup(
            per_global_rank,
            lambda item: (
                None
                if item.gpu_mem_total_bytes is None
                or item.gpu_mem_reserved_peak_bytes is None
                else max(
                    float(item.gpu_mem_total_bytes)
                    - float(item.gpu_mem_reserved_peak_bytes),
                    0.0,
                )
            ),
            higher_is_worse=False,
        ),
    }
    return {
        name: rollup for name, rollup in metrics.items() if rollup is not None
    }


def build_process_card(
    agg: ProcessSummaryAgg,
    *,
    per_global_rank: Dict[int, PerRankProcessSummary],
) -> tuple[str, Dict[str, Any]]:
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
    cpu_capacity_percent = process_cpu_capacity_percent(agg)
    per_global_rank_for_diagnosis = _per_global_rank_to_diagnosis_input(
        per_global_rank
    )
    global_rank_rollup = _global_rank_rollup_to_json(per_global_rank)

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

    mode = _topology_mode(per_global_rank)

    average = _global_average_metrics(
        agg,
        cpu_capacity_percent=cpu_capacity_percent,
    )
    global_summary = BaseGlobal(
        index_by="global_rank",
        window=GlobalWindow(
            kind="sample_window",
            alignment="none",
            samples=agg.process_samples,
        ).to_json(),
        average=average,
        median=_rank_comparison_points(global_rank_rollup, kind="median"),
        worst=_rank_comparison_points(global_rank_rollup, kind="worst"),
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
    return card, summary


def build_process_section_payload(
    data: ProcessSectionData,
) -> Dict[str, Any]:
    """Build the JSON-safe process-section payload from loaded data."""
    _, payload = build_process_card(
        data.aggregate,
        per_global_rank=data.per_global_rank,
    )
    return payload


__all__ = [
    "build_process_card",
    "build_process_section_payload",
]
