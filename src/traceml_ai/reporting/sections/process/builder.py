# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Payload builder for the final-report process section."""

from __future__ import annotations

from statistics import median
from typing import Any, Dict, Iterable, Optional

from traceml_ai.diagnostics.common import DiagnosticResult
from traceml_ai.diagnostics.process import ProcessDiagnosis
from traceml_ai.reporting.schema import (
    BaseGlobal,
    BaseGroups,
    BaseSectionPayload,
    GlobalWindow,
    GroupRow,
    RankMetadata,
)
from traceml_ai.reporting.sections.process.loader import ProcessSectionData
from traceml_ai.reporting.sections.process.model import (
    PROCESS_METRIC_NAMES,
    PerRankProcessSummary,
    build_process_stats_line,
    cpu_capacity_percent,
)
from traceml_ai.reporting.summaries.issue_summary import (
    diagnostic_result_to_json,
)
from traceml_ai.reporting.summaries.summary_formatting import (
    bytes_to_gb,
    duration_from_bounds,
    format_optional,
    share_percent,
)
from traceml_ai.reporting.topology import topology_mode_from_identities


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


def build_process_payload(
    data: ProcessSectionData,
    diagnosis_result: DiagnosticResult[ProcessDiagnosis],
) -> Dict[str, Any]:
    """
    Build the Process section payload and compact card text.

    The section pipeline owns loading and diagnosis. This function only turns
    the loaded process summary plus diagnosis result into JSON and card text.
    """
    agg = data.aggregate
    per_global_rank = data.per_global_rank
    duration_s = duration_from_bounds(agg.first_ts, agg.last_ts)

    # Prepare compact display values used by the human-readable card.
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
    primary_diagnosis = diagnosis_result.primary
    diagnosis_json, issues_json = diagnostic_result_to_json(diagnosis_result)

    per_global_rank_json = _per_global_rank_to_json(per_global_rank)
    row_metrics = {
        rank_key: dict(entry.get("metrics", {}))
        for rank_key, entry in per_global_rank_json.items()
    }

    # Keep the terminal card short; detailed values stay in structured JSON.
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

    # Build cross-rank rollups from the grouped row metrics.
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
    # Assemble the final common section envelope.
    summary = BaseSectionPayload(
        metadata=metadata.to_json(),
        diagnosis=diagnosis_json,
        issues=issues_json,
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


__all__ = [
    "build_process_payload",
]
