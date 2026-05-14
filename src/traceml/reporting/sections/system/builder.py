# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Payload builder for the final-report System section."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, Optional

from traceml.diagnostics.common import (
    DiagnosticIssue,
    diagnosis_to_dict,
    severity_rank,
)
from traceml.diagnostics.system import build_system_diagnosis_result
from traceml.reporting.schema import (
    BaseGlobal,
    BaseGroups,
    BaseMetadata,
    BaseSectionPayload,
    GlobalWindow,
    GroupRow,
)
from traceml.reporting.sections.system.loader import SystemSectionData
from traceml.reporting.sections.system.model import (
    SYSTEM_METRIC_NAMES,
    PerGPUSummary,
    SystemClusterSummary,
    SystemNodeSummary,
    SystemSummaryAgg,
    average_optional,
    node_gpu_headroom_min_gb,
    node_gpu_mem_peak_percent,
    per_gpu_to_diagnosis_input,
    percent,
)
from traceml.reporting.summaries.issue_summary import issue_to_json
from traceml.reporting.summaries.summary_formatting import (
    duration_from_bounds,
    format_optional,
)

SYSTEM_ISSUE_PRIORITY = {
    "VERY_HIGH_GPU_MEMORY": 0,
    "HIGH_GPU_TEMPERATURE": 1,
    "HIGH_GPU_MEMORY": 2,
    "HIGH_GPU_POWER": 3,
    "HIGH_HOST_MEMORY": 4,
    "HIGH_CPU": 5,
    "LOW_GPU_UTILIZATION": 6,
}


def _scope_for_issue(
    issue: DiagnosticIssue,
    node: SystemNodeSummary,
) -> Dict[str, Any]:
    """Return node/gpu scope for one System issue."""
    base: Dict[str, Any] = {
        "level": "node",
        "node": node.identity.label,
        "node_rank": node.identity.node_rank,
    }
    if issue.ranks:
        base["level"] = "gpu"
        base["gpu_idx"] = int(issue.ranks[0])
    return base


def _scope_for_node(node: SystemNodeSummary) -> Dict[str, Any]:
    return {
        "level": "node",
        "node": node.identity.label,
        "node_rank": node.identity.node_rank,
    }


def _scope_text(text: str, scope: Dict[str, Any]) -> str:
    """Add node context to issue text without making normal text noisy."""
    if scope.get("level") == "gpu":
        gpu_idx = scope.get("gpu_idx")
        node = scope.get("node")
        return text.replace(f" on gpu{gpu_idx}", f" on {node} gpu{gpu_idx}")
    if scope.get("level") == "node" and scope.get("node"):
        return f"{text.rstrip('.')} on {scope['node']}."
    return text


def _diagnose_node(node: SystemNodeSummary):
    agg = node.aggregate
    return _diagnose_aggregate(agg, per_gpu=node.per_gpu)


def _diagnose_aggregate(agg: SystemSummaryAgg, *, per_gpu: Dict[int, Any]):
    return build_system_diagnosis_result(
        duration_s=duration_from_bounds(agg.first_ts, agg.last_ts),
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
        per_gpu=per_gpu_to_diagnosis_input(per_gpu),
    )


def _scoped_issue_json(
    issue: DiagnosticIssue,
    node: SystemNodeSummary,
) -> Dict[str, Any]:
    scope = _scope_for_issue(issue, node)
    scoped = replace(issue, summary=_scope_text(issue.summary, scope))
    payload = issue_to_json(scoped)
    payload.pop("ranks", None)
    payload["scope"] = scope
    return payload


def _issue_sort_key(item: tuple[DiagnosticIssue, SystemNodeSummary]) -> tuple:
    issue, node = item
    return (
        SYSTEM_ISSUE_PRIORITY.get(issue.kind, 999),
        -severity_rank(issue.severity),
        -float(issue.score or 0.0),
        node.identity.label,
    )


def _primary_from_issue(
    issue: DiagnosticIssue,
    node: SystemNodeSummary,
) -> Dict[str, Any]:
    scope = _scope_for_issue(issue, node)
    return {
        "severity": issue.severity,
        "status": issue.status,
        "reason": _scope_text(issue.summary, scope),
        "kind": issue.kind,
        "samples_used": node.aggregate.system_samples,
        "scope": scope,
    }


def _primary_from_diagnosis(
    diagnosis: Any,
    *,
    scope: Dict[str, Any],
) -> Dict[str, Any]:
    payload = (
        diagnosis_to_dict(
            diagnosis,
            drop_none=True,
            include_action=False,
        )
        or {}
    )
    payload["scope"] = scope
    return payload


def _node_headroom_min_bytes(node: SystemNodeSummary) -> Optional[float]:
    value_gb = node_gpu_headroom_min_gb(node)
    return None if value_gb is None else float(value_gb) * 1_000_000_000.0


def _system_mode(cluster: SystemClusterSummary) -> str:
    """Return the topology mode represented by the System summary."""
    if cluster.observed_nodes <= 0:
        return "no_data"
    if cluster.expected_nodes <= 1 and cluster.observed_nodes <= 1:
        return "single_node"
    return "multi_node"


def _gpu_mem_percent_from_summaries(
    gpus: Iterable[PerGPUSummary],
) -> Optional[float]:
    """
    Return average GPU memory pressure across physical GPU summaries.

    GPU indices repeat on every node, so cluster-level calculations must treat
    each node/device pair as a separate summary row.
    """
    used_bytes = 0.0
    total_bytes = 0.0
    for gpu in gpus:
        if gpu.mem_avg_bytes is None or gpu.mem_total_bytes is None:
            continue
        used_bytes += float(gpu.mem_avg_bytes)
        total_bytes += float(gpu.mem_total_bytes)
    return percent(used_bytes, total_bytes if total_bytes > 0.0 else None)


def _gpu_mem_avg_bytes_from_summaries(
    gpus: Iterable[PerGPUSummary],
) -> Optional[float]:
    """Return average used GPU memory across physical GPU summaries."""
    return average_optional(gpu.mem_avg_bytes for gpu in gpus)


def _node_metric_values(node: SystemNodeSummary) -> Dict[str, Any]:
    """Return the public row metrics for one system node."""
    agg = node.aggregate
    per_gpu = list(node.per_gpu.values())
    gpu_mem_avg_bytes = _gpu_mem_avg_bytes_from_summaries(per_gpu)
    return {
        "cpu_percent": agg.cpu_avg_percent,
        "ram_bytes": agg.ram_avg_bytes,
        "ram_percent": percent(agg.ram_avg_bytes, agg.ram_total_bytes),
        "gpu_util_percent": agg.gpu_util_avg_percent,
        "gpu_mem_bytes": (
            gpu_mem_avg_bytes
            if gpu_mem_avg_bytes is not None
            else agg.gpu_mem_avg_bytes
        ),
        "gpu_mem_percent": _gpu_mem_percent_from_summaries(per_gpu),
        "gpu_temp_c": agg.gpu_temp_avg_c,
        "gpu_power_w": agg.gpu_power_avg_w,
        "gpu_headroom_bytes": _node_headroom_min_bytes(node),
    }


SYSTEM_LOWER_IS_WORSE_METRICS = {
    "gpu_util_percent",
    "gpu_headroom_bytes",
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
    """Average each public System metric across node rows."""
    values = _metric_values_by_row(row_metrics, SYSTEM_METRIC_NAMES)
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
    sorted_values = sorted(value for _row_id, value in pairs)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        median_value = sorted_values[mid]
    else:
        median_value = (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    return min(
        pairs,
        key=lambda item: (abs(item[1] - median_value), item[1], item[0]),
    )


def _row_metric_points(
    row_metrics: Dict[str, Dict[str, Any]],
    *,
    kind: str,
) -> Dict[str, Any]:
    """Build median or worst `{value, idx}` points from node row metrics."""
    values_by_metric = _metric_values_by_row(row_metrics, SYSTEM_METRIC_NAMES)
    points: Dict[str, Any] = {}
    for metric_name, pairs in values_by_metric.items():
        if not pairs:
            points[metric_name] = {"value": None, "idx": None}
            continue
        if kind == "median":
            row_id, value = _closest_row_to_median(pairs)
        elif kind == "worst":
            lower_is_worse = metric_name in SYSTEM_LOWER_IS_WORSE_METRICS
            row_id, value = (
                min(pairs, key=lambda item: (item[1], item[0]))
                if lower_is_worse
                else max(pairs, key=lambda item: (item[1], item[0]))
            )
        else:
            raise ValueError(f"Unsupported System point kind: {kind}")
        points[metric_name] = {"value": value, "idx": row_id}
    return points


def _node_json(
    node: SystemNodeSummary,
    *,
    metrics: Dict[str, Any],
    primary: Dict[str, Any],
    issues: list[Dict[str, Any]],
) -> Dict[str, Any]:
    identity = {
        "global_rank": node.identity.global_rank,
        "local_rank": node.identity.local_rank,
        "node_rank": node.identity.node_rank,
        "hostname": node.identity.hostname,
        "local_world_size": node.identity.local_world_size,
        "world_size": node.identity.world_size,
    }
    return GroupRow(
        identity=identity,
        diagnosis=primary,
        issues=issues,
        metrics=metrics,
    ).to_json()


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.0f}%"


def _fmt_temp(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.1f}C"


def _fmt_node_idx(value: Any) -> str:
    return "n/a" if value is None else f"n{value}"


def _card_stats(
    cluster: SystemClusterSummary,
    *,
    median: Dict[str, Dict[str, Any]],
    worst: Dict[str, Dict[str, Any]],
) -> str:
    if cluster.observed_nodes <= 1:
        node = next(iter(cluster.nodes.values()), None)
        if node is None:
            return "unavailable"
        ram_peak_pct = percent(
            node.aggregate.ram_peak_bytes,
            node.aggregate.ram_total_bytes,
        )
        parts = [
            f"CPU {_fmt_pct(node.aggregate.cpu_avg_percent)}",
            f"RAM {_fmt_pct(ram_peak_pct)}",
        ]
        if node.aggregate.gpu_available:
            parts.extend(
                [
                    f"GPU util {_fmt_pct(node.aggregate.gpu_util_avg_percent)}",
                    f"GPU memory {_fmt_pct(node_gpu_mem_peak_percent(node))}",
                    f"GPU temp {_fmt_temp(node.aggregate.gpu_temp_peak_c)}",
                ]
            )
        return " | ".join(parts)

    parts = [
        (
            "CPU med/worst "
            f"{_fmt_pct(median['cpu_percent']['value'])}/"
            f"{_fmt_pct(worst['cpu_percent']['value'])} "
            f"{_fmt_node_idx(worst['cpu_percent']['idx'])}"
        ),
        (
            "RAM med/worst "
            f"{_fmt_pct(median['ram_percent']['value'])}/"
            f"{_fmt_pct(worst['ram_percent']['value'])} "
            f"{_fmt_node_idx(worst['ram_percent']['idx'])}"
        ),
    ]
    if cluster.aggregate.gpu_available:
        parts.append(
            "GPU util med/worst "
            f"{_fmt_pct(median['gpu_util_percent']['value'])}/"
            f"{_fmt_pct(worst['gpu_util_percent']['value'])} "
            f"{_fmt_node_idx(worst['gpu_util_percent']['idx'])}"
        )
        parts.append(
            "GPU temp med/worst "
            f"{_fmt_temp(median['gpu_temp_c']['value'])}/"
            f"{_fmt_temp(worst['gpu_temp_c']['value'])} "
            f"{_fmt_node_idx(worst['gpu_temp_c']['idx'])}"
        )
    return " | ".join(parts)


def build_system_card(
    cluster: SystemClusterSummary,
) -> tuple[str, Dict[str, Any]]:
    """Build the cluster-shaped System payload and compact card text."""
    node_payloads: Dict[str, Dict[str, Any]] = {}
    node_metrics: Dict[str, Dict[str, Any]] = {}
    all_issue_pairs: list[tuple[DiagnosticIssue, SystemNodeSummary]] = []

    for label, node in cluster.nodes.items():
        diagnosis = _diagnose_node(node)
        node_issues = [
            _scoped_issue_json(issue, node) for issue in diagnosis.issues
        ]
        if diagnosis.issues:
            node_primary = _primary_from_issue(diagnosis.issues[0], node)
        else:
            node_primary = _primary_from_diagnosis(
                diagnosis.primary,
                scope=_scope_for_node(node),
            )
        metrics = _node_metric_values(node)
        node_metrics[label] = metrics
        node_payloads[label] = _node_json(
            node,
            metrics=metrics,
            primary=node_primary,
            issues=node_issues,
        )
        all_issue_pairs.extend((issue, node) for issue in diagnosis.issues)

    ordered_issue_pairs = sorted(all_issue_pairs, key=_issue_sort_key)
    issues = [
        _scoped_issue_json(issue, node) for issue, node in ordered_issue_pairs
    ]

    cluster_diag = _diagnose_aggregate(cluster.aggregate, per_gpu={})
    if ordered_issue_pairs:
        primary = _primary_from_issue(*ordered_issue_pairs[0])
    else:
        primary = _primary_from_diagnosis(
            cluster_diag.primary,
            scope={"level": "cluster"},
        )

    duration_s = duration_from_bounds(
        cluster.aggregate.first_ts,
        cluster.aggregate.last_ts,
    )
    median = _row_metric_points(node_metrics, kind="median")
    worst = _row_metric_points(node_metrics, kind="worst")
    coverage = f"{cluster.observed_nodes}/{cluster.expected_nodes}"
    card = "\n".join(
        [
            (
                "TraceML System Summary | duration "
                f"{format_optional(duration_s, 's', 1)} | "
                f"nodes {coverage} | samples {cluster.aggregate.system_samples}"
            ),
            "System",
            f"- Diagnosis: {primary.get('status', 'NO DATA')}",
            f"- Scope: nodes {coverage} | samples {cluster.aggregate.system_samples}",
            f"- Stats: {_card_stats(cluster, median=median, worst=worst)}",
            f"- Why: {primary.get('reason', 'No system telemetry was recorded.')}",
        ]
    )

    metadata = BaseMetadata(
        mode=_system_mode(cluster),
        duration_s=duration_s,
        samples=cluster.aggregate.system_samples,
        nodes_expected=cluster.expected_nodes,
        nodes_observed=cluster.observed_nodes,
        nodes_coverage=coverage,
        nodes_partial=cluster.partial,
        gpus_observed=cluster.observed_gpus,
        section_metric_names=SYSTEM_METRIC_NAMES,
    )
    global_summary = BaseGlobal(
        index_by="node_rank",
        window=GlobalWindow(
            kind="sample_window",
            alignment="none",
            samples=cluster.aggregate.system_samples,
        ).to_json(),
        average=_average_metrics_from_rows(node_metrics),
        median=median,
        worst=worst,
    )
    payload = BaseSectionPayload(
        metadata=metadata.to_json(),
        diagnosis=primary,
        issues=issues,
        global_summary=global_summary.to_json(),
        groups=BaseGroups(
            by="node_rank",
            rows=node_payloads,
        ).to_json(),
        units={
            "memory": "bytes",
            "temperature": "C",
            "power": "W",
            "util": "%",
        },
        card=card,
    ).to_json()
    return card, payload


def build_system_section_payload(
    data: SystemSectionData,
) -> Dict[str, Any]:
    """Build the JSON-safe System section payload from loaded data."""
    _, payload = build_system_card(data.cluster)
    return payload


__all__ = [
    "build_system_card",
    "build_system_section_payload",
]
