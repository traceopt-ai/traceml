# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Payload builder for the final-report System section."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

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


def _system_average(
    agg: SystemSummaryAgg, cluster: SystemClusterSummary
) -> Dict[str, Any]:
    per_gpu = {
        gpu_idx: gpu
        for node in cluster.nodes.values()
        for gpu_idx, gpu in node.per_gpu.items()
    }
    gpu_total_bytes = sum(
        float(gpu.mem_total_bytes)
        for gpu in per_gpu.values()
        if gpu.mem_total_bytes is not None
    )
    gpu_mem_percent = percent(
        agg.gpu_mem_avg_bytes,
        gpu_total_bytes if gpu_total_bytes > 0.0 else None,
    )
    node_headroom_values = [
        value
        for node in cluster.nodes.values()
        for value in (_node_headroom_min_bytes(node),)
        if value is not None
    ]
    return {
        "cpu_percent": agg.cpu_avg_percent,
        "ram_bytes": agg.ram_avg_bytes,
        "ram_percent": percent(agg.ram_avg_bytes, agg.ram_total_bytes),
        "gpu_util_percent": agg.gpu_util_avg_percent,
        "gpu_mem_bytes": agg.gpu_mem_avg_bytes,
        "gpu_mem_percent": gpu_mem_percent,
        "gpu_temp_c": agg.gpu_temp_avg_c,
        "gpu_power_w": agg.gpu_power_avg_w,
        "gpu_headroom_bytes": average_optional(node_headroom_values),
    }


def _node_metric_values(node: SystemNodeSummary) -> Dict[str, Any]:
    """Return the public row metrics for one system node."""
    agg = node.aggregate
    gpu_total_bytes = sum(
        float(gpu.mem_total_bytes)
        for gpu in node.per_gpu.values()
        if gpu.mem_total_bytes is not None
    )
    return {
        "cpu_percent": agg.cpu_avg_percent,
        "ram_bytes": agg.ram_avg_bytes,
        "ram_percent": percent(agg.ram_avg_bytes, agg.ram_total_bytes),
        "gpu_util_percent": agg.gpu_util_avg_percent,
        "gpu_mem_bytes": agg.gpu_mem_avg_bytes,
        "gpu_mem_percent": percent(
            agg.gpu_mem_avg_bytes,
            gpu_total_bytes if gpu_total_bytes > 0.0 else None,
        ),
        "gpu_temp_c": agg.gpu_temp_avg_c,
        "gpu_power_w": agg.gpu_power_avg_w,
        "gpu_headroom_bytes": _node_headroom_min_bytes(node),
    }


def _system_node_comparison(
    cluster: SystemClusterSummary,
    *,
    field: str,
) -> Dict[str, Any]:
    specs = {
        "cpu_percent": (
            lambda node: node.aggregate.cpu_peak_percent,
            True,
        ),
        "ram_bytes": (
            lambda node: node.aggregate.ram_peak_bytes,
            True,
        ),
        "ram_percent": (
            lambda node: percent(
                node.aggregate.ram_peak_bytes,
                node.aggregate.ram_total_bytes,
            ),
            True,
        ),
        "gpu_util_percent": (
            lambda node: node.aggregate.gpu_util_avg_percent,
            False,
        ),
        "gpu_mem_bytes": (
            lambda node: node.aggregate.gpu_mem_peak_bytes,
            True,
        ),
        "gpu_mem_percent": (node_gpu_mem_peak_percent, True),
        "gpu_temp_c": (
            lambda node: node.aggregate.gpu_temp_peak_c,
            True,
        ),
        "gpu_power_w": (
            lambda node: node.aggregate.gpu_power_peak_w,
            True,
        ),
        "gpu_headroom_bytes": (_node_headroom_min_bytes, False),
    }
    value_fn, higher_is_worse = specs[field]
    pairs = [
        (label, float(value))
        for label, node in cluster.nodes.items()
        for value in (value_fn(node),)
        if value is not None
    ]
    if not pairs:
        return {
            "median": None,
            "median_node_rank": None,
            "worst": None,
            "worst_node_rank": None,
        }

    values = [value for _label, value in pairs]
    median_value = sorted(values)[len(values) // 2]
    median_label, _ = min(
        pairs,
        key=lambda item: (abs(item[1] - median_value), item[1], item[0]),
    )
    worst_label, worst_value = (
        max(pairs, key=lambda item: (item[1], item[0]))
        if higher_is_worse
        else min(pairs, key=lambda item: (item[1], item[0]))
    )
    return {
        "median": median_value,
        "median_node_rank": median_label,
        "worst": worst_value,
        "worst_node_rank": worst_label,
    }


def _system_median_worst(cluster: SystemClusterSummary) -> tuple[Dict, Dict]:
    median: Dict[str, Any] = {}
    worst: Dict[str, Any] = {}
    for field in (
        "cpu_percent",
        "ram_bytes",
        "ram_percent",
        "gpu_util_percent",
        "gpu_mem_bytes",
        "gpu_mem_percent",
        "gpu_temp_c",
        "gpu_power_w",
        "gpu_headroom_bytes",
    ):
        rollup = _system_node_comparison(cluster, field=field)
        median[field] = {
            "value": rollup["median"],
            "idx": rollup["median_node_rank"],
        }
        worst[field] = {
            "value": rollup["worst"],
            "idx": rollup["worst_node_rank"],
        }
    return median, worst


def _node_json(
    node: SystemNodeSummary,
    *,
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
        metrics=_node_metric_values(node),
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
        node_payloads[label] = _node_json(
            node,
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
    median, worst = _system_median_worst(cluster)
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
        average=_system_average(cluster.aggregate, cluster),
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
