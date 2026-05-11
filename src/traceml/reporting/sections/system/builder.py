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
from traceml.diagnostics.system.policy import DEFAULT_SYSTEM_POLICY
from traceml.reporting.sections.system.loader import SystemSectionData
from traceml.reporting.sections.system.model import (
    MetricRollup,
    SystemClusterSummary,
    SystemNodeSummary,
    SystemSummaryAgg,
    _band_name,
    _highest_gpu_memory_percent,
    _per_gpu_to_diagnosis_input,
    _per_gpu_to_json,
    _percent,
    node_gpu_headroom_min_gb,
    node_gpu_mem_peak_percent,
    rollup_metric,
)
from traceml.reporting.summaries.issue_summary import issue_to_json
from traceml.reporting.summaries.summary_formatting import (
    bytes_to_gb,
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
        per_gpu=_per_gpu_to_diagnosis_input(per_gpu),
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


def _rollup_to_json(rollup: MetricRollup) -> Dict[str, Optional[float]]:
    return {
        "median": rollup.median,
        "worst": rollup.worst,
        "worst_node": rollup.worst_node,
    }


def _node_rollup(cluster: SystemClusterSummary) -> Dict[str, Any]:
    nodes = cluster.nodes
    return {
        "cpu": _rollup_to_json(
            rollup_metric(
                nodes,
                value_fn=lambda node: node.aggregate.cpu_peak_percent,
                higher_is_worse=True,
            )
        ),
        "ram": _rollup_to_json(
            rollup_metric(
                nodes,
                value_fn=lambda node: _percent(
                    node.aggregate.ram_peak_bytes,
                    node.aggregate.ram_total_bytes,
                ),
                higher_is_worse=True,
            )
        ),
        "gpu_util": _rollup_to_json(
            rollup_metric(
                nodes,
                value_fn=lambda node: node.aggregate.gpu_util_avg_percent,
                higher_is_worse=False,
            )
        ),
        "gpu_mem": _rollup_to_json(
            rollup_metric(
                nodes,
                value_fn=node_gpu_mem_peak_percent,
                higher_is_worse=True,
            )
        ),
        "gpu_temp": _rollup_to_json(
            rollup_metric(
                nodes,
                value_fn=lambda node: node.aggregate.gpu_temp_peak_c,
                higher_is_worse=True,
            )
        ),
        "gpu_headroom_gb": _rollup_to_json(
            rollup_metric(
                nodes,
                value_fn=node_gpu_headroom_min_gb,
                higher_is_worse=False,
            )
        ),
    }


def _cluster_json(
    agg: SystemSummaryAgg, cluster: SystemClusterSummary
) -> Dict:
    per_gpu = {
        gpu_idx: gpu
        for node in cluster.nodes.values()
        for gpu_idx, gpu in node.per_gpu.items()
    }
    gpu_mem_peak_percent = _highest_gpu_memory_percent(per_gpu)
    gpu_power_avg_limit_percent = max(
        (
            value
            for node in cluster.nodes.values()
            for value in (
                _percent(gpu.power_avg_w, gpu.power_limit_w)
                for gpu in node.per_gpu.values()
            )
            if value is not None
        ),
        default=None,
    )
    return {
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
            "avg_gb": bytes_to_gb(agg.ram_avg_bytes),
            "peak_gb": bytes_to_gb(agg.ram_peak_bytes),
            "total_gb": bytes_to_gb(agg.ram_total_bytes),
            "peak_percent": _percent(agg.ram_peak_bytes, agg.ram_total_bytes),
            "peak_band": _band_name(
                DEFAULT_SYSTEM_POLICY.ram_peak_percent.classify(
                    _percent(agg.ram_peak_bytes, agg.ram_total_bytes)
                )
            ),
        },
        "gpu": {
            "available": agg.gpu_available,
            "count": cluster.observed_gpus,
            "util_avg_percent": agg.gpu_util_avg_percent,
            "util_peak_percent": agg.gpu_util_peak_percent,
            "util_avg_band": _band_name(
                DEFAULT_SYSTEM_POLICY.gpu_util_avg_percent.classify(
                    agg.gpu_util_avg_percent
                )
            ),
            "mem_avg_gb": bytes_to_gb(agg.gpu_mem_avg_bytes),
            "mem_peak_gb": bytes_to_gb(agg.gpu_mem_peak_bytes),
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
            "headroom_min_gb": min(
                (
                    value
                    for node in cluster.nodes.values()
                    for value in (node_gpu_headroom_min_gb(node),)
                    if value is not None
                ),
                default=None,
            ),
        },
    }


def _node_json(
    node: SystemNodeSummary,
    *,
    primary: Dict[str, Any],
    issues: list[Dict[str, Any]],
) -> Dict[str, Any]:
    agg = node.aggregate
    per_gpu_json = _per_gpu_to_json(node.per_gpu)
    for gpu_idx, gpu in node.per_gpu.items():
        entry = per_gpu_json[str(gpu_idx)]
        entry["headroom_min_gb"] = (
            None
            if gpu.mem_total_bytes is None or gpu.mem_peak_bytes is None
            else max(0.0, gpu.mem_total_bytes - gpu.mem_peak_bytes)
            / 1_000_000_000.0
        )
    return {
        "node_rank": node.identity.node_rank,
        "hostname": node.identity.hostname,
        "global_rank": node.identity.global_rank,
        "local_rank": node.identity.local_rank,
        "local_world_size": node.identity.local_world_size,
        "pid": node.identity.pid,
        "samples": agg.system_samples,
        "primary_diagnosis": primary,
        "issues": issues,
        "cpu": {
            "avg_percent": agg.cpu_avg_percent,
            "peak_percent": agg.cpu_peak_percent,
        },
        "ram": {
            "avg_gb": bytes_to_gb(agg.ram_avg_bytes),
            "peak_gb": bytes_to_gb(agg.ram_peak_bytes),
            "total_gb": bytes_to_gb(agg.ram_total_bytes),
            "peak_percent": _percent(agg.ram_peak_bytes, agg.ram_total_bytes),
        },
        "gpu": {
            "available": agg.gpu_available,
            "count": len(node.per_gpu),
            "util_avg_percent": agg.gpu_util_avg_percent,
            "util_peak_percent": agg.gpu_util_peak_percent,
            "mem_peak_gb": bytes_to_gb(agg.gpu_mem_peak_bytes),
            "mem_peak_percent": node_gpu_mem_peak_percent(node),
            "temp_peak_c": agg.gpu_temp_peak_c,
            "headroom_min_gb": node_gpu_headroom_min_gb(node),
        },
        "per_gpu": per_gpu_json,
    }


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.0f}%"


def _fmt_temp(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.1f}C"


def _card_stats(
    cluster: SystemClusterSummary,
    rollup: Dict[str, Dict[str, Any]],
) -> str:
    if cluster.observed_nodes <= 1:
        node = next(iter(cluster.nodes.values()), None)
        if node is None:
            return "unavailable"
        ram_peak_pct = _percent(
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
            f"{_fmt_pct(rollup['cpu']['median'])}/"
            f"{_fmt_pct(rollup['cpu']['worst'])} "
            f"{rollup['cpu']['worst_node']}"
        ),
        (
            "RAM med/worst "
            f"{_fmt_pct(rollup['ram']['median'])}/"
            f"{_fmt_pct(rollup['ram']['worst'])} "
            f"{rollup['ram']['worst_node']}"
        ),
    ]
    if cluster.aggregate.gpu_available:
        parts.append(
            "GPU util med/worst "
            f"{_fmt_pct(rollup['gpu_util']['median'])}/"
            f"{_fmt_pct(rollup['gpu_util']['worst'])} "
            f"{rollup['gpu_util']['worst_node']}"
        )
        parts.append(
            "GPU temp med/worst "
            f"{_fmt_temp(rollup['gpu_temp']['median'])}/"
            f"{_fmt_temp(rollup['gpu_temp']['worst'])} "
            f"{rollup['gpu_temp']['worst_node']}"
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
    rollup = _node_rollup(cluster)
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
            f"- Stats: {_card_stats(cluster, rollup)}",
            f"- Why: {primary.get('reason', 'No system telemetry was recorded.')}",
        ]
    )

    payload = {
        "overview": {
            "scope": "cluster",
            "duration_s": duration_s,
            "samples": cluster.aggregate.system_samples,
            "nodes": {
                "expected": cluster.expected_nodes,
                "observed": cluster.observed_nodes,
                "coverage": coverage,
                "partial": cluster.partial,
            },
            "gpus": {"observed": cluster.observed_gpus},
        },
        "primary_diagnosis": primary,
        "issues": issues,
        "cluster": _cluster_json(cluster.aggregate, cluster),
        "node_rollup": rollup,
        "per_node": node_payloads,
        "units": {
            "memory": "GB",
            "temperature": "C",
            "power": "W",
            "util": "%",
        },
        "card": card,
    }
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
