# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
Summary-oriented system diagnosis API.

System diagnosis is intentionally conservative and primarily aimed at final
summary interpretation and JSON output. It is not yet used to change live
runtime rendering behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Tuple

from ..common import (
    BaseDiagnosis,
    DiagnosticIssue,
    DiagnosticResult,
    Severity,
    severity_rank,
)
from .context import (
    SystemDiagnosisInput,
    SystemNodeDiagnosisInput,
    SystemSummarySignals,
    build_system_summary_signals,
)
from .rules import SYSTEM_ISSUE_PRIORITY, run_system_rules


@dataclass(frozen=True)
class SystemDiagnosis(BaseDiagnosis):
    """
    Primary system diagnosis used in final-summary cards and JSON.
    """

    kind: str
    samples_used: int
    scope: Dict[str, Any] = field(default_factory=dict)


def _mk_diag(
    *,
    kind: str,
    severity: Severity,
    status: str,
    reason: str,
    action: str,
    samples_used: int,
    scope: Dict[str, Any] | None = None,
) -> SystemDiagnosis:
    return SystemDiagnosis(
        kind=str(kind),
        severity=severity,
        status=str(status),
        reason=str(reason),
        action=str(action),
        samples_used=int(samples_used),
        scope=dict(scope or {}),
    )


def _default_primary(signals: SystemSummarySignals) -> SystemDiagnosis:
    """Build the default non-issue system diagnosis."""
    if signals.samples <= 0:
        return _mk_diag(
            kind="NO_DATA",
            severity="info",
            status="NO DATA",
            reason="No system telemetry was recorded.",
            action="Collect system telemetry for host-level context.",
            samples_used=signals.samples,
            scope={"level": "cluster"},
        )

    has_gpu_metrics = any(
        value is not None
        for value in (
            signals.gpu_util_avg_percent,
            signals.gpu_mem_peak_percent,
            signals.gpu_temp_peak_c,
            signals.gpu_power_avg_limit_percent,
        )
    )
    reason = (
        "CPU, RAM, and GPU showed no system pressure."
        if has_gpu_metrics
        else "CPU and RAM showed no system pressure."
    )
    return _mk_diag(
        kind="NORMAL",
        severity="info",
        status="NORMAL",
        reason=reason,
        action="Use training diagnostics for model-level bottlenecks.",
        samples_used=signals.samples,
        scope={"level": "cluster"},
    )


def _node_scope(
    node: SystemNodeDiagnosisInput,
    issue: DiagnosticIssue | None = None,
) -> Dict[str, Any]:
    """Return the public scope for a node or node-local GPU issue."""
    scope: Dict[str, Any] = {
        "level": "node",
        "node": node.node_label,
        "node_rank": node.node_rank,
    }
    if issue is not None and issue.ranks:
        scope["level"] = "gpu"
        scope["gpu_idx"] = int(issue.ranks[0])
    return scope


def _scope_text(text: str, scope: Dict[str, Any]) -> str:
    """Add node context to issue text without making normal text noisy."""
    if scope.get("level") == "gpu":
        gpu_idx = scope.get("gpu_idx")
        node = scope.get("node")
        suffix = f" on {node} gpu{gpu_idx}"
        existing = f" on gpu{gpu_idx}"
        if existing in text:
            return text.replace(existing, suffix)
        return f"{text.rstrip('.')}{suffix}."
    if scope.get("level") == "node" and scope.get("node"):
        return f"{text.rstrip('.')} on {scope['node']}."
    return text


def _scoped_issue(
    issue: DiagnosticIssue,
    node: SystemNodeDiagnosisInput,
) -> DiagnosticIssue:
    """Attach node scope to a node-local System issue."""
    scope = _node_scope(node, issue)
    evidence = dict(issue.evidence or {})
    evidence["scope"] = scope
    evidence["samples_used"] = int(node.samples)
    return replace(
        issue,
        summary=_scope_text(issue.summary, scope),
        evidence=evidence,
    )


def _issue_sort_key(issue: DiagnosticIssue) -> tuple:
    """Return the cluster-wide System issue priority key."""
    scope = dict((issue.evidence or {}).get("scope") or {})
    return (
        SYSTEM_ISSUE_PRIORITY.get(issue.kind, 999),
        -severity_rank(issue.severity),
        -float(issue.score or 0.0),
        str(scope.get("node") or ""),
    )


def _diagnose_node(
    node: SystemNodeDiagnosisInput,
) -> Tuple[DiagnosticIssue, ...]:
    """Run regular System rules for one node input."""
    signals = build_system_summary_signals(node)
    issues = run_system_rules(signals) if signals.samples > 0 else ()
    return tuple(_scoped_issue(issue, node) for issue in issues)


def _primary_from_issue(
    issue: DiagnosticIssue,
) -> SystemDiagnosis:
    """Promote the highest-priority scoped issue to section diagnosis."""
    scope = dict((issue.evidence or {}).get("scope") or {})
    return _mk_diag(
        kind=issue.kind,
        severity=issue.severity,
        status=issue.status,
        reason=issue.summary,
        action=issue.action,
        samples_used=int((issue.evidence or {}).get("samples_used") or 0),
        scope=scope,
    )


def diagnose_system(
    data: SystemDiagnosisInput,
) -> DiagnosticResult[SystemDiagnosis]:
    """
    Diagnose cluster-level system pressure from prepared node inputs.
    """
    node_issues = []
    for node in data.per_node.values():
        node_issues.extend(_diagnose_node(node))

    issues = tuple(sorted(node_issues, key=_issue_sort_key))
    if issues:
        primary = _primary_from_issue(issues[0])
    else:
        signals = build_system_summary_signals(data)
        primary = _default_primary(signals)

    return DiagnosticResult(
        primary=primary,
        issues=issues,
    )


__all__ = [
    "SystemDiagnosis",
    "diagnose_system",
]
