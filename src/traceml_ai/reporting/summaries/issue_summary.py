# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Shared issue-summary helpers for final reporting."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from traceml_ai.diagnostics.common import DiagnosticIssue, DiagnosticResult


def issue_to_json(issue: DiagnosticIssue) -> Dict[str, Any]:
    """
    Serialize one diagnostic issue into a JSON-friendly shape.
    """
    return {
        "kind": issue.kind,
        "status": issue.status,
        "severity": issue.severity,
        "summary": issue.summary,
        "action": issue.action,
        "metric": issue.metric,
        "phase": issue.phase,
        "score": issue.score,
        "share_pct": issue.share_pct,
        "skew_pct": issue.skew_pct,
        "ranks": [int(rank) for rank in issue.ranks],
        "evidence": dict(issue.evidence or {}),
    }


def issues_to_json(
    issues: Sequence[DiagnosticIssue],
) -> List[Dict[str, Any]]:
    """
    Serialize issues in the priority order provided by the diagnosis layer.
    """
    return [issue_to_json(issue) for issue in issues]


def diagnostic_result_to_json(
    result: DiagnosticResult,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Serialize a diagnostic result using the shared final-summary contract.

    `DiagnosticResult` guarantees a non-empty issue list, so the primary JSON
    diagnosis is always the first serialized finding.
    """
    issues = issues_to_json(result.issues)
    if not issues:
        raise ValueError("DiagnosticResult.issues must not be empty")
    return dict(issues[0]), issues
