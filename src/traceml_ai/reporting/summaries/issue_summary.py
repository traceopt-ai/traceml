# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Shared issue-summary helpers for final reporting."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from traceml_ai.diagnostics.common import DiagnosticIssue


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
