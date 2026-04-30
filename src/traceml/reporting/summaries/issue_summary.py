"""
Shared issue-summary helpers for final reporting.

These helpers keep the end-of-run summary builders compact and consistent
without forcing a heavier object hierarchy than the current schema needs.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from traceml.diagnostics.common import DiagnosticIssue, sort_issues


def _rank_label(rank: int) -> str:
    return f"r{int(rank)}"


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
    Serialize a sequence of issues in stable priority order.
    """
    return [issue_to_json(issue) for issue in sort_issues(tuple(issues))]


def issues_by_rank_json(
    issues: Sequence[DiagnosticIssue],
    *,
    rank_keys: Optional[Iterable[int]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Group issues by rank and preserve any rank-less issues separately.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    if rank_keys is not None:
        for rank in rank_keys:
            out[str(int(rank))] = []

    unassigned: List[Dict[str, Any]] = []
    for issue in sort_issues(tuple(issues)):
        payload = issue_to_json(issue)
        if issue.ranks:
            for rank in issue.ranks:
                out.setdefault(str(int(rank)), []).append(payload)
        else:
            unassigned.append(payload)
    return out, unassigned


def issues_by_metric_json(
    issues: Sequence[DiagnosticIssue],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Group issues by metric and preserve rank-less / metric-less issues separately.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    unassigned: List[Dict[str, Any]] = []
    for issue in sort_issues(tuple(issues)):
        payload = issue_to_json(issue)
        if issue.metric:
            out.setdefault(str(issue.metric), []).append(payload)
        else:
            unassigned.append(payload)
    return out, unassigned


def issue_compact_label(issue: DiagnosticIssue) -> str:
    """
    Render one compact human-readable issue label.
    """
    label = str(issue.status)
    if issue.ranks:
        label += f" @ {','.join(_rank_label(rank) for rank in issue.ranks)}"
    if issue.metric:
        label += f" [{issue.metric}]"
    return label


def issues_compact_text(
    issues: Sequence[DiagnosticIssue],
    *,
    max_items: int = 3,
) -> Optional[str]:
    """
    Render a compact issue summary string suitable for a summary card.
    """
    ordered = list(sort_issues(tuple(issues)))
    if not ordered:
        return None

    limit = max(1, int(max_items))
    pieces = [issue_compact_label(issue) for issue in ordered[:limit]]
    remaining = len(ordered) - len(pieces)
    if remaining > 0:
        pieces.append(f"+{remaining} more")
    return "; ".join(pieces)
