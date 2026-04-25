"""
Reusable diagnosis framework primitives.

This module provides the generic building blocks used by domain-specific
diagnostics such as step time, and is intentionally small:

- `DiagnosticIssue` represents one concrete issue signal.
- `DiagnosticResult` stores one primary diagnosis plus richer supporting data.
- `DiagnosticRule` is the minimal rule contract used by diagnosis runners.

The goal is to keep domain logic modular and extensible without forcing a heavy
inheritance hierarchy onto every diagnostics module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

from .common import BaseDiagnosis, Severity

ContextT = TypeVar("ContextT")


@dataclass(frozen=True)
class DiagnosticIssue:
    """
    One concrete issue detected in an analyzed window.

    Notes
    -----
    - `kind` is the stable machine-readable label.
    - `status` is the user-facing label shown in summaries and dashboards.
    - `score` is an optional normalized scalar used for ranking contributors.
    - `ranks` contains the materially affected ranks in stable order.
    """

    kind: str
    status: str
    severity: Severity
    summary: str
    action: str
    metric: Optional[str] = None
    phase: Optional[str] = None
    score: Optional[float] = None
    share_pct: Optional[float] = None
    skew_pct: Optional[float] = None
    ranks: Tuple[int, ...] = ()
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticResult:
    """
    Full diagnosis output for one analyzed window.

    Runtime UX usually consumes only `primary`. Richer downstream consumers such
    as final summaries and dashboards can additionally use:
    - `issues`
    - `metric_attribution`
    - `per_rank`
    """

    primary: BaseDiagnosis
    issues: Tuple[DiagnosticIssue, ...] = ()
    metric_attribution: Dict[str, Any] = field(default_factory=dict)
    per_rank: Dict[str, Any] = field(default_factory=dict)


class DiagnosticRule(Protocol, Generic[ContextT]):
    """
    Minimal contract for a modular diagnosis rule.
    """

    name: str

    def evaluate(self, context: ContextT) -> Optional[DiagnosticIssue]:
        """
        Return one issue if the rule materially triggers, else None.
        """


def severity_rank(severity: Severity) -> int:
    """
    Map severity to a numeric ranking value.
    """
    return {"crit": 2, "warn": 1, "info": 0}.get(severity, 0)


def sort_issues(
    issues: Sequence[DiagnosticIssue],
) -> Tuple[DiagnosticIssue, ...]:
    """
    Return issues sorted by severity, score, and attribution breadth.
    """
    return tuple(
        sorted(
            issues,
            key=lambda item: (
                severity_rank(item.severity),
                float(item.score or 0.0),
                len(item.ranks),
            ),
            reverse=True,
        )
    )


__all__ = [
    "ContextT",
    "DiagnosticIssue",
    "DiagnosticResult",
    "DiagnosticRule",
    "severity_rank",
    "sort_issues",
]
