"""
Shared diagnosis primitives used across runtime and summary diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import (
    Any,
    Dict,
    Generic,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

Severity = Literal["info", "warn", "crit"]
ContextT = TypeVar("ContextT")


@dataclass(frozen=True)
class BaseDiagnosis:
    """
    Base contract shared by domain-specific diagnosis objects.

    Keep this base class minimal and required-only so subclasses can safely add
    required fields across Python versions.
    """

    severity: Severity
    status: str
    reason: str
    action: str


@dataclass(frozen=True)
class DiagnosticIssue:
    """
    One concrete issue detected in an analyzed window.

    Notes
    -----
    - `kind` is the stable machine-readable label.
    - `status` is the user-facing label shown in summaries and dashboards.
    - `score` is an optional normalized scalar used for ranking issues.
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


PrimaryT = TypeVar("PrimaryT", bound=BaseDiagnosis)


@dataclass(frozen=True)
class DiagnosticResult(Generic[PrimaryT]):
    """
    Full diagnosis output for one analyzed window.

    Runtime UX usually consumes only `primary`. Richer downstream consumers
    such as final summaries and dashboards can additionally use:
    - `issues`
    - `metric_attribution`
    - `per_rank`
    """

    primary: PrimaryT
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


def validate_confidence(confidence: Optional[float]) -> None:
    """
    Validate an optional confidence value.
    """
    if confidence is None:
        return
    if not (0.0 <= float(confidence) <= 1.0):
        raise ValueError("confidence must be between 0.0 and 1.0")


def diagnosis_to_dict(
    diagnosis: Optional[Any],
    *,
    drop_none: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Convert a diagnosis dataclass to a JSON-serializable dictionary.
    """
    if diagnosis is None:
        return None
    if not is_dataclass(diagnosis):
        raise TypeError("diagnosis_to_dict expects a dataclass instance")

    out: Dict[str, Any] = asdict(diagnosis)
    if drop_none:
        out = {k: v for k, v in out.items() if v is not None}
    return out


__all__ = [
    "Severity",
    "ContextT",
    "BaseDiagnosis",
    "DiagnosticIssue",
    "DiagnosticResult",
    "DiagnosticRule",
    "severity_rank",
    "sort_issues",
    "validate_confidence",
    "diagnosis_to_dict",
]
