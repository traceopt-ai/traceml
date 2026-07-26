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
    One concrete diagnostic finding or state in an analyzed window.

    Notes
    -----
    - `kind` is the stable machine-readable label.
    - `status` is the user-facing label shown in summaries and dashboards.
    - `score` is an optional normalized scalar used for ranking issues.
    - `ranks` contains the materially affected ranks in stable order.
    - Neutral states such as NORMAL, BALANCED, NO_DATA, and WARMUP also use
      this shape so final-summary JSON has one stable contract.
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

    `issues` is the canonical ordered list of findings/states. It is never
    empty: if no rule emitted a concrete issue, the neutral `primary`
    diagnosis is converted into one issue-shaped state. Final-summary JSON can
    therefore expose `diagnosis == issues[0]` across every section.
    """

    primary: PrimaryT
    issues: Tuple[DiagnosticIssue, ...] = ()
    metric_attribution: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        issues = ensure_primary_issue(self.primary, self.issues)
        object.__setattr__(self, "issues", issues)


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


def validate_confidence(confidence: Optional[float]) -> None:
    """
    Validate an optional confidence value.
    """
    if confidence is None:
        return
    if not (0.0 <= float(confidence) <= 1.0):
        raise ValueError("confidence must be between 0.0 and 1.0")


def diagnosis_to_issue(diagnosis: Any) -> DiagnosticIssue:
    """
    Convert a primary diagnosis into the shared issue/finding shape.

    This is used for neutral states and data-availability states where no
    diagnostic rule fired but the section still needs one canonical finding.
    """
    if not is_dataclass(diagnosis):
        raise TypeError("diagnosis_to_issue expects a dataclass instance")

    payload = asdict(diagnosis)
    kind = str(payload.get("kind") or payload.get("status") or "UNKNOWN")
    metric = payload.get("metric")
    phase = payload.get("phase")
    ranks: Tuple[int, ...] = ()
    evidence: Dict[str, Any] = {}

    evidence_fields = set(payload) - {
        "kind",
        "severity",
        "status",
        "reason",
        "action",
        "metric",
        "phase",
        "score",
        "share_pct",
        "skew_pct",
    }
    for key in sorted(evidence_fields):
        value = payload.get(key)
        if value is None or value == {}:
            continue
        if key == "worst_rank":
            ranks = (int(value),)
        evidence[key] = value

    return DiagnosticIssue(
        kind=kind,
        status=str(payload.get("status") or kind),
        severity=payload.get("severity") or "info",
        summary=str(payload.get("reason") or ""),
        action=str(payload.get("action") or ""),
        metric=str(metric) if metric is not None else None,
        phase=str(phase) if phase is not None else None,
        score=payload.get("score"),
        share_pct=payload.get("share_pct"),
        skew_pct=payload.get("skew_pct"),
        ranks=ranks,
        evidence=evidence,
    )


def ensure_primary_issue(
    primary: Any,
    issues: Sequence[DiagnosticIssue],
) -> Tuple[DiagnosticIssue, ...]:
    """
    Return a non-empty issue tuple with the primary diagnosis first.
    """
    issue_tuple = tuple(issues or ())
    if not issue_tuple:
        return (diagnosis_to_issue(primary),)

    primary_kind = str(getattr(primary, "kind", "") or "")
    if not primary_kind:
        return issue_tuple

    for idx, issue in enumerate(issue_tuple):
        if issue.kind == primary_kind:
            return (
                issue,
                *issue_tuple[:idx],
                *issue_tuple[idx + 1 :],
            )
    return (diagnosis_to_issue(primary), *issue_tuple)


__all__ = [
    "Severity",
    "ContextT",
    "BaseDiagnosis",
    "DiagnosticIssue",
    "DiagnosticResult",
    "DiagnosticRule",
    "severity_rank",
    "validate_confidence",
    "diagnosis_to_issue",
    "ensure_primary_issue",
]
