"""
Shared diagnosis primitives used across runtime and summary diagnostics.
"""

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Literal, Optional, Tuple

Severity = Literal["info", "warn", "crit"]


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
class DiagnosisContributor:
    """
    One contributing issue or diagnostic signal.

    This is intentionally generic and domain-agnostic. It allows a diagnosis
    engine to surface multiple meaningful signals while still selecting one
    primary diagnosis for compact runtime display.

    Fields
    ------
    kind:
        Stable machine-readable issue label, e.g. `INPUT_STRAGGLER`.
    status:
        User-facing label shown in dashboards or summaries.
    severity:
        Relative seriousness of this contributor.
    summary:
        Concise explanation of the issue.
    action:
        Suggested next step.
    metric:
        Canonical metric key when applicable.
    phase:
        Optional more specific phase label, e.g. `backward`.
    score:
        Normalized scalar score used for ranking or thresholding.
    share_pct:
        Optional share ratio in `[0, 1]`.
    skew_pct:
        Optional skew ratio in `[0, 1]`.
    ranks:
        Ordered tuple of materially involved ranks.
    evidence:
        Small machine-readable evidence block for downstream consumers.
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
class DiagnosisResult:
    """
    Rich diagnosis result returned by extensible diagnosis engines.

    Runtime UX can use `primary` only, while summaries and dashboards can use
    the full result.

    Fields
    ------
    primary:
        The single diagnosis chosen for compact runtime display.
    contributors:
        Ranked contributing issues that were materially present in the same
        analysis window.
    metric_attribution:
        Canonical machine-readable evidence keyed by metric / phase.
    per_rank:
        Optional per-rank local evidence and issue summaries.
    """

    primary: BaseDiagnosis
    contributors: Tuple[DiagnosisContributor, ...] = ()
    metric_attribution: Dict[str, Any] = field(default_factory=dict)
    per_rank: Dict[str, Any] = field(default_factory=dict)


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
    "BaseDiagnosis",
    "DiagnosisContributor",
    "DiagnosisResult",
    "validate_confidence",
    "diagnosis_to_dict",
]
