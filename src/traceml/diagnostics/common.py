"""
Shared diagnosis primitives used across runtime and summary diagnostics.
"""

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Literal, Optional

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
    "validate_confidence",
    "diagnosis_to_dict",
]
