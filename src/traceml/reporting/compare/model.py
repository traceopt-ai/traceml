"""Typed objects shared by compare builders, rules, and renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CompareMetric:
    """One comparable metric from a section."""

    key: str
    label: str
    unit: Optional[str]
    lhs: Any
    rhs: Any
    delta: Optional[float] = None
    pct_change: Optional[float] = None
    delta_unit: Optional[str] = None
    direction: str = "context"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "unit": self.unit,
            "lhs": self.lhs,
            "rhs": self.rhs,
            "delta": self.delta,
            "pct_change": self.pct_change,
            "delta_unit": self.delta_unit,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class CompareDiagnosis:
    """Diagnosis change for a section."""

    lhs: Optional[str]
    rhs: Optional[str]

    @property
    def changed(self) -> bool:
        return self.lhs != self.rhs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lhs": self.lhs,
            "rhs": self.rhs,
            "changed": self.changed,
        }


@dataclass(frozen=True)
class CompareSection:
    """Structured comparison for one final-summary section."""

    name: str
    available: bool
    diagnosis: CompareDiagnosis
    metrics: Dict[str, CompareMetric] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "diagnosis": self.diagnosis.to_dict(),
            "metrics": {
                key: metric.to_dict() for key, metric in self.metrics.items()
            },
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CompareFinding:
    """A rule finding considered for the final compare verdict."""

    status: str
    priority: int
    domain: str
    why: str
    metric: Optional[str] = None
    severity: str = "info"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "priority": self.priority,
            "domain": self.domain,
            "metric": self.metric,
            "severity": self.severity,
            "why": self.why,
        }


__all__ = [
    "CompareDiagnosis",
    "CompareFinding",
    "CompareMetric",
    "CompareSection",
]
