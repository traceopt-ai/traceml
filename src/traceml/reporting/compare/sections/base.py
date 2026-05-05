"""Base helpers for section-level summary comparison."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from traceml.reporting.compare.model import (
    CompareDiagnosis,
    CompareMetric,
    CompareSection,
)


class SectionComparer(Protocol):
    """Compare one named final-summary section."""

    name: str

    def compare(
        self,
        lhs_payload: Dict[str, Any],
        rhs_payload: Dict[str, Any],
    ) -> CompareSection: ...


def as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def as_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def nested_get(obj: Any, *keys: str) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def diagnosis_status(section: Any) -> Optional[str]:
    if not isinstance(section, dict):
        return None
    block = section.get("primary_diagnosis")
    if not isinstance(block, dict):
        block = section.get("diagnosis_presented")
    if not isinstance(block, dict):
        return None
    return as_str(block.get("status"))


def numeric_metric(
    *,
    key: str,
    label: str,
    unit: Optional[str],
    lhs: Any,
    rhs: Any,
    direction: str,
    delta_unit: Optional[str] = None,
) -> CompareMetric:
    lhs_f = as_float(lhs)
    rhs_f = as_float(rhs)
    delta = None
    pct_change = None
    if lhs_f is not None and rhs_f is not None:
        delta = rhs_f - lhs_f
        if abs(lhs_f) > 1e-12:
            pct_change = 100.0 * delta / lhs_f
    return CompareMetric(
        key=key,
        label=label,
        unit=unit,
        lhs=lhs_f,
        rhs=rhs_f,
        delta=delta,
        pct_change=pct_change,
        delta_unit=delta_unit,
        direction=direction,
    )


def text_metric(
    *,
    key: str,
    label: str,
    lhs: Any,
    rhs: Any,
) -> CompareMetric:
    return CompareMetric(
        key=key,
        label=label,
        unit=None,
        lhs=as_str(lhs),
        rhs=as_str(rhs),
        direction="context",
    )


def section_available(lhs: Any, rhs: Any) -> bool:
    return isinstance(lhs, dict) or isinstance(rhs, dict)


def section_diagnosis(lhs: Any, rhs: Any) -> CompareDiagnosis:
    return CompareDiagnosis(
        lhs=diagnosis_status(lhs), rhs=diagnosis_status(rhs)
    )


__all__ = [
    "SectionComparer",
    "as_float",
    "as_str",
    "diagnosis_status",
    "first_present",
    "nested_get",
    "numeric_metric",
    "section_available",
    "section_diagnosis",
    "text_metric",
]
