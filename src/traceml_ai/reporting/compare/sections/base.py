# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Base helpers for section-level summary comparison."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from traceml_ai.diagnostics.step_time.names import (
    OVERHEAD_HEAVY_KIND,
    canonical_issue_kind,
)
from traceml_ai.reporting.compare.model import (
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


def global_average(section: Any, metric_name: str) -> Any:
    """Read one metric from the new final-summary `global.average` block."""
    return nested_get(section, "global", "average", metric_name)


def global_point_value(section: Any, kind: str, metric_name: str) -> Any:
    """Read the numeric value from `global.median/worst.<metric>`."""
    point = nested_get(section, "global", kind, metric_name)
    if not isinstance(point, dict):
        return None
    return point.get("value")


def diagnosis_status(section: Any) -> Optional[str]:
    if not isinstance(section, dict):
        return None
    block = section.get("diagnosis")
    if not isinstance(block, dict):
        return None
    kind = as_str(block.get("kind"))
    if kind is not None and canonical_issue_kind(kind) == OVERHEAD_HEAVY_KIND:
        return "OVERHEAD-HEAVY"
    status = as_str(block.get("status"))
    if status == "WAIT-HEAVY":
        return "OVERHEAD-HEAVY"
    return status


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
    "global_average",
    "global_point_value",
    "nested_get",
    "numeric_metric",
    "section_available",
    "section_diagnosis",
    "text_metric",
]
