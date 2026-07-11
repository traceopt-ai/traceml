# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Structured comparison logic for TraceML final-summary JSON."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from traceml_ai.reporting.compare.io import derive_compare_labels
from traceml_ai.reporting.compare.model import CompareDiagnosis
from traceml_ai.reporting.compare.sections import SECTION_COMPARERS
from traceml_ai.reporting.compare.verdict import build_compare_verdict


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _utc_now_iso() -> str:
    """Return a UTC timestamp without importing the training SDK."""
    return datetime.now(timezone.utc).isoformat()


def _primary_diagnosis_status(payload: Dict[str, Any]) -> str | None:
    """Return the top-level primary diagnosis status, if present."""
    block = payload.get("primary_diagnosis")
    if not isinstance(block, dict):
        return None
    value = block.get("status")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _schema_version(payload: Dict[str, Any]) -> Any:
    """Return the input summary schema version as stored in the payload."""
    return payload.get("schema_version")


def _schema_warnings(
    lhs_payload: Dict[str, Any],
    rhs_payload: Dict[str, Any],
) -> list[str]:
    """Return compare warnings for known schema-semantic mismatches."""
    lhs_version = _schema_version(lhs_payload)
    rhs_version = _schema_version(rhs_payload)
    if lhs_version == rhs_version:
        return []
    return [
        (
            "Summary schema versions differ: A uses "
            f"{lhs_version}, B uses {rhs_version}. Step Time fields changed "
            "in schema 1.6, so Step Time deltas may not be directly "
            "comparable."
        )
    ]


def build_compare_payload(
    *,
    lhs_payload: Dict[str, Any],
    rhs_payload: Dict[str, Any],
    lhs_path: str | Path,
    rhs_path: str | Path,
) -> Dict[str, Any]:
    """Build a structured compare payload from two final summary JSON files."""
    lhs_path = Path(lhs_path).expanduser().resolve()
    rhs_path = Path(rhs_path).expanduser().resolve()
    lhs_label, rhs_label = derive_compare_labels(lhs_path, rhs_path)

    section_results = [
        comparer.compare(lhs_payload, rhs_payload)
        for comparer in SECTION_COMPARERS
    ]
    sections = {result.name: result.to_dict() for result in section_results}

    payload: Dict[str, Any] = {
        "schema_version": 2,
        "generated_at": _utc_now_iso(),
        "lhs": {
            "path": str(lhs_path),
            "label": lhs_label,
            "file_name": lhs_path.name,
            "parent_name": lhs_path.parent.name,
            "schema_version": _schema_version(lhs_payload),
            "duration_s": _as_float(lhs_payload.get("duration_s")),
        },
        "rhs": {
            "path": str(rhs_path),
            "label": rhs_label,
            "file_name": rhs_path.name,
            "parent_name": rhs_path.parent.name,
            "schema_version": _schema_version(rhs_payload),
            "duration_s": _as_float(rhs_payload.get("duration_s")),
        },
        "warnings": _schema_warnings(lhs_payload, rhs_payload),
        "sections": sections,
        "availability": {
            name: section.get("available", False)
            for name, section in sections.items()
        },
        "overview": {
            "duration_s": {
                "lhs": _as_float(lhs_payload.get("duration_s")),
                "rhs": _as_float(rhs_payload.get("duration_s")),
            },
            "primary_diagnosis": CompareDiagnosis(
                lhs=_primary_diagnosis_status(lhs_payload),
                rhs=_primary_diagnosis_status(rhs_payload),
            ).to_dict(),
        },
        "text": "",
    }

    payload["verdict"] = build_compare_verdict(
        lhs_payload=lhs_payload,
        rhs_payload=rhs_payload,
        compare_payload=payload,
    )

    return payload


__all__ = ["build_compare_payload"]
