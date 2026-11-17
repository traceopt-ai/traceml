# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Tracker-friendly projections of TraceML final summaries."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

TRACKER_PREFIX = "traceml"

METRICS_BY_SECTION = {
    "step_time": (
        "input_wait_ms",
        "step_time_ms",
        "traced_step_time_ms",
        "step_time_cpu_ms",
        "step_time_gpu_ms",
        "traced_step_time_cpu_ms",
        "traced_step_time_gpu_ms",
        "dataloader_fetch_cpu_ms",
        "h2d_ms",
        "compute_ms",
        "residual_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
    ),
    "step_memory": (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ),
    "system": (
        "cpu_percent",
        "ram_percent",
        "gpu_util_percent",
        "gpu_mem_percent",
        "gpu_temp_c",
        "gpu_power_w",
    ),
    "process": (
        "cpu_percent",
        "ram_percent",
        "gpu_mem_reserved_percent",
    ),
}

SECTIONS = ("system", "process", "step_time", "step_memory")

# The compact-summary fields a tracker helper copies into its run, so runs
# can be sorted and compared by the verdict and the two step timings.
_COMPARISON_KEYS = (
    f"{TRACKER_PREFIX}/primary/status",
    f"{TRACKER_PREFIX}/step_time/step_time_ms",
    f"{TRACKER_PREFIX}/step_time/input_wait_ms",
)
_NUMERIC_COMPARISON_KEYS = frozenset(_COMPARISON_KEYS[1:])


def _set_scalar(out: Dict[str, Any], key: str, value: Any) -> None:
    """Set one scalar tracker key, skipping missing and non-scalar values."""
    if value is None:
        return
    if isinstance(value, (str, int, float, bool)):
        out[f"{TRACKER_PREFIX}/{key}"] = value


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping or an empty mapping for malformed summary blocks."""
    return value if isinstance(value, Mapping) else {}


def compact_summary(
    payload: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return a flat tracker-friendly summary from final_summary.json data."""
    if payload is None:
        return None

    out: Dict[str, Any] = {}
    _set_scalar(out, "schema_version", payload.get("schema_version"))
    _set_scalar(out, "duration_s", payload.get("duration_s"))

    primary = _mapping(payload.get("primary_diagnosis"))
    _set_scalar(out, "primary/kind", primary.get("kind"))
    _set_scalar(out, "primary/status", primary.get("status"))
    _set_scalar(out, "primary/severity", primary.get("severity"))
    _set_scalar(out, "primary/section", primary.get("section"))

    for section in SECTIONS:
        section_payload = _mapping(payload.get(section))
        if not section_payload:
            continue

        diagnosis = _mapping(section_payload.get("diagnosis"))
        _set_scalar(out, f"{section}/status", diagnosis.get("status"))
        _set_scalar(out, f"{section}/severity", diagnosis.get("severity"))

        global_summary = _mapping(section_payload.get("global"))
        window = _mapping(global_summary.get("window"))
        if section == "step_time":
            _set_scalar(
                out,
                "step_time/diagnosis_clock",
                window.get("diagnosis_clock"),
            )
        average = _mapping(global_summary.get("average"))
        for metric in METRICS_BY_SECTION.get(section, ()):
            _set_scalar(out, f"{section}/{metric}", average.get(metric))

    return out


def _is_finite_number(value: Any) -> bool:
    """Return whether a non-bool number converts to a finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        # An int beyond float range raises OverflowError here.
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(number)


def tracker_comparison_fields(
    payload: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return the three run-comparison fields present in a final summary.

    The keys are ``traceml/primary/status``,
    ``traceml/step_time/step_time_ms`` and
    ``traceml/step_time/input_wait_ms``, taken from ``compact_summary()``.
    A field the summary does not carry is left out. A timing that is a bool,
    not a number, or not convertible to a finite float is left out too, so a
    tracker never records an unmeasured value as a number.
    """
    compact = compact_summary(payload) or {}
    fields: Dict[str, Any] = {}
    for key in _COMPARISON_KEYS:
        if key not in compact:
            continue
        value = compact[key]
        if key in _NUMERIC_COMPARISON_KEYS and not _is_finite_number(value):
            continue
        fields[key] = value
    return fields


__all__ = [
    "METRICS_BY_SECTION",
    "SECTIONS",
    "TRACKER_PREFIX",
    "compact_summary",
    "tracker_comparison_fields",
]
