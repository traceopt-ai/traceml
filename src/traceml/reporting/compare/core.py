"""Structured comparison logic for TraceML final-summary JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from traceml.reporting.compare.io import derive_compare_labels
from traceml.reporting.compare.sections import SECTION_COMPARERS
from traceml.reporting.compare.verdict import build_compare_verdict
from traceml.sdk.protocol import utc_now_iso


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _metric_alias(section: Dict[str, Any], key: str) -> Dict[str, Any]:
    metric = section.get("metrics", {}).get(key)
    return metric if isinstance(metric, dict) else {}


def _legacy_section_aliases(
    sections: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Expose historical top-level compare keys while callers migrate."""
    step_time = sections.get("step_time", {})
    step_memory = sections.get("step_memory", {})
    process = sections.get("process", {})
    system = sections.get("system", {})

    return {
        "step_time": {
            "status": step_time.get("diagnosis", {}),
            "presented": {
                "lhs": {"status": step_time.get("diagnosis", {}).get("lhs")},
                "rhs": {"status": step_time.get("diagnosis", {}).get("rhs")},
            },
            "step_avg_ms": _metric_alias(step_time, "step_avg_ms"),
            "wait_share_pct": _metric_alias(step_time, "wait_share_pct"),
            "compute_ms": _metric_alias(step_time, "compute_ms"),
            "wait_ms": _metric_alias(step_time, "wait_ms"),
            "input_ms": _metric_alias(step_time, "input_ms"),
            "dominant_phase": _metric_alias(step_time, "dominant_phase"),
        },
        "step_memory": {
            "status": step_memory.get("diagnosis", {}),
            "presented": {
                "lhs": {"status": step_memory.get("diagnosis", {}).get("lhs")},
                "rhs": {"status": step_memory.get("diagnosis", {}).get("rhs")},
            },
            "worst_peak_bytes": _metric_alias(
                step_memory,
                "peak_reserved_bytes",
            ),
            "skew_pct": _metric_alias(step_memory, "memory_skew_pct"),
            "trend_worst_delta_bytes": _metric_alias(
                step_memory,
                "trend_worst_delta_bytes",
            ),
        },
        "process": {
            "cpu_avg_percent": _metric_alias(process, "cpu_avg_percent"),
            "ram_peak_gb": _metric_alias(process, "rss_peak_gb"),
        },
        "system": {
            "cpu_avg_percent": _metric_alias(system, "cpu_avg_percent"),
            "ram_peak_gb": _metric_alias(system, "ram_peak_gb"),
            "gpu_util_avg_percent": _metric_alias(
                system,
                "gpu_util_avg_percent",
            ),
            "gpu_memory_peak_percent": _metric_alias(
                system,
                "gpu_memory_peak_percent",
            ),
        },
    }


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
        "generated_at": utc_now_iso(),
        "lhs": {
            "path": str(lhs_path),
            "label": lhs_label,
            "file_name": lhs_path.name,
            "parent_name": lhs_path.parent.name,
            "duration_s": _as_float(lhs_payload.get("duration_s")),
        },
        "rhs": {
            "path": str(rhs_path),
            "label": rhs_label,
            "file_name": rhs_path.name,
            "parent_name": rhs_path.parent.name,
            "duration_s": _as_float(rhs_payload.get("duration_s")),
        },
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
        },
        "text": "",
    }

    payload.update(_legacy_section_aliases(sections))
    payload["verdict"] = build_compare_verdict(
        lhs_payload=lhs_payload,
        rhs_payload=rhs_payload,
        compare_payload=payload,
    )

    return payload


__all__ = ["build_compare_payload"]
