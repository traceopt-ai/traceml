"""
Structured comparison logic for TraceML final summary JSON artifacts.

This module compares two already-loaded final summary payloads and produces
a stable compare-friendly JSON payload that can be rendered, saved, or logged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from traceml.compare.io import derive_compare_labels
from traceml.compare.verdict import build_compare_verdict
from traceml.final_summary_protocol import utc_now_iso

_STEP_PHASES = ("dataloader", "forward", "backward", "optimizer")


def _as_float(value: Any) -> Optional[float]:
    """
    Best-effort float conversion for optional metrics.
    """
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _as_str(value: Any) -> Optional[str]:
    """
    Best-effort string conversion for optional display fields.
    """
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text or None


def _as_dict(value: Any) -> Optional[Dict[str, Any]]:
    """
    Return a dictionary value if present, otherwise None.
    """
    return value if isinstance(value, dict) else None


def _nested_get(obj: Dict[str, Any], *keys: str) -> Any:
    """
    Safe nested dictionary access.
    """
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _value_delta(lhs: Any, rhs: Any) -> Dict[str, Optional[float]]:
    """
    Compare two numeric values and return lhs/rhs/delta/pct_change.

    `pct_change` is computed relative to the left-hand side and is omitted when
    the left value is missing or zero.
    """
    lhs_f = _as_float(lhs)
    rhs_f = _as_float(rhs)

    if lhs_f is None or rhs_f is None:
        return {
            "lhs": lhs_f,
            "rhs": rhs_f,
            "delta": None,
            "pct_change": None,
        }

    delta = rhs_f - lhs_f
    pct_change = None if abs(lhs_f) < 1e-12 else (100.0 * delta / lhs_f)

    return {
        "lhs": lhs_f,
        "rhs": rhs_f,
        "delta": delta,
        "pct_change": pct_change,
    }


def _value_change(lhs: Any, rhs: Any) -> Dict[str, Any]:
    """
    Compare two textual or categorical values.
    """
    lhs_s = _as_str(lhs)
    rhs_s = _as_str(rhs)
    return {
        "lhs": lhs_s,
        "rhs": rhs_s,
        "changed": lhs_s != rhs_s,
    }


def _compare_step_splits(
    lhs_summary: Dict[str, Any],
    rhs_summary: Dict[str, Any],
    *,
    split_key: str,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Compare step split dictionaries phase-by-phase.
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}

    lhs_split = _nested_get(lhs_summary, "timing_primary", split_key)
    rhs_split = _nested_get(rhs_summary, "timing_primary", split_key)

    lhs_split = lhs_split if isinstance(lhs_split, dict) else {}
    rhs_split = rhs_split if isinstance(rhs_split, dict) else {}

    for phase in _STEP_PHASES:
        out[phase] = _value_delta(lhs_split.get(phase), rhs_split.get(phase))

    return out


def _section_available(summary: Any) -> bool:
    """
    Return True when a top-level summary section is present as a dictionary.
    """
    return isinstance(summary, dict)


def _build_headline(
    *,
    lhs_label: str,
    rhs_label: str,
    step_avg: Dict[str, Optional[float]],
    lhs_step_status: Optional[str],
    rhs_step_status: Optional[str],
) -> str:
    """
    Build one concise compare headline for the top of the report.
    """
    pct = step_avg.get("pct_change")
    if pct is not None and pct >= 5.0:
        return f"{rhs_label} is slower than {lhs_label} on average step time."
    if pct is not None and pct <= -5.0:
        return f"{rhs_label} is faster than {lhs_label} on average step time."
    if lhs_step_status != rhs_step_status and rhs_step_status:
        return (
            f"Average step time is similar, but the diagnosis changed to "
            f"{rhs_step_status}."
        )
    return "Average step time is broadly similar between the two runs."


def build_compare_payload(
    *,
    lhs_payload: Dict[str, Any],
    rhs_payload: Dict[str, Any],
    lhs_path: str | Path,
    rhs_path: str | Path,
) -> Dict[str, Any]:
    """
    Build a structured compare payload from two final summary JSON objects.
    """
    lhs_path = Path(lhs_path).expanduser().resolve()
    rhs_path = Path(rhs_path).expanduser().resolve()

    lhs_label, rhs_label = derive_compare_labels(lhs_path, rhs_path)

    lhs_step_time = lhs_payload.get("step_time", {})
    rhs_step_time = rhs_payload.get("step_time", {})
    lhs_step_memory = lhs_payload.get("step_memory", {})
    rhs_step_memory = rhs_payload.get("step_memory", {})

    step_avg = _value_delta(
        _nested_get(lhs_step_time, "timing_primary", "step_avg_ms"),
        _nested_get(rhs_step_time, "timing_primary", "step_avg_ms"),
    )
    lhs_step_status = _nested_get(lhs_step_time, "diagnosis", "status")
    rhs_step_status = _nested_get(rhs_step_time, "diagnosis", "status")

    payload: Dict[str, Any] = {
        "schema_version": 1,
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
        "availability": {
            "system": _section_available(lhs_payload.get("system"))
            or _section_available(rhs_payload.get("system")),
            "process": _section_available(lhs_payload.get("process"))
            or _section_available(rhs_payload.get("process")),
            "step_time": _section_available(lhs_payload.get("step_time"))
            or _section_available(rhs_payload.get("step_time")),
            "step_memory": _section_available(lhs_payload.get("step_memory"))
            or _section_available(rhs_payload.get("step_memory")),
        },
        "overview": {
            "headline": _build_headline(
                lhs_label=lhs_label,
                rhs_label=rhs_label,
                step_avg=step_avg,
                lhs_step_status=_as_str(lhs_step_status),
                rhs_step_status=_as_str(rhs_step_status),
            ),
            "duration_s": _value_delta(
                lhs_payload.get("duration_s"),
                rhs_payload.get("duration_s"),
            ),
            "step_time_status": _value_change(
                lhs_step_status,
                rhs_step_status,
            ),
            "step_memory_status": _value_change(
                _nested_get(lhs_step_memory, "diagnosis", "status"),
                _nested_get(rhs_step_memory, "diagnosis", "status"),
            ),
        },
        "system": {
            "cpu_avg_percent": _value_delta(
                _nested_get(lhs_payload, "system", "cpu_avg_percent"),
                _nested_get(rhs_payload, "system", "cpu_avg_percent"),
            ),
            "ram_peak_gb": _value_delta(
                _nested_get(lhs_payload, "system", "ram_peak_gb"),
                _nested_get(rhs_payload, "system", "ram_peak_gb"),
            ),
            "gpu_available": _value_change(
                _nested_get(lhs_payload, "system", "gpu_available"),
                _nested_get(rhs_payload, "system", "gpu_available"),
            ),
            "gpu_count": _value_delta(
                _nested_get(lhs_payload, "system", "gpu_count"),
                _nested_get(rhs_payload, "system", "gpu_count"),
            ),
        },
        "process": {
            "cpu_avg_percent": _value_delta(
                _nested_get(lhs_payload, "process", "cpu_avg_percent"),
                _nested_get(rhs_payload, "process", "cpu_avg_percent"),
            ),
            "ram_peak_gb": _value_delta(
                _nested_get(lhs_payload, "process", "ram_peak_gb"),
                _nested_get(rhs_payload, "process", "ram_peak_gb"),
            ),
            "takeaway": _value_change(
                _nested_get(lhs_payload, "process", "takeaway"),
                _nested_get(rhs_payload, "process", "takeaway"),
            ),
        },
        "step_time": {
            "status": _value_change(
                _nested_get(lhs_step_time, "diagnosis", "status"),
                _nested_get(rhs_step_time, "diagnosis", "status"),
            ),
            "presented": {
                "lhs": _as_dict(lhs_step_time.get("diagnosis_presented")),
                "rhs": _as_dict(rhs_step_time.get("diagnosis_presented")),
            },
            "step_avg_ms": step_avg,
            "wait_share_pct": _value_delta(
                _nested_get(lhs_step_time, "timing_primary", "wait_share_pct"),
                _nested_get(rhs_step_time, "timing_primary", "wait_share_pct"),
            ),
            "compute_share_pct": _value_delta(
                _nested_get(
                    lhs_step_time,
                    "timing_primary",
                    "compute_share_pct",
                ),
                _nested_get(
                    rhs_step_time,
                    "timing_primary",
                    "compute_share_pct",
                ),
            ),
            "dominant_phase": _value_change(
                _nested_get(lhs_step_time, "timing_primary", "dominant_phase"),
                _nested_get(rhs_step_time, "timing_primary", "dominant_phase"),
            ),
            "split_ms": _compare_step_splits(
                lhs_step_time,
                rhs_step_time,
                split_key="split_ms",
            ),
            "split_pct": _compare_step_splits(
                lhs_step_time,
                rhs_step_time,
                split_key="split_pct",
            ),
        },
        "step_memory": {
            "status": _value_change(
                _nested_get(lhs_step_memory, "diagnosis", "status"),
                _nested_get(rhs_step_memory, "diagnosis", "status"),
            ),
            "presented": {
                "lhs": _as_dict(lhs_step_memory.get("diagnosis_presented")),
                "rhs": _as_dict(rhs_step_memory.get("diagnosis_presented")),
            },
            "primary_metric": _value_change(
                _nested_get(lhs_step_memory, "primary_metric", "metric"),
                _nested_get(rhs_step_memory, "primary_metric", "metric"),
            ),
            "worst_peak_bytes": _value_delta(
                _nested_get(
                    lhs_step_memory,
                    "primary_metric",
                    "worst_peak_bytes",
                ),
                _nested_get(
                    rhs_step_memory,
                    "primary_metric",
                    "worst_peak_bytes",
                ),
            ),
            "skew_pct": _value_delta(
                _nested_get(lhs_step_memory, "primary_metric", "skew_pct"),
                _nested_get(rhs_step_memory, "primary_metric", "skew_pct"),
            ),
            "trend_worst_delta_bytes": _value_delta(
                _nested_get(
                    lhs_step_memory,
                    "primary_metric",
                    "trend",
                    "worst",
                    "delta_bytes",
                ),
                _nested_get(
                    rhs_step_memory,
                    "primary_metric",
                    "trend",
                    "worst",
                    "delta_bytes",
                ),
            ),
        },
        "text": "",
    }

    payload["verdict"] = build_compare_verdict(
        lhs_payload=lhs_payload,
        rhs_payload=rhs_payload,
        compare_payload=payload,
    )

    return payload
