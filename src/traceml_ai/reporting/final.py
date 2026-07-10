# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Final end-of-run report orchestration."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from traceml_ai.core.summaries import SummaryResult
from traceml_ai.loggers.error_log import get_error_logger
from traceml_ai.reporting.config import (
    DEFAULT_SUMMARY_WINDOW_ROWS,
    normalize_summary_window_rows,
)
from traceml_ai.reporting.primary_diagnosis import build_primary_diagnosis
from traceml_ai.reporting.schema import empty_section_payload
from traceml_ai.reporting.sections.base import SummarySection
from traceml_ai.reporting.sections.process import ProcessSummarySection
from traceml_ai.reporting.sections.step_memory import StepMemorySummarySection
from traceml_ai.reporting.sections.step_time import StepTimeSummarySection
from traceml_ai.reporting.sections.system import SystemSummarySection
from traceml_ai.reporting.summaries.summary_formatting import bytes_to_gb
from traceml_ai.reporting.summaries.summary_layout import (
    border,
    row,
    wrap_lines,
)
from traceml_ai.sdk.protocol import (
    get_final_summary_html_path,
    get_final_summary_json_path,
    get_final_summary_txt_path,
    utc_now_iso,
)
from traceml_ai.utils.atomic_io import write_json_atomic, write_text_atomic

SUMMARY_WIDTH = 78
SUMMARY_INNER_TEXT_WIDTH = SUMMARY_WIDTH - 4

_SYSTEM_EVIDENCE_METRICS = (
    ("CPU Util", "cpu_percent"),
    ("GPU Util", "gpu_util_percent"),
    ("GPU Memory", "gpu_mem_bytes"),
    ("GPU Temp", "gpu_temp_c"),
)
_STEP_TIME_EVIDENCE_METRICS = (
    ("Total", "total_step_ms"),
    ("Dataloader", "dataloader_ms"),
    ("Input Wait", "input_wait_ms"),
    ("Step Time", "step_time_ms"),
    ("Compute", "compute_ms"),
    ("Residual", "residual_ms"),
    ("H2D", "h2d_ms"),
)
_STEP_TIME_CPU_COMPAT_METRICS = frozenset(("total_step_ms", "dataloader_ms"))
_SEVERITY_LABELS = {
    "crit": "CRITICAL",
    "critical": "CRITICAL",
    "warn": "WARNING",
    "warning": "WARNING",
    "info": "INFO",
}


def _log_final_report_error(message: str, exc: Exception) -> None:
    """Log final-report failures without raising from cleanup paths."""
    try:
        get_error_logger("FinalReport").exception("[TraceML] %s", message)
    except Exception:
        pass


@dataclass(frozen=True)
class FunctionSummarySection:
    """Summary section backed by a function."""

    name: str
    builder: Callable[..., Dict[str, Any]]

    def build(self, db_path: str) -> SummaryResult:
        """Build one summary section from the SQLite database path."""
        payload = self.builder(db_path, print_to_stdout=False)
        return SummaryResult(
            section=self.name,
            payload=dict(payload or {}),
            text=str((payload or {}).get("card", "")),
        )


def _summary_duration_s(*sections: Dict[str, Any]) -> Optional[float]:
    """Pick the first valid duration from the available sections."""
    for section in sections:
        if not isinstance(section, dict):
            continue
        value = section.get("duration_s")
        if value is None:
            value = section.get("metadata", {}).get("duration_s")
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _safe_int(value: Any, *, allow_zero: bool = True) -> Optional[int]:
    """Return an int when a JSON value is numeric enough for metadata."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed < 0 or (parsed == 0 and not allow_zero):
        return None
    return parsed


def _section_metadata(section: Dict[str, Any]) -> Dict[str, Any]:
    """Return a section metadata mapping when present."""
    metadata = section.get("metadata") if isinstance(section, dict) else None
    return dict(metadata) if isinstance(metadata, dict) else {}


def _section_group_rows(section: Dict[str, Any]) -> Dict[str, Any]:
    """Return section group rows when present."""
    groups = section.get("groups") if isinstance(section, dict) else None
    if not isinstance(groups, dict):
        return {}
    rows = groups.get("rows")
    return dict(rows) if isinstance(rows, dict) else {}


def _run_name_from_manifest(session_root: Optional[str]) -> Optional[str]:
    """Load the public run name from the launcher manifest when available."""
    if not session_root:
        return None

    root = Path(session_root).resolve()
    manifest_path = root / "manifest.json"
    manifest: Dict[str, Any] = {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            manifest = loaded
    except Exception:
        manifest = {}

    run = manifest.get("run")
    run_block = run if isinstance(run, dict) else {}
    candidates = (
        run_block.get("run_name"),
        manifest.get("session_id"),
        root.name,
    )
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return None


def _final_meta_mode(sections: Sequence[Dict[str, Any]]) -> str:
    """Infer run mode from already-built section metadata."""
    saw_single_node = False
    for section in sections:
        mode = str(_section_metadata(section).get("mode") or "")
        if mode == "multi_node":
            return "multi_node"
        if mode == "single_node":
            saw_single_node = True
    return "single_node" if saw_single_node else "no_data"


def _final_meta_world_size(
    sections: Sequence[Dict[str, Any]],
) -> Optional[int]:
    """Infer observed world size from section metadata and rank identities."""
    candidates: List[int] = []
    for section in sections:
        metadata = _section_metadata(section)
        for key in ("global_ranks_seen", "global_ranks_used"):
            value = _safe_int(metadata.get(key), allow_zero=False)
            if value is not None:
                candidates.append(value)

        groups = section.get("groups") if isinstance(section, dict) else None
        group_by = groups.get("by") if isinstance(groups, dict) else None
        rows = _section_group_rows(section)
        if group_by == "global_rank" and rows:
            candidates.append(len(rows))

        for group_row in rows.values():
            if not isinstance(group_row, dict):
                continue
            identity = group_row.get("identity")
            if not isinstance(identity, dict):
                continue
            value = _safe_int(identity.get("world_size"), allow_zero=False)
            if value is not None:
                candidates.append(value)

    return max(candidates) if candidates else None


def _final_meta_nodes_observed(
    system_summary: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
) -> Optional[int]:
    """Infer observed node count from system metadata or row identities."""
    value = _safe_int(
        _section_metadata(system_summary).get("nodes_observed"),
        allow_zero=True,
    )
    if value is not None:
        return value

    node_ranks = set()
    for section in sections:
        for group_row in _section_group_rows(section).values():
            if not isinstance(group_row, dict):
                continue
            identity = group_row.get("identity")
            if not isinstance(identity, dict):
                continue
            node_rank = _safe_int(identity.get("node_rank"), allow_zero=True)
            if node_rank is not None:
                node_ranks.add(node_rank)
    return len(node_ranks) if node_ranks else None


def _final_meta_gpus_observed(
    *,
    system_summary: Dict[str, Any],
    mode: str,
    world_size: Optional[int],
) -> Optional[int]:
    """Infer observed GPU count from system metadata with a safe fallback."""
    value = _safe_int(
        _section_metadata(system_summary).get("gpus_observed"),
        allow_zero=True,
    )
    if value is not None:
        return value
    if mode == "single_node" and world_size is not None:
        return int(world_size)
    return None


def _build_final_meta(
    *,
    session_root: Optional[str],
    system_summary: Dict[str, Any],
    process_summary: Dict[str, Any],
    step_time_summary: Dict[str, Any],
    step_memory_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build run-level identity and observed topology metadata."""
    sections = (
        system_summary,
        process_summary,
        step_time_summary,
        step_memory_summary,
    )
    mode = _final_meta_mode(sections)
    world_size = _final_meta_world_size(sections)
    return {
        "run_name": _run_name_from_manifest(session_root),
        "mode": mode,
        "world_size": world_size,
        "nodes_observed": _final_meta_nodes_observed(
            system_summary,
            sections,
        ),
        "gpus_observed": _final_meta_gpus_observed(
            system_summary=system_summary,
            mode=mode,
            world_size=world_size,
        ),
    }


def _diagnosis(section: Dict[str, Any]) -> Dict[str, Any]:
    """Return a section diagnosis mapping, if present."""
    diagnosis = section.get("diagnosis") if isinstance(section, dict) else None
    return dict(diagnosis) if isinstance(diagnosis, dict) else {}


def _global_block(section: Dict[str, Any], block: str) -> Dict[str, Any]:
    """Return one global rollup block from a section payload."""
    global_summary = (
        section.get("global") if isinstance(section, dict) else None
    )
    if not isinstance(global_summary, dict):
        return {}
    value = global_summary.get(block)
    return dict(value) if isinstance(value, dict) else {}


def _point_value(point: Any) -> Optional[float]:
    """Return a numeric `{value, idx}` point value, if available."""
    if not isinstance(point, dict):
        return None
    value = point.get("value")
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _point_idx(point: Any) -> Optional[Any]:
    """Return a `{value, idx}` point index, if available."""
    if not isinstance(point, dict):
        return None
    return point.get("idx")


def _average_value(section: Dict[str, Any], metric: str) -> Optional[float]:
    """Return one average metric value from a section payload."""
    value = _global_block(section, "average").get(metric)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _point(section: Dict[str, Any], block: str, metric: str) -> Dict[str, Any]:
    """Return one median/worst point from a section payload."""
    point = _global_block(section, block).get(metric)
    return dict(point) if isinstance(point, dict) else {}


def _status_label(value: Any, default: str = "NO DATA") -> str:
    """Return an uppercase status label for compact terminal output."""
    text = str(value or default).replace("_", " ").strip()
    return " ".join(text.upper().split()) or default


def _severity_label(value: Any) -> str:
    """Return a normalized severity label for compact terminal output."""
    text = str(value or "info").strip().lower()
    return _SEVERITY_LABELS.get(text, text.upper() if text else "INFO")


def _format_value(metric: str, value: Optional[float]) -> str:
    """Format final-summary evidence values using compact CLI units."""
    if value is None:
        return "n/a"
    if metric.endswith("_ms"):
        return f"{value:.1f}ms"
    if metric.endswith("_bytes"):
        gb = bytes_to_gb(value)
        return "n/a" if gb is None else f"{gb:.2f}GB"
    if metric.endswith("_percent"):
        return f"{value:.1f}%"
    if metric.endswith("_c"):
        return f"{value:.0f}C"
    return f"{value:.1f}"


def _format_share(value: Optional[float], total: Optional[float]) -> str:
    """Format a share as a percent of a selected denominator."""
    if value is None or total is None or total <= 0.0:
        return "n/a"
    return f"{100.0 * value / total:.1f}%"


def _format_skew(
    metric: str,
    median: Optional[float],
    worst: Optional[float],
) -> str:
    """
    Format worst-vs-median skew for compact evidence tables.

    Percentage-valued utilization rows use percentage-point skew. Temperature
    uses an absolute Celsius skew. Other metrics use relative percent skew
    when the median is positive. The value is intentionally unsigned because
    the ``worst`` point already encodes the bad direction for that metric.
    """
    if median is None or worst is None:
        return "n/a"
    delta = abs(worst - median)
    if metric.endswith("_percent"):
        return f"{delta:.1f}pp"
    if metric.endswith("_c"):
        return f"{delta:.0f}C"
    if median <= 0.0:
        return "n/a"
    return f"{100.0 * delta / median:.1f}%"


def _clip(text: str, width: int) -> str:
    """Clip text to a fixed table cell width without breaking layout."""
    clean = str(text)
    if len(clean) <= width:
        return clean
    if width <= 1:
        return clean[:width]
    return clean[: width - 1] + "."


def _table_line(values: Sequence[Any], widths: Sequence[int]) -> str:
    """Return one fixed-width table line for the compact text summary."""
    cells = [
        f"{_clip(str(value), width):<{width}}"
        for value, width in zip(values, widths)
    ]
    return "  ".join(cells).rstrip()


def _table_rule(widths: Sequence[int]) -> str:
    """Return a separator matching a fixed-width table."""
    return "-" * (sum(widths) + 2 * (len(widths) - 1))


def _append_wrapped_text(lines: List[str], text: str) -> None:
    """Append one potentially long logical line inside the summary border."""
    for wrapped in wrap_lines(text, SUMMARY_INNER_TEXT_WIDTH):
        lines.append(row(wrapped, width=SUMMARY_WIDTH))


def _append_verdict_lines(
    lines: List[str],
    primary_diagnosis: Dict[str, Any],
) -> None:
    """Append the compact run-level verdict, why, and next-action lines."""
    status = _status_label(primary_diagnosis.get("status"))
    severity = _severity_label(primary_diagnosis.get("severity"))
    summary = str(primary_diagnosis.get("summary") or "")
    action = str(primary_diagnosis.get("action") or "")
    _append_wrapped_text(lines, f"TraceML Verdict: {status} / {severity}")
    if summary:
        _append_wrapped_text(lines, f"Why: {summary}")
    if action:
        _append_wrapped_text(lines, f"Next: {action}")


def _append_section_status(
    lines: List[str],
    *,
    system_summary: Dict[str, Any],
    process_summary: Dict[str, Any],
    step_time_summary: Dict[str, Any],
    step_memory_summary: Dict[str, Any],
) -> None:
    """Append a compact status table for all section-local diagnoses."""
    rows = (
        ("Step Time", _diagnosis(step_time_summary)),
        ("System", _diagnosis(system_summary)),
        ("Process", _diagnosis(process_summary)),
        ("Step Memory", _diagnosis(step_memory_summary)),
    )
    widths = (12, 22, 10)
    lines.append(row("Section Status", width=SUMMARY_WIDTH))
    lines.append(
        row(
            _table_line(("Section", "Status", "Severity"), widths),
            width=SUMMARY_WIDTH,
        )
    )
    lines.append(row(_table_rule(widths), width=SUMMARY_WIDTH))
    for label, diagnosis in rows:
        lines.append(
            row(
                _table_line(
                    (
                        label,
                        _status_label(diagnosis.get("status")),
                        _severity_label(diagnosis.get("severity")),
                    ),
                    widths,
                ),
                width=SUMMARY_WIDTH,
            )
        )


def _is_multi_process(step_time_summary: Dict[str, Any]) -> bool:
    """Return whether Step Time observed more than one global rank/process."""
    value = _section_metadata(step_time_summary).get("global_ranks_used")
    parsed = _safe_int(value, allow_zero=True)
    return bool(parsed is not None and parsed > 1)


def _row_identity(section: Dict[str, Any], idx: Any) -> Dict[str, Any]:
    """Return a grouped-row identity for a median/worst point index."""
    if idx is None:
        return {}
    rows = _section_group_rows(section)
    row_data = rows.get(str(idx), {})
    identity = row_data.get("identity") if isinstance(row_data, dict) else None
    return dict(identity) if isinstance(identity, dict) else {}


def _system_scope(system_summary: Dict[str, Any], idx: Any) -> str:
    """Return node-level scope text for System evidence rows."""
    identity = _row_identity(system_summary, idx)
    node_rank = _safe_int(identity.get("node_rank"), allow_zero=True)
    if node_rank is not None:
        return f"node=n{node_rank}"
    parsed = _safe_int(idx, allow_zero=True)
    if parsed is not None:
        return f"node=n{parsed}"
    return "n/a" if idx is None else f"node={idx}"


def _step_scope(step_time_summary: Dict[str, Any], idx: Any) -> str:
    """Return rank/node scope text for Step Time evidence rows."""
    identity = _row_identity(step_time_summary, idx)
    rank = _safe_int(identity.get("global_rank"), allow_zero=True)
    if rank is None:
        rank = _safe_int(idx, allow_zero=True)
    parts = []
    if rank is not None:
        parts.append(f"rank=r{rank}")
    node_rank = _safe_int(identity.get("node_rank"), allow_zero=True)
    if node_rank is not None:
        parts.append(f"node=n{node_rank}")
    if parts:
        return " ".join(parts)
    return "n/a" if idx is None else f"rank={idx}"


def _append_system_evidence_single(
    lines: List[str],
    system_summary: Dict[str, Any],
) -> None:
    """Append average-only System evidence for single-process runs."""
    widths = (16, 16)
    lines.append(row("System Evidence", width=SUMMARY_WIDTH))
    lines.append(
        row(_table_line(("Metric", "Average"), widths), width=SUMMARY_WIDTH)
    )
    lines.append(row(_table_rule(widths), width=SUMMARY_WIDTH))
    for label, metric in _SYSTEM_EVIDENCE_METRICS:
        value = _average_value(system_summary, metric)
        lines.append(
            row(
                _table_line(
                    (label, _format_value(metric, value)),
                    widths,
                ),
                width=SUMMARY_WIDTH,
            )
        )


def _append_system_evidence_multi(
    lines: List[str],
    system_summary: Dict[str, Any],
) -> None:
    """Append median/worst System evidence for multi-process runs."""
    widths = (14, 12, 12, 10, 18)
    lines.append(row("System Evidence", width=SUMMARY_WIDTH))
    lines.append(
        row(
            _table_line(
                ("Metric", "Median", "Worst", "Skew", "Scope"),
                widths,
            ),
            width=SUMMARY_WIDTH,
        )
    )
    lines.append(row(_table_rule(widths), width=SUMMARY_WIDTH))
    for label, metric in _SYSTEM_EVIDENCE_METRICS:
        median = _point_value(_point(system_summary, "median", metric))
        worst_point = _point(system_summary, "worst", metric)
        worst = _point_value(worst_point)
        lines.append(
            row(
                _table_line(
                    (
                        label,
                        _format_value(metric, median),
                        _format_value(metric, worst),
                        _format_skew(metric, median, worst),
                        _system_scope(system_summary, _point_idx(worst_point)),
                    ),
                    widths,
                ),
                width=SUMMARY_WIDTH,
            )
        )


def _append_step_time_evidence_single(
    lines: List[str],
    step_time_summary: Dict[str, Any],
) -> None:
    """Append selected-clock average/share Step Time evidence."""
    widths = (16, 16, 12)
    share_total = _average_value(step_time_summary, "step_time_ms")
    lines.append(row("Step Time Evidence", width=SUMMARY_WIDTH))
    lines.append(
        row(
            _table_line(("Phase", "Average", "Share"), widths),
            width=SUMMARY_WIDTH,
        )
    )
    lines.append(row(_table_rule(widths), width=SUMMARY_WIDTH))
    for label, metric in _STEP_TIME_EVIDENCE_METRICS:
        value = _average_value(step_time_summary, metric)
        if metric in _STEP_TIME_CPU_COMPAT_METRICS:
            # These fields stay CPU-clocked for compatibility; selected-clock
            # phase shares use step_time_ms as the denominator.
            share = "compat"
        elif metric == "step_time_ms":
            share = "100.0%"
        else:
            share = _format_share(value, share_total)
        lines.append(
            row(
                _table_line(
                    (label, _format_value(metric, value), share),
                    widths,
                ),
                width=SUMMARY_WIDTH,
            )
        )


def _append_step_time_evidence_multi(
    lines: List[str],
    step_time_summary: Dict[str, Any],
) -> None:
    """Append median/worst Step Time evidence for multi-process runs."""
    widths = (14, 12, 12, 10, 18)
    lines.append(row("Step Time Evidence", width=SUMMARY_WIDTH))
    lines.append(
        row(
            _table_line(("Phase", "Median", "Worst", "Skew", "Scope"), widths),
            width=SUMMARY_WIDTH,
        )
    )
    lines.append(row(_table_rule(widths), width=SUMMARY_WIDTH))
    for label, metric in _STEP_TIME_EVIDENCE_METRICS:
        median = _point_value(_point(step_time_summary, "median", metric))
        worst_point = _point(step_time_summary, "worst", metric)
        worst = _point_value(worst_point)
        lines.append(
            row(
                _table_line(
                    (
                        label,
                        _format_value(metric, median),
                        _format_value(metric, worst),
                        _format_skew(metric, median, worst),
                        _step_scope(
                            step_time_summary,
                            _point_idx(worst_point),
                        ),
                    ),
                    widths,
                ),
                width=SUMMARY_WIDTH,
            )
        )


def _build_final_summary_text_from_sections(
    *,
    primary_diagnosis: Dict[str, Any],
    system_summary: Dict[str, Any],
    process_summary: Dict[str, Any],
    step_time_summary: Dict[str, Any],
    step_memory_summary: Dict[str, Any],
) -> str:
    """
    Build the compact printed end-of-run summary from per-domain sections.

    Section-local ``card`` strings stay in the JSON payload for detailed
    inspection. The top-level text is intentionally verdict-first and lossy:
    it shows the primary finding plus the minimum evidence needed to orient
    the next debugging step.
    """
    duration_s = _summary_duration_s(
        step_time_summary,
        process_summary,
        system_summary,
    )

    lines: List[str] = [
        border(width=SUMMARY_WIDTH),
        row(
            "TraceML Run Summary"
            + (
                f" | duration {duration_s:.1f}s"
                if duration_s is not None
                else ""
            ),
            width=SUMMARY_WIDTH,
        ),
        border(width=SUMMARY_WIDTH),
        row(width=SUMMARY_WIDTH),
    ]

    _append_verdict_lines(lines, primary_diagnosis)

    lines.extend(
        [
            row(width=SUMMARY_WIDTH),
        ]
    )
    _append_section_status(
        lines,
        system_summary=system_summary,
        process_summary=process_summary,
        step_time_summary=step_time_summary,
        step_memory_summary=step_memory_summary,
    )

    multi_process = _is_multi_process(step_time_summary)

    lines.extend(
        [
            row(width=SUMMARY_WIDTH),
        ]
    )
    if multi_process:
        _append_system_evidence_multi(lines, system_summary)
    else:
        _append_system_evidence_single(lines, system_summary)

    lines.extend(
        [
            row(width=SUMMARY_WIDTH),
        ]
    )
    if multi_process:
        _append_step_time_evidence_multi(lines, step_time_summary)
    else:
        _append_step_time_evidence_single(lines, step_time_summary)

    lines.append(border(width=SUMMARY_WIDTH))
    return "\n".join(lines)


def _fallback_section_result(section: SummarySection) -> SummaryResult:
    """Return a stable section payload when one section fails."""
    name = str(getattr(section, "name", "unknown"))
    index_by = "node_rank" if name == "system" else "global_rank"
    payload = empty_section_payload(section_name=name, index_by=index_by)
    return SummaryResult(
        section=name,
        payload=payload,
        text=str(payload.get("card", "")),
    )


@dataclass(frozen=True)
class FinalReportGenerator:
    """
    Build TraceML's final summary payload from registered sections.

    ``sections`` is ordered because final-summary text and JSON key order are
    intentionally stable for users and compare tooling.
    """

    sections: Sequence[SummarySection]

    def generate(
        self,
        db_path: str,
        *,
        session_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate the structured final summary payload.

        Section failures are isolated and logged. A failed section contributes
        a schema-valid ``NO DATA`` payload so one broken report domain does
        not prevent artifact generation during aggregator shutdown.
        """
        results: Dict[str, SummaryResult] = {}
        for section in self.sections:
            try:
                result = section.build(db_path)
            except Exception as exc:
                _log_final_report_error(
                    "Final summary section failed: "
                    f"{getattr(section, 'name', section)}",
                    exc,
                )
                result = _fallback_section_result(section)
            results[result.section] = result

        system_summary = dict(
            results.get("system", SummaryResult("system")).payload
        )
        process_summary = dict(
            results.get("process", SummaryResult("process")).payload
        )
        step_time_summary = dict(
            results.get("step_time", SummaryResult("step_time")).payload
        )
        step_memory_summary = dict(
            results.get("step_memory", SummaryResult("step_memory")).payload
        )

        primary_diagnosis = build_primary_diagnosis(
            system_summary=system_summary,
            process_summary=process_summary,
            step_time_summary=step_time_summary,
            step_memory_summary=step_memory_summary,
        )

        final_text = _build_final_summary_text_from_sections(
            primary_diagnosis=primary_diagnosis,
            system_summary=system_summary,
            process_summary=process_summary,
            step_time_summary=step_time_summary,
            step_memory_summary=step_memory_summary,
        )

        return {
            "schema_version": 1.6,
            "generated_at": utc_now_iso(),
            "duration_s": _summary_duration_s(
                step_time_summary,
                process_summary,
                system_summary,
            ),
            "meta": _build_final_meta(
                session_root=session_root,
                system_summary=system_summary,
                process_summary=process_summary,
                step_time_summary=step_time_summary,
                step_memory_summary=step_memory_summary,
            ),
            "primary_diagnosis": primary_diagnosis,
            "system": system_summary,
            "process": process_summary,
            "step_time": step_time_summary,
            "step_memory": step_memory_summary,
            "text": final_text,
        }


def build_final_report_generator(
    *,
    summary_window_rows: int = DEFAULT_SUMMARY_WINDOW_ROWS,
) -> FinalReportGenerator:
    """Build a final-report generator with one shared summary window."""
    row_limit = normalize_summary_window_rows(summary_window_rows)
    return FinalReportGenerator(
        sections=(
            SystemSummarySection(max_system_rows=row_limit),
            ProcessSummarySection(max_process_rows=row_limit),
            StepTimeSummarySection(max_rows=row_limit),
            StepMemorySummarySection(window_size=row_limit),
        )
    )


DEFAULT_FINAL_REPORT_GENERATOR = build_final_report_generator()


def build_summary_payload(
    db_path: str,
    *,
    generator: Optional[FinalReportGenerator] = None,
    session_root: Optional[str] = None,
    summary_window_rows: int = DEFAULT_SUMMARY_WINDOW_ROWS,
) -> Dict[str, Any]:
    """
    Build the structured final summary payload for one session database.
    """
    active_generator = generator or build_final_report_generator(
        summary_window_rows=summary_window_rows,
    )
    return active_generator.generate(db_path, session_root=session_root)


def _write_html_artifact(payload: Dict[str, Any], session_root: Path) -> None:
    """
    Render and write the optional HTML report (best-effort).

    Runs after the JSON/TXT artifacts so that a rendering failure can never
    block them. Any failure is logged and reported to stderr, never raised.
    """
    try:
        from traceml_ai.reporting import html as html_report

        html_report.write_html_report(
            payload,
            get_final_summary_html_path(session_root),
        )
    except Exception as exc:  # best-effort: never break the run
        _log_final_report_error("HTML report failed", exc)
        try:
            print(f"[TraceML] HTML report failed: {exc}", file=sys.stderr)
        except Exception:
            pass


def write_summary_artifacts(
    *,
    db_path: str,
    payload: Dict[str, Any],
    session_root: Optional[str] = None,
    write_html: bool = False,
) -> None:
    """
    Write final summary artifacts to disk.

    Artifacts written
    -----------------
    - legacy DB-adjacent artifacts for compatibility
    - canonical session-root artifacts for public API consumers
    - optional self-contained HTML report (``write_html``, best-effort)
    """
    final_text = str(payload.get("text", ""))

    legacy_json_path = Path(str(db_path) + "_summary_card.json").resolve()
    legacy_txt_path = Path(str(db_path) + "_summary_card.txt").resolve()

    write_json_atomic(legacy_json_path, payload)
    write_text_atomic(legacy_txt_path, final_text + "\n")

    if session_root:
        session_root_path = Path(session_root).resolve()
        write_json_atomic(
            get_final_summary_json_path(session_root_path),
            payload,
        )
        write_text_atomic(
            get_final_summary_txt_path(session_root_path),
            final_text + "\n",
        )
        if write_html:
            _write_html_artifact(payload, session_root_path)


def generate_summary(
    db_path: str,
    *,
    session_root: Optional[str] = None,
    print_to_stdout: bool = True,
    summary_window_rows: int = DEFAULT_SUMMARY_WINDOW_ROWS,
    write_html: bool = False,
) -> Dict[str, Any]:
    """
    Generate, write, and optionally print the final end-of-run summary.
    """
    payload = build_summary_payload(
        db_path,
        session_root=session_root,
        summary_window_rows=summary_window_rows,
    )
    write_summary_artifacts(
        db_path=db_path,
        payload=payload,
        session_root=session_root,
        write_html=write_html,
    )

    if print_to_stdout:
        print(payload["text"])

    return payload


__all__ = [
    "DEFAULT_FINAL_REPORT_GENERATOR",
    "FinalReportGenerator",
    "FunctionSummarySection",
    "ProcessSummarySection",
    "StepMemorySummarySection",
    "StepTimeSummarySection",
    "SystemSummarySection",
    "build_final_report_generator",
    "build_summary_payload",
    "generate_summary",
    "write_summary_artifacts",
]
