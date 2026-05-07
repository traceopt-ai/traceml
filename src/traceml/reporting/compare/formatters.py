"""Text formatter for TraceML compare payloads."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from traceml.core.rendering import Formatter
from traceml.reporting.summaries.summary_layout import border, row, wrap_lines
from traceml.utils.formatting import fmt_mem_new

COMPARE_WIDTH = 88
COMPARE_INNER_WIDTH = COMPARE_WIDTH - 4


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _format_value(value: Any, unit: Optional[str]) -> str:
    if value is None:
        return "n/a"
    if unit == "ms":
        return f"{float(value):.1f} ms"
    if unit == "percent":
        return f"{float(value):.1f}%"
    if unit == "gb":
        return f"{float(value):.1f} GB"
    if unit == "bytes":
        return fmt_mem_new(float(value))
    return str(value)


def _format_delta(metric: Dict[str, Any]) -> str:
    lhs = metric.get("lhs")
    rhs = metric.get("rhs")
    if isinstance(lhs, str) or isinstance(rhs, str):
        return "changed" if lhs != rhs else "same"

    delta = _as_float(metric.get("delta"))
    if delta is None:
        return "n/a"

    unit = metric.get("unit")
    delta_unit = metric.get("delta_unit")
    pct = _as_float(metric.get("pct_change"))

    if unit == "bytes":
        sign = "+" if delta > 0 else ""
        base = f"{sign}{fmt_mem_new(delta)}"
    elif unit == "ms":
        base = f"{delta:+.1f} ms"
    elif delta_unit == "percentage_point":
        base = f"{delta:+.1f} pp"
    elif unit == "percent":
        base = f"{delta:+.1f}%"
    elif unit == "gb":
        base = f"{delta:+.1f} GB"
    else:
        base = f"{delta:+.1f}"

    if pct is not None and unit not in {"percent"}:
        return f"{base} ({pct:+.1f}%)"
    return base


def _diagnosis_row(
    label: str, section: Dict[str, Any]
) -> tuple[str, str, str, str]:
    diagnosis = section.get("diagnosis", {})
    lhs = diagnosis.get("lhs") or "n/a"
    rhs = diagnosis.get("rhs") or "n/a"
    return (
        label,
        str(lhs),
        str(rhs),
        "changed" if lhs != rhs else "same",
    )


def _metric_row(metric: Dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(metric.get("label") or "Metric"),
        _format_value(metric.get("lhs"), metric.get("unit")),
        _format_value(metric.get("rhs"), metric.get("unit")),
        _format_delta(metric),
    )


def _has_signal(metric: Dict[str, Any]) -> bool:
    return metric.get("lhs") is not None or metric.get("rhs") is not None


def _rows_for_section(
    section_name: str,
    section: Dict[str, Any],
) -> list[tuple[str, str, str, str]]:
    metric_order = {
        "step_time": (
            "step_avg_ms",
            "compute_ms",
            "wait_ms",
            "input_ms",
        ),
        "step_memory": (
            "peak_reserved_bytes",
            "memory_skew_pct",
        ),
        "process": (
            "cpu_avg_percent",
            "rss_peak_gb",
        ),
        "system": (
            "cpu_avg_percent",
            "ram_peak_gb",
            "gpu_util_avg_percent",
            "gpu_memory_peak_percent",
        ),
    }
    diagnosis_labels = {
        "step_time": "Step time diagnosis",
        "step_memory": "Step memory diagnosis",
        "process": "Process diagnosis",
        "system": "System diagnosis",
    }
    rows = [_diagnosis_row(diagnosis_labels[section_name], section)]
    metrics = section.get("metrics", {})
    for key in metric_order[section_name]:
        metric = metrics.get(key)
        if isinstance(metric, dict) and _has_signal(metric):
            rows.append(_metric_row(metric))
    return rows


def _section_title(section_name: str) -> str:
    return {
        "step_time": "Step Time",
        "step_memory": "Step Memory",
        "process": "Process",
        "system": "System",
    }[section_name]


def _format_table_rows(rows: Iterable[tuple[str, str, str, str]]) -> list[str]:
    rows = list(rows)
    widths = [30, 16, 16, 17]
    out = [
        f"{'Metric':<{widths[0]}} "
        f"{'A':<{widths[1]}} "
        f"{'B':<{widths[2]}} "
        f"{'Delta':<{widths[3]}}".rstrip()
    ]
    for metric, lhs, rhs, delta in rows:
        out.append(
            f"{metric:<{widths[0]}} "
            f"{lhs:<{widths[1]}} "
            f"{rhs:<{widths[2]}} "
            f"{delta:<{widths[3]}}".rstrip()
        )
    return out


def _append_card_line(lines: List[str], text: str = "") -> None:
    lines.append(row(text, width=COMPARE_WIDTH))


def _append_wrapped_card_line(lines: List[str], text: str) -> None:
    for wrapped in wrap_lines(text, COMPARE_INNER_WIDTH):
        _append_card_line(lines, wrapped)


class CompareTextFormatter(Formatter[Dict[str, Any], str]):
    """Render compare JSON as a compact table."""

    name = "compare_text"

    def format(self, payload: Dict[str, Any]) -> str:
        lhs = payload.get("lhs", {})
        rhs = payload.get("rhs", {})
        verdict = payload.get("verdict", {})
        sections = payload.get("sections", {})

        lines: List[str] = [
            border(width=COMPARE_WIDTH),
            row("TraceML Compare", width=COMPARE_WIDTH),
            border(width=COMPARE_WIDTH),
            row(width=COMPARE_WIDTH),
        ]

        _append_wrapped_card_line(lines, f"A: {lhs.get('label', 'A')}")
        _append_wrapped_card_line(lines, f"B: {rhs.get('label', 'B')}")
        _append_card_line(lines, "Delta: B - A")
        _append_card_line(lines)
        _append_wrapped_card_line(
            lines,
            f"Verdict: {verdict.get('status', 'INCONCLUSIVE')}",
        )
        _append_wrapped_card_line(lines, f"Why: {verdict.get('why', 'n/a')}")

        for section_name in (
            "step_time",
            "step_memory",
            "process",
            "system",
        ):
            section = sections.get(section_name)
            if isinstance(section, dict):
                _append_card_line(lines)
                _append_card_line(lines, _section_title(section_name))
                for text in _format_table_rows(
                    _rows_for_section(section_name, section)
                ):
                    _append_wrapped_card_line(lines, text)

        notes = self._notes(payload)
        if notes:
            _append_card_line(lines)
            _append_card_line(lines, "Notes")
            for note in notes:
                _append_wrapped_card_line(lines, f"- {note}")

        lines.append(border(width=COMPARE_WIDTH))

        return "\n".join(lines)

    def _notes(self, payload: Dict[str, Any]) -> list[str]:
        verdict = payload.get("verdict", {})
        comparability = verdict.get("comparability", {})
        overall = comparability.get("overall", {})
        state = overall.get("state")
        if state in {"partial", "insufficient"}:
            return [
                overall.get("reason")
                or "Some primary compare signals are missing."
            ]
        return []


__all__ = ["CompareTextFormatter"]
