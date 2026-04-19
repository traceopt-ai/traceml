"""
Human-readable rendering for TraceML run comparison.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from traceml.reporting.summaries.summary_layout import border, row, wrap_lines
from traceml.utils.formatting import fmt_mem_new

_COMPARE_WIDTH = 78
_COMPARE_INNER_WIDTH = _COMPARE_WIDTH - 4


def _append_wrapped(lines: List[str], text: str) -> None:
    for wrapped in wrap_lines(text, _COMPARE_INNER_WIDTH):
        lines.append(row(wrapped, width=_COMPARE_WIDTH))


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _as_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text or None


def _is_effectively_zero(value: Optional[float], eps: float = 1e-9) -> bool:
    return value is not None and abs(value) <= eps


def _has_numeric_signal(
    block: Dict[str, Any],
    *,
    delta_eps: float = 1e-9,
) -> bool:
    lhs = _as_float(block.get("lhs"))
    rhs = _as_float(block.get("rhs"))
    delta = _as_float(block.get("delta"))

    if lhs is None and rhs is None:
        return False

    if delta is None:
        return lhs is not None or rhs is not None

    if abs(delta) > delta_eps:
        return True

    if _is_effectively_zero(lhs, delta_eps) and _is_effectively_zero(
        rhs, delta_eps
    ):
        return False

    return False


def _has_text_signal(block: Dict[str, Any]) -> bool:
    lhs = _as_str(block.get("lhs"))
    rhs = _as_str(block.get("rhs"))
    changed = bool(block.get("changed"))

    if changed:
        return True
    if lhs is None and rhs is None:
        return False
    return False


def _format_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}ms"


def _format_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _format_gb(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f} GB"


def _format_pp_delta(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.1f}pp"


def _format_numeric_delta(
    delta: Optional[float],
    pct_change: Optional[float],
    *,
    unit: str = "",
    ndigits: int = 1,
) -> str:
    if delta is None:
        return "n/a"

    sign = "+" if delta >= 0 else "-"
    if pct_change is None:
        return f"{sign}{abs(delta):.{ndigits}f}{unit}"

    return (
        f"{sign}{abs(delta):.{ndigits}f}{unit} "
        f"({sign}{abs(pct_change):.1f}%)"
    )


def _format_bytes_delta(delta: Optional[float]) -> str:
    if delta is None:
        return "n/a"
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{fmt_mem_new(abs(delta))}"


def _format_compare_presence(
    block: Dict[str, Any],
    *,
    formatter,
) -> Optional[str]:
    """
    Render a compare block when one or both sides are missing.

    Returns None when both values are present, so the normal delta formatter
    should be used instead.
    """
    lhs = _as_float(block.get("lhs"))
    rhs = _as_float(block.get("rhs"))

    if lhs is not None and rhs is not None:
        return None
    if lhs is None and rhs is None:
        return "unavailable in both runs"
    if lhs is None:
        return f"unavailable in A; B = {formatter(rhs)}"
    return f"A = {formatter(lhs)}; unavailable in B"


def _format_from_delta_block(
    block: Dict[str, Any],
    *,
    formatter,
) -> str:
    lhs = formatter(_as_float(block.get("lhs")))
    rhs = formatter(_as_float(block.get("rhs")))
    return f"{lhs} -> {rhs}"


def _render_split_shift(split_pct: Dict[str, Any]) -> Optional[str]:
    """
    Render only the step phases that changed materially enough to mention.
    """
    phase_map = {
        "dataloader": "DL",
        "forward": "FWD",
        "backward": "BWD",
        "optimizer": "OPT",
    }

    parts: List[str] = []
    for phase, short in phase_map.items():
        block = split_pct.get(phase, {})
        delta = _as_float(block.get("delta"))
        if delta is None or abs(delta) < 0.05:
            continue
        parts.append(f"{short} {_format_pp_delta(delta)}")

    if not parts:
        return None

    return " | ".join(parts)


def _title_case(value: Optional[str]) -> str:
    """
    Format a small label for human-readable output.
    """
    if not value:
        return "n/a"
    return str(value).replace("_", " ").title()


def build_compare_text(payload: Dict[str, Any]) -> str:
    """
    Render a compact terminal-friendly compare report.

    The report is intentionally split into:
    1. a short conservative verdict
    2. real top changes only
    3. compact evidence sections
    """
    lhs = payload.get("lhs", {})
    rhs = payload.get("rhs", {})
    verdict = payload.get("verdict", {})
    step_time = payload.get("step_time", {})
    step_memory = payload.get("step_memory", {})
    process = payload.get("process", {})
    system = payload.get("system", {})

    outcome = _as_str(verdict.get("outcome"))

    step_avg = step_time.get("step_avg_ms", {})
    wait_share = step_time.get("wait_share_pct", {})
    dominant_phase = step_time.get("dominant_phase", {})
    step_status = step_time.get("status", {})
    mem_status = step_memory.get("status", {})
    worst_peak = step_memory.get("worst_peak_bytes", {})
    mem_skew = step_memory.get("skew_pct", {})
    mem_trend = step_memory.get("trend_worst_delta_bytes", {})

    process_cpu = process.get("cpu_avg_percent", {})
    process_rss = process.get("ram_peak_gb", {})
    process_takeaway = process.get("takeaway", {})

    system_cpu = system.get("cpu_avg_percent", {})
    system_ram = system.get("ram_peak_gb", {})
    gpu_change = system.get("gpu_available", {})
    gpu_count = system.get("gpu_count", {})

    step_presented_rhs = (
        step_time.get("presented", {}).get("rhs")
        if isinstance(step_time.get("presented"), dict)
        else None
    )
    mem_presented_rhs = (
        step_memory.get("presented", {}).get("rhs")
        if isinstance(step_memory.get("presented"), dict)
        else None
    )

    lines: List[str] = [
        border(width=_COMPARE_WIDTH),
        row("TraceML Compare", width=_COMPARE_WIDTH),
        border(width=_COMPARE_WIDTH),
        row(width=_COMPARE_WIDTH),
    ]

    _append_wrapped(lines, f"- A: {lhs.get('label', 'lhs')}")
    _append_wrapped(lines, f"- B: {rhs.get('label', 'rhs')}")
    _append_wrapped(lines, "- Format: A -> B | delta = B - A")

    lines.append(row(width=_COMPARE_WIDTH))
    _append_wrapped(lines, "Verdict")
    _append_wrapped(lines, f"- Result: {verdict.get('summary', 'n/a')}")
    _append_wrapped(
        lines,
        f"- Severity: {_title_case(_as_str(verdict.get('severity')))}",
    )

    why = _as_str(verdict.get("why"))
    if why:
        _append_wrapped(lines, f"- Why: {why}")

    largest_shift = _as_str(verdict.get("largest_shift"))
    if largest_shift:
        _append_wrapped(lines, f"- Largest shift: {largest_shift}")

    _append_wrapped(lines, f"- Action: {verdict.get('action', 'n/a')}")

    top_changes = verdict.get("top_changes", [])
    lines.append(row(width=_COMPARE_WIDTH))
    _append_wrapped(lines, "Top Changes")
    if isinstance(top_changes, list) and top_changes:
        for item in top_changes:
            if not isinstance(item, dict):
                continue
            summary = _as_str(item.get("summary"))
            significance = _as_str(item.get("significance"))
            if not summary:
                continue
            if significance:
                _append_wrapped(lines, f"- {summary} ({significance})")
            else:
                _append_wrapped(lines, f"- {summary}")
    else:
        _append_wrapped(
            lines, "- No material or clearly interpretable changes detected."
        )

    lines.append(row(width=_COMPARE_WIDTH))
    _append_wrapped(lines, "Step Time")
    _append_wrapped(
        lines,
        f"- Diagnosis: {step_status.get('lhs', 'n/a')} -> {step_status.get('rhs', 'n/a')}",
    )
    step_avg_presence = _format_compare_presence(
        step_avg, formatter=_format_ms
    )
    if step_avg_presence is not None:
        _append_wrapped(lines, f"- Step avg: {step_avg_presence}")
    else:
        _append_wrapped(
            lines,
            "- Step avg: "
            f"{_format_from_delta_block(step_avg, formatter=_format_ms)} | "
            f"{_format_numeric_delta(step_avg.get('delta'), step_avg.get('pct_change'), unit='ms')}",
        )

    if isinstance(step_presented_rhs, dict) and step_status.get("changed"):
        rhs_reason = _as_str(step_presented_rhs.get("reason"))
        rhs_action = _as_str(step_presented_rhs.get("action"))
        if rhs_reason:
            _append_wrapped(lines, f"- Why B: {rhs_reason}")
        if rhs_action and outcome == "regression":
            _append_wrapped(lines, f"- Next: {rhs_action}")

    wait_presence = _format_compare_presence(wait_share, formatter=_format_pct)
    if wait_presence is not None:
        _append_wrapped(lines, f"- Wait share: {wait_presence}")
    elif _has_numeric_signal(wait_share, delta_eps=0.05):
        _append_wrapped(
            lines,
            "- Wait share: "
            f"{_format_from_delta_block(wait_share, formatter=_format_pct)} | "
            f"{_format_pp_delta(_as_float(wait_share.get('delta')))}",
        )

    if _has_text_signal(dominant_phase):
        _append_wrapped(
            lines,
            f"- Dominant: {dominant_phase.get('lhs', 'n/a')} -> {dominant_phase.get('rhs', 'n/a')}",
        )

    split_line = _render_split_shift(step_time.get("split_pct", {}))
    if split_line:
        _append_wrapped(lines, f"- Split shift: {split_line}")

    show_memory_section = bool(mem_status.get("changed")) or any(
        [
            _has_numeric_signal(worst_peak, delta_eps=1.0),
            _has_numeric_signal(mem_skew, delta_eps=0.05),
            _has_numeric_signal(mem_trend, delta_eps=1.0),
        ]
    )

    if show_memory_section:
        lines.append(row(width=_COMPARE_WIDTH))
        _append_wrapped(lines, "Step Memory")
        _append_wrapped(
            lines,
            f"- Diagnosis: {mem_status.get('lhs', 'n/a')} -> {mem_status.get('rhs', 'n/a')}",
        )

        if isinstance(mem_presented_rhs, dict) and mem_status.get("changed"):
            rhs_reason = _as_str(mem_presented_rhs.get("reason"))
            rhs_action = _as_str(mem_presented_rhs.get("action"))
            if rhs_reason:
                _append_wrapped(lines, f"- Why B: {rhs_reason}")
            if rhs_action and outcome == "regression":
                _append_wrapped(lines, f"- Next: {rhs_action}")

        worst_peak_presence = _format_compare_presence(
            worst_peak, formatter=fmt_mem_new
        )
        if worst_peak_presence is not None:
            _append_wrapped(lines, f"- Worst peak: {worst_peak_presence}")
        elif _has_numeric_signal(worst_peak, delta_eps=1.0):
            _append_wrapped(
                lines,
                "- Worst peak: "
                f"{_format_from_delta_block(worst_peak, formatter=fmt_mem_new)} | "
                f"{_format_bytes_delta(_as_float(worst_peak.get('delta')))}",
            )

        skew_presence = _format_compare_presence(
            mem_skew, formatter=_format_pct
        )
        if skew_presence is not None:
            _append_wrapped(lines, f"- Skew: {skew_presence}")
        elif _has_numeric_signal(mem_skew, delta_eps=0.05):
            _append_wrapped(
                lines,
                "- Skew: "
                f"{_format_from_delta_block(mem_skew, formatter=_format_pct)} | "
                f"{_format_pp_delta(_as_float(mem_skew.get('delta')))}",
            )

        if _has_numeric_signal(mem_trend, delta_eps=1.0):
            _append_wrapped(
                lines,
                "- Trend (worst): "
                f"{_format_from_delta_block(mem_trend, formatter=fmt_mem_new)} | "
                f"{_format_bytes_delta(_as_float(mem_trend.get('delta')))}",
            )

    show_context = outcome in {"regression", "improvement"}

    process_lines: List[str] = []
    if _has_numeric_signal(process_cpu, delta_eps=0.05):
        process_lines.append(
            "- CPU avg: "
            f"{_format_from_delta_block(process_cpu, formatter=_format_pct)} | "
            f"{_format_pp_delta(_as_float(process_cpu.get('delta')))}"
        )
    if _has_numeric_signal(process_rss, delta_eps=0.01):
        process_lines.append(
            "- RSS peak: "
            f"{_format_from_delta_block(process_rss, formatter=_format_gb)} | "
            f"{_format_numeric_delta(process_rss.get('delta'), process_rss.get('pct_change'), unit=' GB')}"
        )
    if _has_text_signal(process_takeaway):
        process_lines.append(
            f"- Takeaway: {process_takeaway.get('lhs', 'n/a')} -> {process_takeaway.get('rhs', 'n/a')}"
        )

    if show_context and process_lines:
        lines.append(row(width=_COMPARE_WIDTH))
        _append_wrapped(lines, "Process Context")
        for text in process_lines:
            _append_wrapped(lines, text)

    system_lines: List[str] = []
    if _has_numeric_signal(system_cpu, delta_eps=0.05):
        system_lines.append(
            "- CPU avg: "
            f"{_format_from_delta_block(system_cpu, formatter=_format_pct)} | "
            f"{_format_pp_delta(_as_float(system_cpu.get('delta')))}"
        )
    if _has_numeric_signal(system_ram, delta_eps=0.01):
        system_lines.append(
            "- RAM peak: "
            f"{_format_from_delta_block(system_ram, formatter=_format_gb)} | "
            f"{_format_numeric_delta(system_ram.get('delta'), system_ram.get('pct_change'), unit=' GB')}"
        )
    if _has_text_signal(gpu_change):
        system_lines.append(
            f"- GPU available: {gpu_change.get('lhs', 'n/a')} -> {gpu_change.get('rhs', 'n/a')}"
        )
    if _has_numeric_signal(gpu_count, delta_eps=0.0):
        system_lines.append(
            "- GPU count: "
            f"{gpu_count.get('lhs', 'n/a')} -> {gpu_count.get('rhs', 'n/a')}"
        )

    if (show_context or _has_text_signal(gpu_change)) and system_lines:
        lines.append(row(width=_COMPARE_WIDTH))
        _append_wrapped(lines, "System Context")
        for text in system_lines:
            _append_wrapped(lines, text)

    lines.append(border(width=_COMPARE_WIDTH))
    return "\n".join(lines)
