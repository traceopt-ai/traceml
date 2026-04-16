"""
Human-readable rendering for TraceML run comparison.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from traceml.aggregator.summaries.summary_layout import border, row, wrap_lines
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

    # suppress pure zero-to-zero noise
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


def _format_duration(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}s"


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
    Render only the step phases that changed materially.
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


def build_compare_text(payload: Dict[str, Any]) -> str:
    """
    Render a compact terminal-friendly compare report.
    """
    lhs = payload.get("lhs", {})
    rhs = payload.get("rhs", {})
    overview = payload.get("overview", {})
    step_time = payload.get("step_time", {})
    step_memory = payload.get("step_memory", {})
    process = payload.get("process", {})
    system = payload.get("system", {})

    duration = overview.get("duration_s", {})
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

    _append_wrapped(lines, "Overview")
    _append_wrapped(lines, f"- Headline: {overview.get('headline', 'n/a')}")
    _append_wrapped(
        lines,
        "- Duration: "
        f"{_format_from_delta_block(duration, formatter=_format_duration)} | "
        f"{_format_numeric_delta(duration.get('delta'), duration.get('pct_change'), unit='s')}",
    )
    _append_wrapped(
        lines,
        "- Diagnoses: "
        f"{step_status.get('lhs', 'n/a')} -> {step_status.get('rhs', 'n/a')} | "
        f"{mem_status.get('lhs', 'n/a')} -> {mem_status.get('rhs', 'n/a')}",
    )

    lines.append(row(width=_COMPARE_WIDTH))
    _append_wrapped(lines, "Step Time")
    _append_wrapped(
        lines,
        "- Step avg: "
        f"{_format_from_delta_block(step_avg, formatter=_format_ms)} | "
        f"{_format_numeric_delta(step_avg.get('delta'), step_avg.get('pct_change'), unit='ms')}",
    )

    if _has_numeric_signal(wait_share, delta_eps=0.05):
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

    memory_lines: List[str] = []
    if _has_numeric_signal(worst_peak, delta_eps=1.0):
        memory_lines.append(
            "- Worst peak: "
            f"{_format_from_delta_block(worst_peak, formatter=fmt_mem_new)} | "
            f"{_format_bytes_delta(_as_float(worst_peak.get('delta')))}"
        )
    if _has_numeric_signal(mem_skew, delta_eps=0.05):
        memory_lines.append(
            "- Skew: "
            f"{_format_from_delta_block(mem_skew, formatter=_format_pct)} | "
            f"{_format_pp_delta(_as_float(mem_skew.get('delta')))}"
        )
    if _has_numeric_signal(mem_trend, delta_eps=1.0):
        memory_lines.append(
            "- Trend (worst): "
            f"{_format_from_delta_block(mem_trend, formatter=fmt_mem_new)} | "
            f"{_format_bytes_delta(_as_float(mem_trend.get('delta')))}"
        )

    if memory_lines:
        lines.append(row(width=_COMPARE_WIDTH))
        _append_wrapped(lines, "Step Memory")
        for text in memory_lines:
            _append_wrapped(lines, text)

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

    if process_lines:
        lines.append(row(width=_COMPARE_WIDTH))
        _append_wrapped(lines, "Process")
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

    if system_lines:
        lines.append(row(width=_COMPARE_WIDTH))
        _append_wrapped(lines, "System")
        for text in system_lines:
            _append_wrapped(lines, text)

    lines.append(border(width=_COMPARE_WIDTH))
    return "\n".join(lines)
