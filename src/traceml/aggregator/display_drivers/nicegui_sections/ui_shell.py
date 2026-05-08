"""Shared UI helpers for the NiceGUI overview dashboard."""

from __future__ import annotations

from traceml.utils.formatting import fmt_mem_new

CARD_STYLE = """
background: #ffffff;
backdrop-filter: blur(12px);
-webkit-backdrop-filter: blur(12px);
border-radius: 14px;
border: 1px solid rgba(0,0,0,0.08);
box-shadow: 0 4px 12px rgba(0,0,0,0.10);
min-width: 0;
max-width: 100%;
overflow: hidden;
"""

PAGE_GAP_CLASS = "gap-2"
VIEWPORT_STYLE = "height: calc(100vh - 86px); overflow: auto;"
BODY_CLASS = "text-[12px] text-gray-700 leading-tight"


def severity_color(severity: str) -> str:
    """Return a stable color for a normalized severity string."""
    return {
        "crit": "#c62828",
        "warn": "#ef6c00",
        "info": "#2e7d32",
    }.get(str(severity).lower(), "#455a64")


def severity_chip_html(severity: str, *, label: str | None = None) -> str:
    """Render a small rounded severity chip."""
    sev = str(severity).lower()
    text = label or {
        "crit": "CRITICAL",
        "warn": "WARNING",
        "info": "STABLE",
    }.get(sev, "UNKNOWN")
    return (
        f"<span style='background:{severity_color(sev)}; color:#fff; "
        "border-radius:999px; padding:3px 10px; font-size:11px; "
        f"font-weight:700;'>{text}</span>"
    )


def compact_metric_html(label: str, value: str) -> str:
    """Render one compact label/value block for KPI grids."""
    return (
        "<div style='display:flex; flex-direction:column; gap:1px;'>"
        f"<span style='font-size:10px; font-weight:700; color:#d47a00; line-height:1.15;'>{label}</span>"
        f"<span style='font-size:11px; color:#4b5563; line-height:1.2; word-break:break-word;'>{value}</span>"
        "</div>"
    )


def safe_ms(value: float | None) -> str:
    """Format milliseconds for compact dashboard output."""
    try:
        return f"{float(value):.1f} ms"
    except Exception:
        return "-"


def safe_pct(value: float | None) -> str:
    """Format a ratio as a percentage string."""
    try:
        return f"{float(value) * 100.0:.1f}%"
    except Exception:
        return "-"


def safe_mem(value_bytes: float | None) -> str:
    """Format bytes using the shared memory formatter."""
    try:
        return fmt_mem_new(float(value_bytes))
    except Exception:
        return "-"
