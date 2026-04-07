"""
Compact NiceGUI section for unified model diagnostics.

This card keeps a small footprint while exposing the most actionable metadata:
- overall severity
- one compact step-time block
- one compact step-memory block
- short evidence row for each block

The section remains resilient to incomplete payloads by retaining the last
valid payload on transient failures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from nicegui import ui

from .ui_shell import CARD_STYLE, severity_chip_html


def build_model_diagnostics_section(
    *, title: str = "Model Diagnostics"
) -> Dict[str, Any]:
    """Build a fixed-size diagnostics rail card."""
    card = ui.card().classes("w-full h-full p-3")
    card.style(
        CARD_STYLE + "height: 100%; overflow-y: auto; overflow-x: hidden;"
    )

    with card:
        ui.label(title).classes("text-sm font-bold mb-2").style(
            "color:#455a64;"
        )
        overall_html = ui.html("", sanitize=False).classes("mb-3")
        time_html = ui.html("", sanitize=False).classes("mb-2")
        memory_html = ui.html("", sanitize=False)
        empty_hint = ui.html(
            "<div style='text-align:center; padding:10px; color:#888; font-style:italic;'>"
            "Waiting for diagnostics...</div>",
            sanitize=False,
        )

    return {
        "overall_html": overall_html,
        "time_html": time_html,
        "memory_html": memory_html,
        "empty_hint": empty_hint,
        "_last_ok_payload": None,
    }


def update_model_diagnostics_section(
    panel: Dict[str, Any], payload: Any
) -> None:
    """Update diagnostics using the latest valid payload, else keep last good."""
    try:
        items = _extract_items(payload)
        if len(items) < 2:
            payload = panel.get("_last_ok_payload")
            items = _extract_items(payload)

        if len(items) < 2:
            return

        overall = _extract_overall(payload)
        step_time = _find_item(items, "step_time")
        step_memory = _find_item(items, "step_memory")
        if step_time is None or step_memory is None:
            return

        panel["overall_html"].content = severity_chip_html(
            overall, label=f"Overall: {overall.upper()}"
        )
        panel["time_html"].content = _render_item_html("Step Time", step_time)
        panel["memory_html"].content = _render_item_html(
            "Step Memory", step_memory
        )
        panel["empty_hint"].content = ""
        panel["_last_ok_payload"] = payload

    except Exception:
        pass


def _extract_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        items = payload.get("items", [])
        return items if isinstance(items, list) else []
    return []


def _extract_overall(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("overall_severity", "info"))
    return "info"


def _find_item(
    items: List[Dict[str, Any]], source: str
) -> Optional[Dict[str, Any]]:
    for item in items:
        if str(item.get("source")) == source:
            return item
    return None


def _render_item_html(title: str, item: Dict[str, Any]) -> str:
    status = str(item.get("status", "NO DATA"))
    reason = str(item.get("reason", "No diagnosis available.")).strip()
    if len(reason) > 110:
        reason = reason[:107] + "..."

    confidence = _render_confidence(item.get("confidence_label"))
    evidence = _render_evidence(item.get("evidence"))

    return (
        "<div style='display:flex; flex-direction:column; gap:4px; "
        "padding:10px 0; border-top:1px solid #ececec;'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center; gap:8px;'>"
        f"<div style='font-size:13px; font-weight:700; color:#374151;'>{title}</div>"
        f"{confidence}"
        "</div>"
        f"<div style='font-size:11px; font-weight:700; color:#6b7280;'>Status: {status}</div>"
        f"<div style='font-size:12px; color:#4b5563; line-height:1.35;'>{reason}</div>"
        f"{evidence}"
        "</div>"
    )


def _render_confidence(value: Any) -> str:
    label = str(value or "").strip().lower()
    if not label:
        return ""

    color = {
        "high": "#2e7d32",
        "medium": "#ef6c00",
        "low": "#6b7280",
    }.get(label, "#6b7280")

    return (
        f"<span style='font-size:10px; font-weight:700; color:{color}; "
        f"text-transform:uppercase;'>{label}</span>"
    )


def _render_evidence(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return ""

    ordered_keys = [
        "window",
        "worst",
        "gap",
        "wait",
        "dominant",
        "imb",
        "trend",
        "pressure",
    ]
    labels = {
        "window": "window",
        "worst": "worst",
        "gap": "gap",
        "wait": "wait",
        "dominant": "dominant",
        "imb": "imb",
        "trend": "trend",
        "pressure": "pressure",
    }

    parts: List[str] = []
    for key in ordered_keys:
        raw = value.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        parts.append(f"{labels[key]} {text}")

    if not parts:
        return ""

    return (
        "<div style='font-size:11px; color:#6b7280; line-height:1.3;'>"
        + " | ".join(parts)
        + "</div>"
    )
