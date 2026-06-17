"""Model diagnostics rail — overall verdict + per-source severity rows (PR2).

Renderer payload {overall_severity, items:[{source,status,reason,
confidence_label,evidence}]} is unchanged; only the presentation changes.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List

from nicegui import ui

from . import theme

_SOURCE_NAMES = {
    "step_time": "Step time",
    "step_memory": "Step memory",
    "system": "System",
    "process": "Process",
}
# Color buckets come from the engine's own severity (info|warn|crit), already
# present in the payload — never re-parsed from the status text. New diagnosis
# kinds therefore color correctly with no change here (single source of truth).
_NEUTRAL_KINDS = ("NO_DATA", "WARMUP", "NO_GPU")


def _row_sev(item: Dict[str, Any]) -> str:
    """Color bucket for one diagnosis row, from the engine's per-item severity.

    ``info`` splits into neutral grey for 'no verdict yet' states
    (NO_DATA/WARMUP/NO_GPU) and healthy green for a real all-clear diagnosis.
    """
    sev = str(item.get("severity", "")).lower()
    if sev == "crit":
        return "crit"
    if sev == "warn":
        return "warn"
    kind = str(item.get("kind", "")).upper()
    if any(k in kind for k in _NEUTRAL_KINDS):
        return "neutral"
    return "healthy"


def _overall_sev(severity: Any) -> str:
    """Color bucket for the overall pill, from engine overall_severity."""
    s = str(severity).lower()
    if s == "crit":
        return "crit"
    if s == "warn":
        return "warn"
    return "neutral"


def build_model_diagnostics_section(
    *, title: str = "Diagnostics"
) -> Dict[str, Any]:
    card = ui.element("div").classes("glass reveal")
    card.style(
        "padding:18px 20px; width:100%; height:100%; "
        "display:flex; flex-direction:column; overflow-y:auto; overflow-x:hidden;"
    )
    with card:
        with (
            ui.row()
            .classes("w-full items-center")
            .style("margin-bottom:10px; gap:10px;")
        ):
            ui.label(title).classes("ctitle")
            ui.element("div").style("flex:1;")
            overall = ui.label("").classes("sevpill")
        body = ui.html("", sanitize=False).classes("w-full")
        hint = ui.label("Waiting for diagnostics").style(
            "font-family:var(--mono); font-size:11px; color:var(--muted); "
            "font-style:italic; padding-top:6px;"
        )
    return {"overall": overall, "body": body, "hint": hint, "_last": None}


def update_model_diagnostics_section(
    panel: Dict[str, Any], payload: Any
) -> None:
    try:
        items = _items(payload)
        if not items:
            payload = panel.get("_last")
            items = _items(payload)
        if not items:
            return
        panel["_last"] = payload
        panel["hint"].text = ""

        sev = _overall_sev(_overall(payload))
        col = theme.SEV.get(sev, theme.SEV["neutral"])
        panel["overall"].text = str(_overall(payload)).upper()
        panel["overall"].style(
            f"background:{_tint(col)}; color:{col}; border:1px solid {_tint(col, 0.28)};"
        )

        panel["body"].content = "".join(_row_html(it) for it in items)
    except Exception:
        pass


def _items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        it = payload.get("items", [])
        return it if isinstance(it, list) else []
    return []


def _overall(payload: Any) -> str:
    return (
        str(payload.get("overall_severity", "info"))
        if isinstance(payload, dict)
        else "info"
    )


def _tint(hex_color: str, alpha: float = 0.12) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _row_html(item: Dict[str, Any]) -> str:
    source = _SOURCE_NAMES.get(
        str(item.get("source")), str(item.get("source", "")).title()
    )
    status = str(item.get("status", "NO DATA"))
    reason = html.escape(str(item.get("reason", "")).strip())
    if len(reason) > 130:
        reason = reason[:127] + "…"
    sev = _row_sev(item)
    col = theme.SEV.get(sev, theme.SEV["neutral"])
    conf = str(item.get("confidence_label", "") or "").strip()
    conf_html = (
        f"<span style='font-family:var(--mono); font-size:9.5px; font-weight:600; "
        f"text-transform:uppercase; color:{theme.MUTED};'>{html.escape(conf)}</span>"
        if conf
        else ""
    )
    ev = _evidence_html(item.get("evidence"))
    return (
        f"<div class='diagrow'>"
        f"<div class='diagdot' style='background:{col};'></div>"
        f"<div style='flex:1; min-width:0;'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center; gap:8px;'>"
        f"<span style='font-family:var(--sans); font-size:13px; font-weight:600; color:{theme.INK};'>{html.escape(source)}</span>"
        f"{conf_html}</div>"
        f"<div style='font-family:var(--mono); font-size:10px; font-weight:600; letter-spacing:.05em; color:{col}; margin-top:2px;'>{html.escape(status)}</div>"
        f"<div style='font-family:var(--sans); font-size:12px; color:#4b5563; line-height:1.4; margin-top:3px;'>{reason}</div>"
        f"{ev}</div></div>"
    )


def _evidence_html(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    order = [
        "window",
        "worst",
        "gap",
        "wait",
        "dominant",
        "imb",
        "trend",
        "pressure",
    ]
    parts: List[str] = []
    for key in order:
        raw = value.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        parts.append(
            f"<span style='font-family:var(--mono); font-size:10px; color:{theme.MUTED}; "
            f"background:rgba(17,24,39,0.04); padding:1px 6px; border-radius:6px;'>"
            f"{html.escape(key)} {html.escape(str(raw).strip())}</span>"
        )
    if not parts:
        return ""
    return f"<div style='display:flex; flex-wrap:wrap; gap:5px; margin-top:6px;'>{''.join(parts)}</div>"
