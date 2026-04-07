"""
Compact System Metrics dashboard section.

This card is optimized for overview usage:
- compact chart
- dense KPI grid
- stable dimensions before and after data arrives

It is presentation-only and preserves previous visible values on transient gaps.
"""

from __future__ import annotations

from typing import Any, Dict, List

import plotly.graph_objects as go
from nicegui import ui

from traceml.utils.formatting import fmt_mem_new

from .ui_shell import CARD_STYLE, compact_metric_html


def build_system_section() -> Dict[str, Any]:
    """Build a compact System Metrics card."""
    card = ui.card().classes("w-full h-full p-3")
    card.style(
        CARD_STYLE + "height: 100%; overflow-y: auto; overflow-x: hidden;"
    )

    with card:
        with ui.row().classes("w-full items-center justify-between mb-1"):
            ui.label("System Metrics").classes("text-sm font-bold").style(
                "color:#d47a00;"
            )
            window_text = ui.html("window: -", sanitize=False).classes(
                "text-[11px] text-gray-500"
            )

        graph = _build_graph()
        kpis = ui.html("", sanitize=False).classes("mt-2")
        status_text = ui.html("", sanitize=False).classes(
            "text-[11px] text-gray-500 mt-2"
        )

    return {
        "window_text": window_text,
        "graph": graph,
        "kpis": kpis,
        "status_text": status_text,
        "_last_good_payload": None,
        "_last_window_text": None,
        "_last_kpis": None,
    }


def update_system_section(
    panel: Dict[str, Any], payload: Dict[str, Any] | None
) -> None:
    """Update the System Metrics card while preserving last good values."""
    try:
        if _usable_payload(payload):
            panel["_last_good_payload"] = payload
        else:
            payload = panel.get("_last_good_payload")

        if not payload:
            return

        window_len = int(payload.get("window_len", 0) or 0)
        roll = payload.get("rollups") or {}
        series = payload.get("series") or {}

        window_text = (
            f"window: last {window_len} samples"
            if window_len > 0
            else "window: -"
        )
        if panel.get("_last_window_text") != window_text:
            panel["window_text"].content = window_text
            panel["_last_window_text"] = window_text

        panel["status_text"].content = str(roll.get("status") or "")
        _update_graph(panel, series)

        kpis_html = _render_kpis(roll)
        if panel.get("_last_kpis") != kpis_html:
            panel["kpis"].content = kpis_html
            panel["_last_kpis"] = kpis_html

    except Exception:
        pass


def _build_graph():
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="lines", line=dict(color="#4caf50"))
    )
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", line=dict(color="#ff9800"), yaxis="y2"
        )
    )
    fig.update_layout(
        height=104,
        margin=dict(l=8, r=8, t=4, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        showlegend=False,
        xaxis=dict(
            type="date",
            showgrid=False,
            title="Time",
            tickformat="%H:%M:%S",
            hoverformat="%H:%M:%S",
        ),
        yaxis=dict(
            range=[0, 100],
            title=dict(text="CPU (%)", font=dict(color="#4caf50")),
            tickfont=dict(color="#4caf50"),
        ),
        yaxis2=dict(
            range=[0, 100],
            overlaying="y",
            side="right",
            title=dict(text="GPU (%)", font=dict(color="#ff9800")),
            tickfont=dict(color="#ff9800"),
        ),
    )
    return ui.plotly(fig).classes("w-full")


def _usable_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    series = payload.get("series") or {}
    return bool(
        isinstance(series.get("x_time"), list) and series.get("x_time")
    )


def _update_graph(panel: Dict[str, Any], series: Dict[str, Any]) -> None:
    try:
        x = _to_str_list(series.get("x_time"))
        cpu = _to_float_list(series.get("cpu"))
        gpu = _to_float_list(series.get("gpu_avg"))

        fig = panel["graph"].figure

        n = min(len(x), len(cpu))
        fig.data[0].x = x[:n]
        fig.data[0].y = cpu[:n]

        if gpu:
            m = min(len(x), len(gpu))
            fig.data[1].x = x[:m]
            fig.data[1].y = gpu[:m]
        else:
            fig.data[1].x = []
            fig.data[1].y = []

        panel["graph"].update()
    except Exception:
        pass


def _render_kpis(roll: Dict[str, Any]) -> str:
    cpu = roll.get("cpu") or {}
    ram = roll.get("ram") or {}

    items = [
        compact_metric_html(
            "CPU now/p50/p95",
            f"{_num(cpu, 'now'):.0f}% / {_num(cpu, 'p50'):.0f}% / {_num(cpu, 'p95'):.0f}%",
        ),
        compact_metric_html(
            "RAM now/p95/total",
            f"{fmt_mem_new(_num(ram, 'now'))} / {fmt_mem_new(_num(ram, 'p95'))} / {fmt_mem_new(_num(ram, 'total'))}",
        ),
    ]

    if not bool(roll.get("gpu_available", False)):
        items.extend(
            [
                compact_metric_html("GPU Util", "N/A"),
                compact_metric_html("GPU Skew", "N/A"),
                compact_metric_html("GPU Mem", "N/A"),
                compact_metric_html("GPU Temp", "N/A"),
            ]
        )
    else:
        gpu = roll.get("gpu_util") or {}
        delta = roll.get("gpu_delta") or {}
        mem = roll.get("gpu_mem") or {}
        temp = roll.get("temp") or {}
        items.extend(
            [
                compact_metric_html(
                    "GPU Util",
                    f"{_num(gpu, 'now'):.0f}% / {_num(gpu, 'p95'):.0f}%",
                ),
                compact_metric_html(
                    "GPU Skew",
                    f"{_num(delta, 'now'):.0f}% / {_num(delta, 'p95'):.0f}%",
                ),
                compact_metric_html(
                    "GPU Mem",
                    f"{fmt_mem_new(_num(mem, 'now'))} / {fmt_mem_new(_num(mem, 'p95'))}",
                ),
                compact_metric_html(
                    "GPU Temp",
                    f"{_num(temp, 'now'):.0f}C / {_num(temp, 'p95'):.0f}C",
                ),
            ]
        )

    return (
        "<div style='display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); "
        "gap:6px; padding-top:6px; border-top:1px solid #ececec;'>"
        + "".join(items)
        + "</div>"
    )


def _num(mapping: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(mapping.get(key, default))
    except Exception:
        return default


def _to_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        try:
            s = str(item).strip()
            if s:
                out.append(s)
        except Exception:
            continue
    return out


def _to_float_list(value: Any) -> List[float]:
    if not isinstance(value, list):
        return []
    out: List[float] = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            continue
    return out
