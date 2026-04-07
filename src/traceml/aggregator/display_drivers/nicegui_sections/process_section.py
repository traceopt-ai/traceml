"""
Compact Process Metrics dashboard section.

This card mirrors the System card visually so the overview feels cohesive:
- compact chart
- dense KPI grid
- fixed dimensions from first render
"""

from __future__ import annotations

from typing import Any, Dict, List

import plotly.graph_objects as go
from nicegui import ui

from traceml.utils.formatting import fmt_mem_new

from .ui_shell import CARD_STYLE, compact_metric_html


def build_process_section() -> Dict[str, Any]:
    """Build a compact Process Metrics card."""
    card = ui.card().classes("w-full h-full p-3")
    card.style(
        CARD_STYLE + "height: 100%; overflow-y: auto; overflow-x: hidden;"
    )

    with card:
        with ui.row().classes("w-full items-center justify-between mb-1"):
            ui.label("Process Metrics").classes("text-sm font-bold").style(
                "color:#d47a00;"
            )
            window_text = ui.html("window: -", sanitize=False).classes(
                "text-[11px] text-gray-500"
            )

        graph, fig = _build_graph()
        kpis = ui.html("", sanitize=False).classes("mt-2")

    return {
        "window_text": window_text,
        "graph": graph,
        "kpis": kpis,
        "_fig": fig,
        "_last_ok_data": None,
        "_last_ok_window": None,
        "_last_window_text": None,
        "_last_kpis": None,
    }


def update_process_section(
    panel: Dict[str, Any], data: Dict[str, Any] | None, window_n: int = 100
) -> None:
    """Update Process Metrics while keeping the last valid view on gaps."""
    try:
        if data and (data.get("history") or []):
            history = data["history"]
            window = history[-window_n:]
            if window:
                panel["_last_ok_data"] = data
                panel["_last_ok_window"] = window
        else:
            data = panel.get("_last_ok_data") or {}
            window = panel.get("_last_ok_window") or []

        if not window:
            return

        window_text = f"window: last {len(window)} samples"
        if panel.get("_last_window_text") != window_text:
            panel["window_text"].content = window_text
            panel["_last_window_text"] = window_text

        roll = _compute_rollups(window)
        _update_graph(panel, data, window)

        kpis_html = _render_kpis(roll, data)
        if panel.get("_last_kpis") != kpis_html:
            panel["kpis"].content = kpis_html
            panel["_last_kpis"] = kpis_html

    except Exception:
        pass


def _build_graph():
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", line=dict(color="#4caf50"), yaxis="y"
        )
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
            title=dict(text="RAM (%)", font=dict(color="#4caf50")),
            tickfont=dict(color="#4caf50"),
        ),
        yaxis2=dict(
            range=[0, 100],
            overlaying="y",
            side="right",
            title=dict(text="GPU Mem (%)", font=dict(color="#ff9800")),
            tickfont=dict(color="#ff9800"),
        ),
    )
    plot = ui.plotly(fig).classes("w-full")
    return plot, fig


def _update_graph(
    panel: Dict[str, Any], data: Dict[str, Any], window: List[Dict[str, Any]]
) -> None:
    try:
        fig = panel["_fig"]
        series = data.get("series", {}) if isinstance(data, dict) else {}
        x_time = series.get("x_time", [])

        ram_total = max(float(window[-1].get("ram_total", 1.0) or 1.0), 1.0)
        ram_pct = [
            (float(r.get("ram_used_max", 0.0) or 0.0) / ram_total) * 100.0
            for r in window
        ]

        if x_time and len(x_time) >= len(window):
            x = x_time[-len(window) :]
        else:
            x = [
                r.get("ts") if r.get("ts") is not None else i
                for i, r in enumerate(window)
            ]

        fig.data[0].x = x
        fig.data[0].y = ram_pct

        gpu_window = [r for r in window if r.get("gpu_used") is not None]
        if gpu_window and window[-1].get("gpu_total") is not None:
            gpu_total = max(
                float(window[-1].get("gpu_total", 1.0) or 1.0), 1.0
            )
            gpu_pct = [
                (float(r.get("gpu_used", 0.0) or 0.0) / gpu_total) * 100.0
                for r in gpu_window
            ]

            if x_time and len(x_time) >= len(window):
                gpu_x = [
                    x[idx]
                    for idx, r in enumerate(window)
                    if r.get("gpu_used") is not None
                ]
            else:
                gpu_x = [
                    r.get("ts") if r.get("ts") is not None else i
                    for i, r in enumerate(gpu_window)
                ]

            fig.data[1].x = gpu_x
            fig.data[1].y = gpu_pct
        else:
            fig.data[1].x = []
            fig.data[1].y = []

        panel["graph"].update_figure(fig)
    except Exception:
        pass


def _render_kpis(roll: Dict[str, Any], snap: Dict[str, Any]) -> str:
    items = [
        compact_metric_html(
            "CPU now/p50/p95",
            f"{roll['cpu']['now']:.0f}% / {roll['cpu']['p50']:.0f}% / {roll['cpu']['p95']:.0f}%",
        ),
        compact_metric_html(
            "RAM now/p95/total",
            f"{fmt_mem_new(roll['ram']['now'])} / {fmt_mem_new(roll['ram']['p95'])} / {fmt_mem_new(roll['ram']['total'])}",
        ),
    ]

    if roll["gpu_available"]:
        items.extend(
            [
                compact_metric_html(
                    "GPU Mem now/p95",
                    f"{fmt_mem_new(roll['gpu']['now'])} / {fmt_mem_new(roll['gpu']['p95'])}",
                ),
                compact_metric_html(
                    "GPU Imbalance",
                    (
                        fmt_mem_new(snap.get("gpu_used_imbalance"))
                        if snap.get("gpu_used_imbalance") is not None
                        else "-"
                    ),
                ),
            ]
        )
    else:
        items.extend(
            [
                compact_metric_html("GPU Mem", "N/A"),
                compact_metric_html("GPU Imbalance", "-"),
            ]
        )

    return (
        "<div style='display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); "
        "gap:6px; padding-top:6px; border-top:1px solid #ececec;'>"
        + "".join(items)
        + "</div>"
    )


def _percentile(vals: List[float], p: float) -> float:
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return 0.0
    k = (len(vals) - 1) * p / 100.0
    f = int(k)
    c = min(int(k) + 1, len(vals) - 1)
    return vals[f] if f == c else vals[f] * (c - k) + vals[c] * (k - f)


def _compute_rollups(window: List[Dict[str, Any]]) -> Dict[str, Any]:
    last = window[-1]

    cpu_hist = [float(r.get("cpu_max", 0.0) or 0.0) for r in window]
    ram_hist = [float(r.get("ram_used_max", 0.0) or 0.0) for r in window]
    ram_total = float(last.get("ram_total", 0.0) or 0.0)

    gpu_available = last.get("gpu_used") is not None
    gpu_hist = [
        float(r.get("gpu_used", 0.0) or 0.0)
        for r in window
        if r.get("gpu_used") is not None
    ]

    return {
        "gpu_available": gpu_available,
        "cpu": {
            "now": cpu_hist[-1],
            "p50": _percentile(cpu_hist, 50),
            "p95": _percentile(cpu_hist, 95),
        },
        "ram": {
            "now": ram_hist[-1],
            "p95": _percentile(ram_hist, 95),
            "total": ram_total,
        },
        "gpu": {
            "now": gpu_hist[-1] if gpu_hist else 0.0,
            "p95": _percentile(gpu_hist, 95) if gpu_hist else 0.0,
        },
    }
