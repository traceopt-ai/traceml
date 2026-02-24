"""
Process Metrics Dashboard (NiceGUI)

This module renders a live, process-level observability card.

Metric semantics
----------------
CPU (now / p50 / p95)
    - CPU usage of the *training process*
    - Percentiles computed over a rolling window of local samples

RAM (now / p95 / total)
    - RSS memory of the training process
    - Host-level RAM, but process-scoped

GPU memory (now / p95 / total)
    - GPU memory used by the training process
    - Local rank history only

GPU memory imbalance
    - max(mem_used) - min(mem_used) across ranks (current only)

Time series graph
-----------------
Displays:
    - RAM % (left axis)
    - GPU memory % (right axis)
Over a rolling window of the most recent N local samples.

Design notes
------------
- Rolling window statistics (no full-run aggregation)
- Graph uses *local rank-0 history only*
- Cross-rank signals are snapshot-only
- Display-only logic
"""

import plotly.graph_objects as go
from nicegui import ui

from traceml.utils.formatting import fmt_mem_new

METRIC_TITLE = "text-l font-bold mb-1 ml-1 break-words whitespace-normal"
LABEL = "text-[11px] font-semibold tracking-wide leading-tight"
VAL = "text-[12.5px] text-gray-700 leading-tight"
SUB = "text-[11px] text-gray-500 leading-tight"


def build_process_section():
    """
    Build the static Process Metrics dashboard card.

    Returns
    -------
    dict
        Handles to UI elements consumed by update_process_section.
    """
    card = ui.card().classes("m-2 p-2 w-full")
    card.style(
        """
        background: #ffffff;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        height: 360px;
        overflow-y: auto;
        overflow-x: hidden;
        """
    )

    with card:
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Process Metrics").classes(METRIC_TITLE).style(
                "color:#d47a00;"
            )
            with ui.icon("info").classes("text-gray-400 cursor-pointer"):
                with ui.menu().props("anchor='bottom left' self='top left'"):
                    ui.markdown(
                        """
            **Process Metrics**

            - **CPU**: worst rank (current), percentiles over rolling window
            - **RAM**: worst rank (current), percentiles over rolling window
            - **GPU mem**: most constrained rank with least headroom
            - **Imbalance**: max − min across ranks (current)
            """
                    )

            window_text = ui.html("window: –", sanitize=False).classes(
                "text-xs text-gray-500 mr-1"
            )

        graph, fig = _build_graph()

        with ui.grid(columns=3).classes("w-full gap-1 mt-1"):
            _, cpu_v, _ = _tile("CPU (now/p50/p95)")
            _, ram_v, _ = _tile("RAM (now/p95/total)")
            _, gmem_v, _ = _tile("GPU Mem (now/p95)")
            _, imb_v, _ = _tile("GPU Mem Imbalance")

    return {
        "window_text": window_text,
        "graph": graph,
        "cpu_v": cpu_v,
        "ram_v": ram_v,
        "gmem_v": gmem_v,
        "imb_v": imb_v,
        "_last_ok_data": None,
        "_last_ok_window": None,
        "_fig": fig
    }


def _build_graph():
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            yaxis="y",
            line=dict(color="#4caf50"),  # RAM green
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            yaxis="y2",
            line=dict(color="#ff9800"),  # GPU orange
        )
    )

    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=2, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        showlegend=False,
        xaxis=dict(showgrid=False),
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


def _tile(title):
    box = ui.column().classes("w-full px-1 py-1").style("min-height: 46px;")
    with box:
        ui.html(title, sanitize=False).classes(LABEL).style("color:#ff9800;")
        v = ui.html("–", sanitize=False).classes(VAL)
    return box, v, None


def update_process_section(panel, data, window_n=100):
    try:
        if data and (data.get("history") or []):
            history = data["history"]
            window = history[-window_n:]
            if window:
                panel["_last_ok_data"] = data
                panel["_last_ok_window"] = window
        else:
            # fallback to last good (prevents flicker)
            data = panel.get("_last_ok_data") or {}
            window = panel.get("_last_ok_window") or []

        if not window:
            panel["window_text"].content = "window: –"
            return

        panel["window_text"].content = f"window: last {len(window)} samples"
        roll = _compute_rollups(window)
        _update_tiles(panel, roll, data)
        _update_graph(panel, window)

    except Exception:
        # never crash/freeze the update loop
        try:
            data = panel.get("_last_ok_data") or {}
            window = panel.get("_last_ok_window") or []
            if window:
                roll = _compute_rollups(window)
                _update_tiles(panel, roll, data)
                _update_graph(panel, window)
        except Exception:
            pass
        return


def _update_tiles(panel, roll, snap):
    cpu = roll["cpu"]
    panel["cpu_v"].content = (
        f"{cpu['now']:.0f}%/ {cpu['p50']:.0f}% / {cpu['p95']:.0f}%"
    )
    ram = roll["ram"]
    panel["ram_v"].content = (
        f"{fmt_mem_new(ram['now'])}/"
        f"{fmt_mem_new(ram['p95'])}/"
        f"{fmt_mem_new(ram['total'])}"
    )
    if roll["gpu_available"]:
        g = roll["gpu"]
        panel["gmem_v"].content = (
            f"{fmt_mem_new(g['now'])}/" f"{fmt_mem_new(g['p95'])}"
        )
    else:
        panel["gmem_v"].content = "Not available"

    panel["imb_v"].content = (
        fmt_mem_new(snap.get("gpu_used_imbalance"))
        if snap.get("gpu_used_imbalance") is not None
        else "–"
    )


def _update_graph(panel, window):
    window = [
        r for r in window
        if isinstance(r, dict)
        and r.get("ram_total") is not None
        and r.get("ram_used_max") is not None
    ]
    if not window:
        return

    fig = panel["_fig"]

    x = list(range(len(window)))

    ram_total = max(window[-1]["ram_total"], 1.0)
    ram_pct = [(r["ram_used_max"] / ram_total) * 100.0 for r in window]

    # Update RAM trace (trace 0)
    fig.data[0].x = x
    fig.data[0].y = ram_pct

    # Update GPU trace (trace 1)
    if window[-1].get("gpu_used") is not None:
        gpu_total = max(window[-1].get("gpu_total", 1.0), 1.0)
        gpu_pct = [
            (r["gpu_used"] / gpu_total) * 100.0
            for r in window
            if r.get("gpu_used") is not None
        ]
        fig.data[1].x = list(range(len(gpu_pct)))
        fig.data[1].y = gpu_pct
    else:
        fig.data[1].x = []
        fig.data[1].y = []

    panel["graph"].update_figure(fig)


def _percentile(vals, p):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return 0.0
    k = (len(vals) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[f] if f == c else vals[f] * (c - k) + vals[c] * (k - f)


def _compute_rollups(window):
    last = window[-1]

    cpu_hist = [r["cpu_max"] for r in window]
    ram_hist = [r["ram_used_max"] for r in window]
    ram_total = last["ram_total"]

    gpu_available = last.get("gpu_used") is not None
    gpu_hist = [r["gpu_used"] for r in window if r.get("gpu_used") is not None]

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
