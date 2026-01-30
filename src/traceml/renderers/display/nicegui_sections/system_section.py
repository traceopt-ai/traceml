"""
System Metrics Dashboard (NiceGUI)

This module renders a live, system-level observability card for a single
training job (single-process or distributed). All metrics are computed as
rolling, time-based summaries over the most recent N system samples.

Metric semantics
----------------
CPU utilization (now / p50 / p95)
    - now: CPU utilization from the latest sample
    - p50: median CPU utilization over the rolling window
    - p95: near-worst CPU utilization over the rolling window
    Percentiles are computed over time, not across ranks.

GPU utilization (avg) and imbalance
    - GPU Util (avg):
        For each sample, GPU utilization is averaged across all visible GPUs.
        Reported as now / p50 / p95 over time.
    - GPU Util Imbalance:
        For each sample, computed as:
            max(GPU util) - min(GPU util)
        Highlights load imbalance or straggler GPUs.
        Reported as now / p95 over time.

RAM usage and headroom
    - now: current host RAM usage
    - p95: near-worst RAM usage over the rolling window
    - headroom: total RAM - current RAM usage
    - total: total host RAM capacity
    RAM metrics are host-level (not per-process).

Worst-GPU memory usage
    For each sample, GPU memory usage is inspected per device and the
    maximum (worst) GPU is selected. This is reported as now / p95 /
    headroom and is intentionally conservative to surface OOM risk even
    when other GPUs have spare capacity.

GPU temperature status
    For each sample, the maximum temperature across all GPUs is used.
    Status is derived from the current value:
        - OK:   < 80 °C
        - Warm: 80–85 °C
        - Hot:  >= 85 °C

Time series graph
    Displays CPU utilization (left axis) and average GPU utilization
    (right axis) over the rolling window. Intended to reveal phase
    changes, stalls, and CPU/GPU utilization coupling during training.

Design notes
------------
- Display-only logic; telemetry is not modified
- Rolling-window statistics (no full-run aggregation)
- Worst-case signals are preferred over averages for failure detection
"""

from itertools import islice

import plotly.graph_objects as go
from nicegui import ui

from traceml.renderers.display.nicegui_sections.helper import extract_x_axis
from traceml.utils.formatting import fmt_mem_new

METRIC_TITLE = "text-l font-bold mb-1 ml-1 break-words whitespace-normal"
LABEL = "text-[11px] font-semibold tracking-wide leading-tight"
VAL = "text-[12.5px] text-gray-700 leading-tight"
SUB = "text-[11px] text-gray-500 leading-tight"


def build_system_section():
    """
    Build the static System Metrics dashboard card.

    Returns
    -------
    dict
        Handles to UI elements that are later updated by
        `update_system_section`.
    """
    card = ui.card().classes("m-2 p-2 w-full")
    card.style(
        """
        background: ffffff;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        height: 360px;
        overflow-y: auto;
        overflow-x: hidden;
        """,
    )

    with card:
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("System Metrics").classes(METRIC_TITLE).style(
                "color:#d47a00;",
            )
            with ui.icon("info").classes("text-gray-400 cursor-pointer"):
                with (
                    ui.menu()
                    .props("anchor='bottom left' self='top left' auto-close")
                    .classes("w-96 p-2"),
                ):
                    ui.markdown(
                        """
            **System Metrics (node-local)**

            - **CPU**: host stats over rolling window.
            - **RAM**: host stats over rolling window.
            - **GPU Util**: average across GPUs visible on this node; skew = max − min.
            - **GPU Mem**: worst GPU (max mem across local GPUs).
            - **Temp**: max GPU temperature on this node.
            """,
                    )
            window_text = ui.html("window: –", sanitize=False).classes(
                "text-xs text-gray-500 mr-1",
            )

        graph = _build_graph()

        # 2 rows x 3 columns
        with ui.grid(columns=3).classes("w-full gap-1 mt-1"):
            _, cpu_v, cpu_s = _tile("CPU (now/p50/p95)")
            _, gpu_v, gpu_s = _tile("GPU Util (now/p50/p95)")
            _, imb_v, imb_s = _tile("GPU Util Skew (now/p95)")
            _, ram_v, ram_s = _tile("RAM (now/p95/total)")
            _, gmem_v, gmem_s = _tile("GPU Mem (now/p95)")
            _, temp_v, temp_s = _tile("Temp (max GPU)")

    return {
        "window_text": window_text,
        "graph": graph,
        "cpu_v": cpu_v,
        "cpu_s": cpu_s,
        "gpu_v": gpu_v,
        "gpu_s": gpu_s,
        "imb_v": imb_v,
        "imb_s": imb_s,
        "ram_v": ram_v,
        "ram_s": ram_s,
        "gmem_v": gmem_v,
        "gmem_s": gmem_s,
        "temp_v": temp_v,
        "temp_s": temp_s,
    }


def _build_graph():
    """Create an empty Plotly graph with fixed layout."""
    fig = go.Figure()
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=2, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        showlegend=False,
        xaxis=dict(showgrid=False),
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


def _tile(title):
    """Create a compact metric tile."""
    box = ui.column().classes("w-full px-1 py-1").style("min-height: 46px;")
    with box:
        ui.html(title, sanitize=False).classes(LABEL).style("color:#ff9800;")
        v = ui.html("–", sanitize=False).classes(VAL)
        s = ui.html("", sanitize=False).classes(SUB)
    return box, v, s


def update_system_section(panel, data, window_n=100):
    """
    Update the System Metrics dashboard with new telemetry.

    Parameters
    ----------
    panel : dict
        UI handles returned by build_system_section
    data : dict
        Renderer output containing the system table
    window_n : int
        Rolling window size
    """
    table = (data or {}).get("table") or []
    if not table:
        panel["window_text"].content = "window: –"
        return

    window = _last_n(table, window_n)
    panel["window_text"].content = f"window: last {len(window)} samples"

    roll = _compute_rollups(window)

    _update_tiles(panel, roll)
    _update_graph(panel, window)


def _update_tiles(panel, roll):
    """Update all metric tiles."""
    cpu = roll["cpu"]
    panel["cpu_v"].content = (
        f"<b>{cpu['now']:.0f}%</b> / {cpu['p50']:.0f}% / {cpu['p95']:.0f}%"
    )

    ram = roll["ram"]
    if ram["total"] > 0:
        panel["ram_v"].content = (
            f"<b>{fmt_mem_new(ram['now'])}</b>/"
            f"{fmt_mem_new(ram['p95'])}/"
            f"({fmt_mem_new(ram['total'])})"
        )
    else:
        panel["ram_v"].content = "–"

    if not roll["gpu_available"]:
        for k in ("gpu_v", "imb_v", "gmem_v", "temp_v"):
            panel[k].content = "Not available"
        return

    gpu = roll["gpu_util"]
    panel["gpu_v"].content = (
        f"<b>{gpu['now']:.0f}%</b> / {gpu['p50']:.0f}% / {gpu['p95']:.0f}%"
    )

    imb = roll["gpu_delta"]
    panel["imb_v"].content = f"{imb['now']:.0f}% / {imb['p95']:.0f}%"

    gmem = roll["gpu_mem"]
    panel["gmem_v"].content = (
        f"<b>{fmt_mem_new(gmem['now'])}</b>/" f"{fmt_mem_new(gmem['p95'])}"
    )

    temp = roll["temp"]
    panel["temp_v"].content = f"{temp['now']:.0f}°C · p95 {temp['p95']:.0f}°C"
    panel["temp_s"].content = f"Status: {temp['status']}"


def _update_graph(panel, window):
    """Rebuild and update the CPU/GPU utilization time series graph."""
    x = extract_x_axis(window)
    if not x or len(x) != len(window):
        x = list(range(len(window)))

    cpu_hist = [r.get("cpu", 0.0) or 0.0 for r in window]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=cpu_hist,
            mode="lines",
            line=dict(color="#4caf50"),
            yaxis="y",
        ),
    )

    if window[-1].get("gpu_available"):
        gpu_hist = []
        for r in window:
            utils = _gpu_utils(r)
            gpu_hist.append(sum(utils) / len(utils) if utils else 0.0)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=gpu_hist,
                mode="lines",
                line=dict(color="#ff9800"),
                yaxis="y2",
            ),
        )

    fig.update_layout(panel["graph"].figure.layout)
    panel["graph"].update_figure(fig)


def _last_n(table, n):
    """
    Return the last n records efficiently.
    Works for deque and list.
    Cost: O(n)
    """
    if not table:
        return []
    # deque supports reversed() efficiently
    if hasattr(table, "__reversed__"):
        return list(islice(reversed(table), n))[::-1]

    # fallback for other sequences
    return table[-n:] if len(table) > n else table


def _percentile(vals, p):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return 0.0
    k = (len(vals) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[f] if f == c else vals[f] * (c - k) + vals[c] * (k - f)


def _gpu_utils(rec):
    gpus = rec.get("gpus") or []
    return [g[0] for g in gpus if g]


def _gpu_mems(rec):
    gpus = rec.get("gpus") or []
    return [(g[1], g[2]) for g in gpus if g]


def _gpu_temps(rec):
    gpus = rec.get("gpus") or []
    return [g[3] for g in gpus if g]


def _compute_rollups(window):
    """Compute rolling statistics for dashboard display."""
    last = window[-1]

    cpu_hist = [r.get("cpu", 0.0) or 0.0 for r in window]

    ram_used = [r.get("ram_used", 0.0) or 0.0 for r in window]
    ram_total = last.get("ram_total", 0.0) or 0.0

    gpu_available = bool(last.get("gpu_available", False))

    gpu_avg_hist = []
    gpu_delta_hist = []
    gpu_mem_hist = []
    temp_hist = []

    for r in window:
        utils = _gpu_utils(r)
        mems = _gpu_mems(r)
        temps = _gpu_temps(r)

        if utils:
            gpu_avg_hist.append(sum(utils) / len(utils))
            gpu_delta_hist.append(max(utils) - min(utils))
        else:
            gpu_avg_hist.append(0.0)
            gpu_delta_hist.append(0.0)

        gpu_mem_hist.append(max((m for m, _ in mems), default=0.0))
        temp_hist.append(max(temps, default=0.0))

    max_gpu_capacity = max(
        (m for _, m in _gpu_mems(last)),
        default=0.0,
    )

    return {
        "gpu_available": gpu_available,
        "cpu": {
            "now": cpu_hist[-1],
            "p50": _percentile(cpu_hist, 50),
            "p95": _percentile(cpu_hist, 95),
        },
        "ram": {
            "now": ram_used[-1],
            "p95": _percentile(ram_used, 95),
            "total": ram_total,
            "headroom": max(ram_total - ram_used[-1], 0.0),
        },
        "gpu_util": {
            "now": gpu_avg_hist[-1],
            "p50": _percentile(gpu_avg_hist, 50),
            "p95": _percentile(gpu_avg_hist, 95),
        },
        "gpu_delta": {
            "now": gpu_delta_hist[-1],
            "p95": _percentile(gpu_delta_hist, 95),
        },
        "gpu_mem": {
            "now": gpu_mem_hist[-1],
            "p95": _percentile(gpu_mem_hist, 95),
            "headroom": max(max_gpu_capacity - gpu_mem_hist[-1], 0.0),
        },
        "temp": {
            "now": temp_hist[-1],
            "p95": _percentile(temp_hist, 95),
            "status": (
                "Hot"
                if temp_hist[-1] >= 85
                else "Warm" if temp_hist[-1] >= 80 else "OK"
            ),
        },
    }
