"""
System Metrics Dashboard (NiceGUI) section.

UI-only module:
- Builds the System Metrics card
- Updates it from the dashboard payload produced by SystemMetricsComputer

Expected payload schema (from SystemRenderer.get_dashboard_renderable()):
{
  "window_len": int,
  "gpu_available": bool,
  "rollups": {
    "gpu_available": bool,
    "cpu": {"now","p50","p95"},
    "ram": {"now","p95","total","headroom"},
    "gpu_util": {"now","p50","p95"},
    "gpu_delta": {"now","p95"},
    "gpu_mem": {"now","p95","headroom"},
    "temp": {"now","p95","status"},
  },
  "series": {
    "cpu": [..],
    "gpu_avg": [..],   # may be empty if gpu not available
  }
}
"""

import plotly.graph_objects as go
from nicegui import ui

from traceml.aggregator.display_drivers.nicegui_sections.helper import extract_x_axis
from traceml.utils.formatting import fmt_mem_new

METRIC_TITLE = "text-l font-bold mb-1 ml-1 break-words whitespace-normal"
LABEL = "text-[11px] font-semibold tracking-wide leading-tight"
VAL = "text-[12.5px] text-gray-700 leading-tight"
SUB = "text-[11px] text-gray-500 leading-tight"


def build_system_section() -> dict:
    """Build the static System Metrics dashboard card and return UI handles."""
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
        """
    )

    with card:
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("System Metrics").classes(METRIC_TITLE).style("color:#d47a00;")
            with ui.icon("info").classes("text-gray-400 cursor-pointer"):
                with (
                    ui.menu()
                    .props("anchor='bottom left' self='top left' auto-close")
                    .classes("w-96 p-2")
                ):
                    ui.markdown(
                        """
                        **System Metrics (node-local)**
                        
                        - **CPU**: host stats over rolling window.
                        - **RAM**: host stats over rolling window.
                        - **GPU Util**: average across GPUs visible on this node; skew = max − min.
                        - **GPU Mem**: worst GPU (max mem across local GPUs).
                        - **Temp**: max GPU temperature on this node.
                        """
                    )
            window_text = ui.html("window: –", sanitize=False).classes("text-xs text-gray-500 mr-1")

        graph = _build_graph()

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


def update_system_section(panel: dict, payload: dict) -> None:
    try:
        payload = payload or {}
        window_len = int(payload.get("window_len", 0) or 0)
        roll = payload.get("rollups") or {}
        series = payload.get("series") or {}

        if window_len <= 0 or not roll:
            panel["window_text"].content = "window: –"
            return

        panel["window_text"].content = f"window: last {window_len} samples"

        # If GPU is unavailable, proactively clear temp subtitle to avoid stale status text.
        if not roll.get("gpu_available", False):
            try:
                panel["temp_s"].content = ""
            except Exception:
                pass

        _update_tiles(panel, roll)
        _update_graph(panel, series)

    except Exception:
        try:
            panel["window_text"].content = "window: –"
        except Exception:
            pass
        return


def _build_graph():
    """Create an empty Plotly graph with fixed layout (same as your current config)."""
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


def _tile(title: str):
    """Create a compact metric tile."""
    box = ui.column().classes("w-full px-1 py-1").style("min-height: 46px;")
    with box:
        ui.html(title, sanitize=False).classes(LABEL).style("color:#ff9800;")
        v = ui.html("–", sanitize=False).classes(VAL)
        s = ui.html("", sanitize=False).classes(SUB)
    return box, v, s


def _update_tiles(panel: dict, roll: dict) -> None:
    cpu = roll.get("cpu")
    ram = roll.get("ram")
    if not isinstance(cpu, dict) or not isinstance(ram, dict):
        return  # payload incomplete; keep previous values rather than crash

    panel["cpu_v"].content = f"<b>{cpu['now']:.0f}%</b> / {cpu['p50']:.0f}% / {cpu['p95']:.0f}%"

    if ram["total"] > 0:
        panel["ram_v"].content = (
            f"<b>{fmt_mem_new(ram['now'])}</b>/"
            f"{fmt_mem_new(ram['p95'])}/"
            f"({fmt_mem_new(ram['total'])})"
        )
    else:
        panel["ram_v"].content = "–"

    if not roll.get("gpu_available", False):
        for k in ("gpu_v", "imb_v", "gmem_v", "temp_v"):
            panel[k].content = "Not available"
        panel["temp_s"].content = ""  # prevent stale "Status: ..." when GPU disappears
        return

    gpu = roll.get("gpu_util")
    imb = roll.get("gpu_delta")
    gmem = roll.get("gpu_mem")
    temp = roll.get("temp")
    if not all(isinstance(x, dict) for x in (gpu, imb, gmem, temp)):
        return  # incomplete GPU payload; don't crash

    panel["gpu_v"].content = f"<b>{gpu['now']:.0f}%</b> / {gpu['p50']:.0f}% / {gpu['p95']:.0f}%"
    panel["imb_v"].content = f"{imb['now']:.0f}% / {imb['p95']:.0f}%"
    panel["gmem_v"].content = f"<b>{fmt_mem_new(gmem['now'])}</b>/{fmt_mem_new(gmem['p95'])}"

    panel["temp_v"].content = f"{temp['now']:.0f}°C · p95 {temp['p95']:.0f}°C"
    panel["temp_s"].content = f"Status: {temp['status']}"


def _update_graph(panel: dict, series: dict) -> None:
    cpu_hist = list(series.get("cpu") or [])
    gpu_hist = list(series.get("gpu_avg") or [])

    if not cpu_hist:
        # clear graph explicitly
        fig = go.Figure()
        fig.update_layout(panel["graph"].figure.layout)
        panel["graph"].update_figure(fig)
        return

    x = list(range(len(cpu_hist)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=cpu_hist, mode="lines", line=dict(color="#4caf50"), yaxis="y"))

    if gpu_hist:
        m = min(len(cpu_hist), len(gpu_hist))
        fig.add_trace(go.Scatter(x=x[:m], y=gpu_hist[:m], mode="lines", line=dict(color="#ff9800"), yaxis="y2"))

    fig.update_layout(panel["graph"].figure.layout)
    panel["graph"].update_figure(fig)