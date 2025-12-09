from nicegui import ui
import plotly.graph_objects as go
from traceml.utils.formatting import fmt_mem_new
from traceml.renderers.display.nicegui_sections.helper import (
    level_bar_continuous,
    extract_time_axis
)


def build_system_section():

    card = ui.card().classes("m-2 p-4 w-full")
    card.style("""
        background: rgba(245, 245, 245, 0.35);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    """)

    with card:
        ui.label("System Metrics") \
            .classes("text-xl font-bold mb-3") \
            .style("color:#ff9800;")
        graph = _build_graph_section()
        cpu_text, cpu_bar = _build_cpu_section()
        ram_text, ram_bar = _build_ram_section()
        gpu_text, gpu_bar = _build_gpu_section()


    return {
        "cpu_text": cpu_text, "cpu_bar": cpu_bar,
        "ram_text": ram_text, "ram_bar": ram_bar,
        "gpu_text": gpu_text, "gpu_bar": gpu_bar,
        "graph": graph,
    }


def _build_cpu_section():
    with ui.row().classes("items-center justify-between w-full"):
        cpu_text = ui.html("CPU: –", sanitize=False).classes("text-lg").style("color:#333")
        cpu_bar = ui.html("", sanitize=False)
    return cpu_text, cpu_bar


def _build_ram_section():
    with ui.row().classes("items-center justify-between w-full"):
        ram_text = ui.html("RAM: –", sanitize=False).classes("text-lg").style("color:#333")
        ram_bar = ui.html("", sanitize=False)
    return ram_text, ram_bar


def _build_gpu_section():
    with ui.row().classes("items-center justify-between w-full"):
        gpu_text = ui.html("GPU: –", sanitize=False).classes("text-lg").style("color:#333")
        gpu_bar = ui.html("", sanitize=False)
    return gpu_text, gpu_bar


def _build_graph_section():
    fig = go.Figure()
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",

        xaxis=dict(showgrid=False, visible=False),

        yaxis=dict(
            range=[0, 100],
            title=dict(text="CPU Util (%)", font=dict(color="#4caf50")),
            tickfont=dict(color="#4caf50"),
        ),

        yaxis2=dict(
            range=[0, 100],
            overlaying="y",
            side="right",
            title=dict(text="GPU Util (%)", font=dict(color="#ff9800")),
            tickfont=dict(color="#ff9800"),
        ),
    )
    graph = ui.plotly(fig).classes("w-full mt-4")
    return graph


def update_system_section(panel, data):

    # CPU
    cpu_pct = data["cpu"]
    panel["cpu_text"].content = f"⚙️ CPU: {cpu_pct:.1f}%"
    panel["cpu_bar"].content = level_bar_continuous(cpu_pct)

    # RAM
    ru, rt = data["ram_used"], data["ram_total"]
    if rt:
        pct = (ru * 100.0) / rt
        panel["ram_text"].content = (
            f"💾 RAM: {fmt_mem_new(ru)} / {fmt_mem_new(rt)} ({pct:.1f}%)"
        )
        panel["ram_bar"].content = level_bar_continuous(pct)
    else:
        panel["ram_text"].content = "RAM: –"
        panel["ram_bar"].content = ""

    # GPU
    if not data["gpu_available"]:
        panel["gpu_text"].content = "🎮 GPU: Not available"
        panel["gpu_bar"].content = ""
    else:

        util = data["gpu_util_total"]
        panel["gpu_text"].content = f"🎮 GPU: {util:.1f}%"
        panel["gpu_bar"].content = level_bar_continuous(util)

    _update_graph_section(panel, data["table"])
    return



def _update_graph_section(panel, system_table):
    """
    Updates the Plotly graph in system panel using the historical samples.
    system_table = list of records from db.tables["system"]
    """

    if not system_table:
        return

    fig = go.Figure()
    x_hist = extract_time_axis(system_table)
    _update_cpu_graph(system_table, fig, x_hist)
    _update_gpu_graph(system_table, fig, x_hist)

    gpu_available = system_table[-1].get("gpu_available", False)
    _update_graph_layout(gpu_available, fig)
    panel["graph"].update_figure(fig)



def _update_cpu_graph(system_table, fig, x_hist):
    cpu_hist = [rec.get("cpu_percent", 0) for rec in system_table][-100:]
    fig.add_trace(go.Scatter(
        y=cpu_hist,
        x=x_hist,
        mode="lines",
        name="CPU Util(%)",
        yaxis="y",
        line=dict(color="#4caf50"),
    ))



def _update_gpu_graph(system_table, fig, x_hist):
    gpu_available = system_table[-1].get("gpu_available", False)
    if gpu_available:
        gpu_hist = []
        for rec in system_table:
            gpu_raw = rec.get("gpu_raw", {}) or {}
            if gpu_raw:
                gpu_hist.append(sum(v.get("util", 0) for v in gpu_raw.values()))
            else:
                gpu_hist.append(0)
        gpu_hist = gpu_hist[-100:]

        fig.add_trace(go.Scatter(
            y=gpu_hist,
            x=x_hist,
            mode="lines",
            name="GPU Util (%)",
            yaxis="y2",
            line=dict(color="#ff9800"),
        ))


def _update_graph_layout(gpu_available, fig):
    common_layout = dict(
        height=180,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",

        xaxis=dict(
            showgrid=False,
            tickangle=-30,
            tickmode="auto",
            nticks=10,          # LIMIT LABELS TO 10
        ),

        showlegend=False,
    )

    if gpu_available:
        fig.update_layout(
            **common_layout,
            yaxis=dict(
                range=[0, 100],
                title=dict(text="CPU Util(%)", font=dict(color="#4caf50")),
                tickfont=dict(color="#4caf50"),
            ),
            yaxis2=dict(
                range=[0, 100],
                overlaying="y",
                side="right",
                title=dict(text="GPU Util(%)", font=dict(color="#ff9800")),
                tickfont=dict(color="#ff9800"),
            ),
        )
    else:
        fig.update_layout(
            **common_layout,
            yaxis=dict(
                range=[0, 100],
                title=dict(text="CPU Util(%)", font=dict(color="#4caf50")),
                tickfont=dict(color="#4caf50"),
            ),
        )
