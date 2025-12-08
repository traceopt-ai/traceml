from nicegui import ui
import time
import plotly.graph_objects as go
from traceml.utils.formatting import fmt_mem_new
from traceml.renderers.display.nicegui_sections.helper import (
    level_bar_continuous,
    extract_time_axis
)

def build_process_section():

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

        ui.label("Process Metrics") \
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


def update_process_section(panel, data):

    # CPU
    cpu = data["cpu_used"]
    cores = data["cpu_logical_core_count"]

    panel["cpu_text"].content = (
        f"⚙️ CPU ({cores} cores): {cpu:.1f}%"
    )
    panel["cpu_bar"].content = level_bar_continuous(cpu/cores)

    # RAM
    ru, rt = data["ram_used"], data["ram_total"]
    if rt:
        pct = (ru * 100.0) / rt
        panel["ram_text"].content = (
            f"💾 RAM: {fmt_mem_new(ru)} / {fmt_mem_new(rt)} ({pct:.1f}%)"
        )
        panel["ram_bar"].content = level_bar_continuous(pct)
    else:
        panel["ram_text"].content = "💾 RAM: –"
        panel["ram_bar"].content = ""

    # GPU
    used = data["gpu_used"]
    reserved = data["gpu_reserved"]
    total = data["gpu_total"]

    if used is None or total is None:
        panel["gpu_text"].content = "🎮 GPU: Not available"
        panel["gpu_bar"].content = ""
    else:

        used_pct = (used * 100.0) / total
        panel["gpu_text"].content = (
            f"🎮 GPU: {fmt_mem_new(used)} used / "
            f"{fmt_mem_new(reserved)} reserved / "
            f"{fmt_mem_new(total)} total"
        )
        panel["gpu_bar"].content = level_bar_continuous(used_pct)

    _update_graph_section(panel, data["table"])


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
    graph = ui.plotly(fig).classes("w-full mt-4")
    return graph


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


def _update_graph_section(panel, process_table):
    """
    Updates the Plotly graph in process panel using the historical samples.
    process_table = list of records from db.tables["process"]
    """

    if not process_table:
        return

    fig = go.Figure()
    x_hist = extract_time_axis(process_table)
    _update_ram_graph(process_table, fig, x_hist)
    _update_gpu_graph(process_table, fig, x_hist)

    gpu_available = process_table[-1].get("gpu_available", False)
    _update_graph_layout(gpu_available, fig)
    panel["graph"].update_figure(fig)


def _update_ram_graph(process_table, fig, x_hist):
    ram_total = process_table[-1].get("ram_total", 1)
    ram_hist = [
        round(rec.get("ram_used", 0)/ram_total, 2) for rec in process_table
    ][-100:]
    fig.add_trace(go.Scatter(
        y=ram_hist,
        x=x_hist,
        mode="lines",
        name="RAM (%)",
        yaxis="y",
        line=dict(color="#4caf50"),
    ))

<TO DO UPDATE FROM HERE NAMING TO BE CORRECTED>
def _update_gpu_graph(process_table, fig, x_hist):
    gpu_available = process_table[-1].get("gpu_available", False)
    if gpu_available:
        gpu_hist = []
        gpu_total = process_table[-1].get("gpu_total", 1)
        for rec in process_table:
            gpu_raw = rec.get("gpu_process_memory", {}) or {}
            if gpu_raw:
                gpu_hist.append(sum(v.get("used", 0)/gpu_total*100 for v in gpu_raw.values()))
            else:
                gpu_hist.append(0)
        gpu_hist = gpu_hist[-100:]
        gpu_hist = gpu_hist/gpu_total
    else:
        gpu_hist = None

    if gpu_available:
        fig.add_trace(go.Scatter(
            y=gpu_hist,
            x=x_hist,
            mode="lines",
            name="GPU Mem (%)",
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
    else:
        fig.update_layout(
            **common_layout,
            yaxis=dict(
                range=[0, 100],
                title=dict(text="RAM (%)", font=dict(color="#4caf50")),
                tickfont=dict(color="#4caf50"),
            ),
        )
