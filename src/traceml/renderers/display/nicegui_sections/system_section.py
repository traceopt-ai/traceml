from nicegui import ui
import plotly.graph_objects as go
from traceml.utils.formatting import fmt_mem_new
from traceml.renderers.display.nicegui_sections.helper import (
    level_bar_continuous,
    extract_time_axis
)


def build_system_section():

    card = ui.card().classes("m-2 p-2 w-full")
    card.style("""
        background: ffffff;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        overflow-y: auto;
        line-height: 1.1;
        height: 350px;
    """)

    with card:
        ui.label("System Metrics") \
            .classes("text-xl font-bold mb-1 ml-1 break-words whitespace-normal") \
            .style("color:#d47a00;")
        graph = _build_graph_section()
        with ui.grid(columns=4).classes("w-full gap-x-3 gap-y-2"):
            # CPU + GPU UTIL ==========
            cpu_text = ui.html("CPU: –", sanitize=False).classes("text-sm").style("color:#333")
            cpu_bar  = ui.html("", sanitize=False)
            gpu_util_text = ui.html("GPU Util: –", sanitize=False).classes("text-sm")
            gpu_util_bar  = ui.html("", sanitize=False)

            # RAM + GPU MEM
            ram_text = ui.html("RAM: –", sanitize=False).classes("text-sm").style("color:#333")
            ram_bar  = ui.html("", sanitize=False)
            gpu_mem_text = ui.html("GPU Mem: –", sanitize=False).classes("text-sm")
            gpu_mem_bar  = ui.html("", sanitize=False)

            # TEMP + POWER
            temp_text = ui.html("Temp: –", sanitize=False).classes("text-sm").style("color:#333")
            empty1    = ui.html("", sanitize=False)
            #
            power_text = ui.html("Power: –", sanitize=False).classes("text-sm").style("color:#333")
            empty2 = ui.html("", sanitize=False)

    return {
        "cpu_text": cpu_text, "cpu_bar": cpu_bar,
        "gpu_util_text": gpu_util_text, "gpu_util_bar": gpu_util_bar,
        "ram_text": ram_text, "ram_bar": ram_bar,
        "gpu_mem_text": gpu_mem_text, "gpu_mem_bar": gpu_mem_bar,
        "graph": graph,
        "temp_text": temp_text,
        "power_text": power_text,
    }


def _build_graph_section():
    fig = go.Figure()
    fig.update_layout(
        height=175,
        margin=dict(l=10, r=10, t=10, b=35),
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
    graph = ui.plotly(fig).classes("w-full mt-1")
    return graph


def update_system_section(panel, data):

    # CPU
    cpu_pct = data["cpu"]
    panel["cpu_text"].content = f"CPU: {cpu_pct:.1f}%"
    panel["cpu_bar"].content = level_bar_continuous(cpu_pct)

    # RAM
    ru, rt = data["ram_used"], data["ram_total"]
    if rt:
        pct = (ru * 100.0) / rt
        panel["ram_text"].content = (
            f"RAM: {fmt_mem_new(ru)}/{fmt_mem_new(rt)} ({pct:.1f}%)"
        )
        panel["ram_bar"].content = level_bar_continuous(pct)
    else:
        panel["ram_text"].content = "RAM: –"
        panel["ram_bar"].content = ""

    # GPU Util
    if not data.get("gpu_available", False):
        panel["gpu_util_text"].content = "GPU Util: Not available"
        panel["gpu_util_bar"].content = ""
        panel["gpu_mem_text"].content = "GPU Mem: Not available"
        panel["gpu_mem_bar"].content = ""
        panel["temp_text"].content = "Temp: Not available"
        panel["power_text"].content = "Power: Not available"

    else:
        util_avg = _compute_gpu_avg_util(data)
        panel["gpu_util_text"].content = f"GPU Util: {util_avg:.1f}%"
        panel["gpu_util_bar"].content = level_bar_continuous(util_avg)
        gmu = data.get("gpu_mem_used", 0.0) or 0.0
        gmt = data.get("gpu_mem_total", 1.0)

        g_pct = (gmu * 100.0) / gmt
        panel["gpu_mem_text"].content = (
            f"GPU Mem: {fmt_mem_new(gmu)}/{fmt_mem_new(gmt)} ({g_pct:.1f}%)"
        )
        panel["gpu_mem_bar"].content = level_bar_continuous(g_pct)
        temp = data.get("gpu_temp_max", 0.0)
        panel["temp_text"].content = f"Temp: {temp:.0f}°C"
        p_used = data.get("gpu_power_usage", None)
        p_lim = data.get("gpu_power_limit", None)
        panel["power_text"].content = f"Power: {p_used:.0f}W/{p_lim:.0f}W"

    _update_graph_section(panel, data["table"])
    return



def _compute_gpu_avg_util(data):
    """Compute average GPU utilization (0–100)."""
    if not data.get("gpu_available", False):
        return None

    util_total = data.get("gpu_util_total", 0.0) or 0.0
    gpu_count = data.get("gpu_count", 1) or 1

    util_avg = util_total / gpu_count
    return util_avg


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
        height=175,
        margin=dict(l=10, r=10, t=10, b=35),
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
