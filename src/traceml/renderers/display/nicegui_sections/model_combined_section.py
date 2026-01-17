from nicegui import ui
import plotly.graph_objects as go
from traceml.renderers.utils import fmt_time_run, fmt_mem_new


METRIC_TEXT = "text-sm leading-normal text-gray-700"
METRIC_TITLE = "text-l font-bold mb-1 ml-1 break-words whitespace-normal"


def _build_graph(title: str):
    fig = go.Figure()
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=4, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            title="step",
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            tickfont=dict(size=9),
        ),
    )
    return ui.plotly(fig).classes("w-full")


def build_model_combined_section():
    with ui.grid(columns=3).classes("m-2 w-full gap-x-3"):
        dl = _metric_card("Dataloader Fetch Time", fmt_time_run)
        step = _metric_card("Training Step Time", fmt_time_run)
        mem = _metric_card("GPU Step Memory", fmt_mem_new)

    return {
        "dl": dl,
        "step": step,
        "mem": mem,
    }


def _update_graph(graph, x, y, y_label: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(width=2),
        )
    )
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=4, b=18),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        xaxis=dict(
            title=dict(text="Training Step", font=dict(color="#4caf50")),
            showgrid=False,
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(color="#4caf50")),
            tickfont=dict(size=9),
        ),
    )
    graph.update_figure(fig)


def _format_stats(stats, fmt):
    return (
        f"last {fmt(stats['last'])} | "
        f"p50 {fmt(stats['p50'])} | "
        f"p95 {fmt(stats['p95'])} | "
        f"avg {fmt(stats['avg100'])} {stats['trend']}"
    )


def _metric_card(title: str, formatter):
    card = ui.card().classes("p-2 w-full")
    card.style(
        """
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 10px rgba(0,0,0,0.10);
        height: 300px;
        """
    )

    with card:
        ui.label(title).classes(
            "text-l font-bold mb-1 ml-1 break-words whitespace-normal"
        ).style("color:#d47a00;")
        graph = _build_graph("")
        stats = ui.html("", sanitize=False).classes(METRIC_TEXT).style("color:#333")

    return {
        "card": card,
        "graph": graph,
        "stats": stats,
        "formatter": formatter,
    }


def _format_stats_table(stats, fmt):
    trend = stats.get("trend", "")
    trend_symbol = "—"
    trend_color = "#666"

    if trend == "+":
        trend_symbol = "↑"
        trend_color = "#d32f2f"  # red
    elif trend == "-":
        trend_symbol = "↓"
        trend_color = "#2e7d32"  # green

    return f"""
    <table style="
        width:100%;
        border-collapse: collapse;
        font-size:14px;
        margin-top:6px;
    ">
        <thead>
            <tr style="border-bottom:1px solid #e0e0e0;">
                <th style="text-align:left; padding:4px 6px; font-weight:700; color:#000;">
                    Last
                </th>
                <th style="text-align:right; padding:4px 6px; font-weight:700; color:#000;">
                    p50(100)
                </th>
                <th style="text-align:right; padding:4px 6px; font-weight:700; color:#000;">
                    p95(100)
                </th>
                <th style="text-align:right; padding:4px 6px; font-weight:700; color:#000;">
                    Avg(100)
                </th>
                <th style="text-align:center; padding:4px 6px; font-weight:700; color:#000;">
                    Trend
                </th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="text-align:left; padding:6px; color:#000;">
                    {fmt(stats["last"])}
                </td>
                <td style="text-align:right; padding:6px; color:#000;">
                    {fmt(stats["p50"])}
                </td>
                <td style="text-align:right; padding:6px; color:#000;">
                    {fmt(stats["p95"])}
                </td>
                <td style="text-align:right; padding:6px; color:#000;">
                    {fmt(stats["avg100"])}
                </td>
                <td style="
                    text-align:center;
                    padding:6px;
                    font-weight:700;
                    color:{trend_color};
                ">
                    {trend_symbol}
                </td>
            </tr>
        </tbody>
    </table>
    """


def update_model_combined_section(panel, telemetry):

    mapping = {
        "dl": "dataLoader_fetch",
        "step": "step_time",
        "mem": "step_gpu_memory",
    }

    for key, metric in mapping.items():
        entry = panel[key]
        tlm = telemetry[metric]

        if key in ["dl", "step"]:
            y_label = "Time(ms)"
        else:
            y_label = "Memory(MB)"
            tlm["y"] = [v / 1024 / 1024 for v in tlm["y"]]

        _update_graph(entry["graph"], tlm["x"], tlm["y"], y_label)
        entry["stats"].content = _format_stats_table(tlm["stats"], entry["formatter"])
