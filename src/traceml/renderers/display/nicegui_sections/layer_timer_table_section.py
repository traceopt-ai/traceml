from nicegui import ui
from traceml.utils.formatting import fmt_time_ms


def build_layer_timer_table_section():
    card = ui.card().classes("m-2 p-4 w-full")
    card.style("""
        background: rgba(245, 245, 245, 0.35);
        backdrop-filter: blur(12px);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    """)

    with card:
        ui.label("Per Layer Timing Stats").classes(
            "text-lg font-bold mb-2"
        ).style("color:#d47a00;")

        container = ui.html("", sanitize=False).style(
            "max-height: 350px; overflow-y: auto; width: 100%;"
        )

    return {"table": container}


def update_layer_timer_table_section(panel, dashboard_data):
    rows = dashboard_data["top_items"]
    other = dashboard_data["other"]

    html = """
    <table style="width:100%; border-collapse: collapse; font-size:14px;">
        <thead style="position: sticky; top: 0; background: #f0f0f0;">
            <tr>
                <th style="text-align:left;">Layer</th>
                <th style="text-align:right;">Calls</th>
                <th style="text-align:right;">Total</th>
                <th style="text-align:right;">Avg</th>
                <th style="text-align:right;">Self</th>
                <th style="text-align:right;">%</th>
            </tr>
        </thead>
        <tbody>
    """

    for r in rows:
        html += f"""
        <tr>
            <td>{r['layer']}</td>
            <td style="text-align:right;">{r['calls']}</td>
            <td style="text-align:right;">{fmt_time_ms(r['total_ms'])}</td>
            <td style="text-align:right;">{fmt_time_ms(r['avg_ms'])}</td>
            <td style="text-align:right;">{fmt_time_ms(r['self_ms'])}</td>
            <td style="text-align:right;">{r['pct']:.1f}%</td>
        </tr>
        """

    if other["total_ms"] > 0:
        html += f"""
        <tr style="color:gray;">
            <td>Other Layers</td>
            <td style="text-align:right;">{other['calls']}</td>
            <td style="text-align:right;">{fmt_time_ms(other['total_ms'])}</td>
            <td style="text-align:right;">—</td>
            <td style="text-align:right;">—</td>
            <td style="text-align:right;">{other['pct']:.1f}%</td>
        </tr>
        """

    html += "</tbody></table>"
    panel["table"].content = html
