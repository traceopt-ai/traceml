from nicegui import ui

def build_layer_table_section():
    card = ui.card().classes("m-2 p-4 w-full")
    card.style("""
        background: rgba(245, 245, 245, 0.35);
        backdrop-filter: blur(12px);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    """)

    with card:
        ui.label("Memory Leaderboard").classes("text-lg font-bold mb-2")
        container = ui.html("",  sanitize=False).style(
            "max-height: 260px; overflow-y: auto; width: 100%;"
        )

    return {"table": container}


def update_layer_table_section(panel, dashboard_data):
    rows = dashboard_data["top_layers"]

    # Build HTML
    html = """
    <table style="width:100%; border-collapse: collapse; font-size:14px;">
        <thead style="position: sticky; top: 0; background: #f0f0f0;">
            <tr>
                <th style="text-align:left;">Layer</th>
                <th style="text-align:right;">Total</th>
                <th style="text-align:right;">%</th>
                <th style="text-align:right;">Params</th>
                <th style="text-align:right;">Act Peak</th>
                <th style="text-align:right;">Grad Peak</th>
            </tr>
        </thead>
        <tbody>
    """

    for r in rows:
        html += f"""
        <tr>
            <td>{r['layer']}</td>
            <td style="text-align:right;">{r['total_cost']:.1f}</td>
            <td style="text-align:right;">{r['pct']:.1f}%</td>
            <td style="text-align:right;">{r['param_memory']:.1f}</td>
            <td style="text-align:right;">{r['activation_peak']:.1f}</td>
            <td style="text-align:right;">{r['gradient_peak']:.1f}</td>
        </tr>
        """

    html += "</tbody></table>"

    # Update the HTML container
    panel["table"].content = html

