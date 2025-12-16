from nicegui import ui
from traceml.utils.formatting import fmt_mem_new

def build_layer_memory_table_section():
    card = ui.card().classes("m-2 p-4 w-full")
    card.style("""
        background: rgba(245, 245, 245, 0.35);
        backdrop-filter: blur(12px);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    """)

    with card:
        ui.label("Per Layer Memory Stats").classes("text-lg font-bold mb-2").style("color:#d47a00;")
        container = ui.html("", sanitize=False).style(
             "max-height: 350px; overflow-y: auto; width: 100%;"
        )

    return {"table": container}


def update_layer_memory_table_section(panel, dashboard_data):
    rows = dashboard_data["top_items"]      # NEW: matches service
    other = dashboard_data["other"]

    # Build HTML table
    html = """
    <table style="width:100%; border-collapse: collapse; font-size:14px;">
        <thead style="position: sticky; top: 0; background: #f0f0f0;">
            <tr>
                <th style="text-align:left;">Layer</th>
                <th style="text-align:right;">Params</th>
                <th style="text-align:right;">Activation (curr/peak)</th>
                <th style="text-align:right;">Gradient (curr/peak)</th>
                <th style="text-align:right;">Total Current</th>
                <th style="text-align:right;">%</th>
            </tr>
        </thead>
        <tbody>
    """
    for r in rows:
        html += f"""
        <tr>
            <td>{r['layer']}</td>
            <td style="text-align:right;">{fmt_mem_new(r['param_memory'])}</td>
            <td style="text-align:right;">{fmt_mem_new(r['activation_current'])} 
            / {fmt_mem_new(r['activation_peak'])}</td>
            <td style="text-align:right;">{fmt_mem_new(r['gradient_current'])} 
            / {fmt_mem_new(r['gradient_peak'])}</td>
            <td style="text-align:right;">{fmt_mem_new(r['total_current_memory'])}</td>
            <td style="text-align:right;">{r['pct']:.1f}%</td>
        </tr>
        """

    if other["total_current_memory"] > 0:
        html += f"""
        <tr style="color:gray;">
            <td>Other Layers</td>
            <td style="text-align:right;">{fmt_mem_new(other['param_memory'])}</td>
            <td style="text-align:right;">{fmt_mem_new(other['activation_current'])} 
            / {fmt_mem_new(other['activation_peak'])}</td>
            <td style="text-align:right;">{fmt_mem_new(other['gradient_current'])} 
            / {fmt_mem_new(other['gradient_peak'])}</td>
            <td style="text-align:right;">{fmt_mem_new(other['total_current_memory'])}</td>
            <td style="text-align:right;">{other['pct']:.1f}%</td>
        </tr>
        """

    html += "</tbody></table>"
    panel["table"].content = html


