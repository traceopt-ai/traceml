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
    rows = dashboard_data.get("all_items", [])

    html = """
    <table style="width:100%; border-collapse: collapse; font-size:14px;">
        <thead style="position: sticky; top: 0; background: #f0f0f0; z-index: 1;">
            <tr>
                <th style="text-align:left; padding:6px;">Layer</th>
                <th style="text-align:right; padding:6px;">Activation (curr / peak)</th>
                <th style="text-align:right; padding:6px;">Gradient (curr / peak)</th>
                <th style="text-align:right; padding:6px;">%</th>
            </tr>
        </thead>
        <tbody>
    """

    if rows:
        for r in rows:
            html += f"""
            <tr>
                <td style="padding:6px;">{r.get("layer", "—")}</td>
                <td style="text-align:right; padding:6px;">
                    {fmt_time_ms(r.get("activation_current_ms", 0.0))} /
                    {fmt_time_ms(r.get("activation_peak_ms", 0.0))}
                </td>
                <td style="text-align:right; padding:6px;">
                    {fmt_time_ms(r.get("gradient_current_ms", 0.0))} /
                    {fmt_time_ms(r.get("gradient_peak_ms", 0.0))}
                </td>
                <td style="text-align:right; padding:6px;">
                    {float(r.get("pct", 0.0)):.1f}%
                </td>
            </tr>
            """
    else:
        html += """
        <tr>
            <td colspan="4"
                style="
                    text-align:center;
                    padding:16px;
                    color:#888;
                    font-style:italic;
                ">
                No timing data detected.<br/>
                Ensure timing hooks are attached and the model has executed at least one step.
            </td>
        </tr>
        """

    html += """
        </tbody>
    </table>
    """

    panel["table"].content = html

