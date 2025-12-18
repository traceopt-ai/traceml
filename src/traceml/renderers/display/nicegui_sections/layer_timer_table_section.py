from nicegui import ui
from traceml.utils.formatting import fmt_time_ms


def build_layer_timer_table_section():
    card = ui.card().classes("m-2 p-4 w-full")
    card.style("""
        height: 350px;
        display: flex;
        flex-direction: column;

        background: ffffff;
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
            "flex: 1; overflow-y: auto; width: 100%; padding-right: 12px;"
        )

    return {"table": container}


def update_layer_timer_table_section(panel, dashboard_data):
    rows = dashboard_data.get("all_items", [])

    html = """
    <table style="width:100%; border-collapse: collapse; font-size:14px;">
        <thead style="position: sticky; top: 0; background: #f0f0f0; z-index: 1;">
            <tr>
                <th style="text-align:left; padding:4px 8px;">Layer</th>
                <th style="text-align:right; padding:4px 8px;">Activation (curr / peak)</th>
                <th style="text-align:right; padding:4px 12px;">Gradient (curr / peak)</th>
                <th style="text-align:right; padding:4px 12px;">%</th>
            </tr>
        </thead>
        <tbody>
    """

    if rows:
        for r in rows:
            html += f"""
            <tr>
                <td style="padding:4px 8px;">
                    {r.get("layer", "—")}
                </td>
                <td style="text-align:right; padding:4px 8px;">
                    {fmt_time_ms(r.get("activation_current_ms", 0.0))} /
                    {fmt_time_ms(r.get("activation_peak_ms", 0.0))}
                </td>
                <td style="text-align:right; padding:4px 12px;">
                    {fmt_time_ms(r.get("gradient_current_ms", 0.0))} /
                    {fmt_time_ms(r.get("gradient_peak_ms", 0.0))}
                </td>
                <td style="text-align:right; padding:4px 12px;">
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
                No per-layer timing data detected.<br/>
                Ensure timing hooks are attached and the model has executed at least one forward pass.
            </td>
        </tr>
        """

    html += """
        </tbody>
    </table>
    """

    panel["table"].content = html

