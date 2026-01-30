from nicegui import ui

from traceml.utils.formatting import fmt_mem_new

METRIC_TEXT = "text-sm leading-normal text-gray-700"
METRIC_TITLE = "text-l font-bold mb-1 ml-1 break-words whitespace-normal"


def build_layer_memory_table_section():
    card = ui.card().classes("m-2 p-4 w-full")
    card.style(
        """
        height: 350px;
        display: flex;
        flex-direction: column;

        background: #ffffff;
        backdrop-filter: blur(12px);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    """
    )

    with card:
        ui.label("Per Layer Memory Stats").classes(METRIC_TITLE).style("color:#d47a00;")
        container = ui.html("", sanitize=False).style(
            "flex: 1; overflow-y: auto; width: 100%; padding-right: 12px;"
        )

    return {"table": container}


def update_layer_memory_table_section(panel, dashboard_data):
    rows = dashboard_data.get("all_items", [])

    html = """
    <table style="width:100%; border-collapse: collapse; font-size:14px;">
        <thead style="position: sticky; top: 0; background: #f0f0f0; z-index:1;">
            <tr>
                <th style="text-align:left; padding:4px 8px;">Layer</th>
                <th style="text-align:right; padding:4px 8px;">Params</th>
                <th style="text-align:right; padding:4px 8px;">Forward (curr / peak)</th>
                <th style="text-align:right; padding:4px 12px;">Backward (curr / peak)</th>
                <th style="text-align:right; padding:4px 12px;">Total Current</th>
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
                    {fmt_mem_new(r.get("param_memory", 0.0))}
                </td>
                <td style="text-align:right; padding:4px 8px;">
                    {fmt_mem_new(r.get("forward_current", 0.0))} /
                    {fmt_mem_new(r.get("forward_peak", 0.0))}
                </td>
                <td style="text-align:right; padding:4px 12px;">
                    {fmt_mem_new(r.get("backward_current", 0.0))} /
                    {fmt_mem_new(r.get("backward_peak", 0.0))}
                </td>
                <td style="text-align:right; padding:4px 12px;">
                    {fmt_mem_new(r.get("total_current_memory", 0.0))}
                </td>
                <td style="text-align:right; padding:4px 12px;">
                    {float(r.get("pct", 0.0)):.1f}%
                </td>
            </tr>
            """
    else:
        html += """
        <tr>
            <td colspan="6"
                style="
                    text-align:center;
                    padding:16px;
                    color:#888;
                    font-style:italic;
                ">
                No per-layer memory data detected.<br/>
                Ensure memory hooks are attached and at least one forward pass has completed.
            </td>
        </tr>
        """

    html += """
        </tbody>
    </table>
    """
    panel["table"].content = html
