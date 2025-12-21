from nicegui import ui

from traceml.renderers.utils import fmt_time_run

def build_step_timing_table_section():
    card = ui.card().classes("m-2 p-4 w-full")
    card.style("""
        height: 350px;
        min-height: 350px;
        max-height: 350px;
        display: flex;
        flex-direction: column;
        background: #ffffff;
        backdrop-filter: blur(12px);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    """)

    with card:
        ui.label("Step Timings") \
            .classes("text-lg font-bold mb-2") \
            .style("color:#d47a00;")

        container = ui.html("", sanitize=False).style("""
            height: 260px;
            overflow-y: auto;
            width: 100%;
            padding-right: 12px;
        """)

    return {"table": container}


def update_step_timing_table_section(panel, dashboard_data):
    if not dashboard_data:
        panel["table"].content = """
        <div style="
            text-align:center;
            padding:16px;
            color:#888;
            font-style:italic;
        ">
            No step timing data detected.<br/>
            Run at least one training step.
        </div>
        """
        return

    rows = []
    for name, vals in dashboard_data.items():
        gpu_avg = vals.get("gpu_avg_s", 0.0)
        gpu_peak = vals.get("gpu_max_s", 0.0)
        cpu_avg = vals.get("cpu_avg_s", 0.0)
        cpu_peak = vals.get("cpu_max_s", 0.0)

        if gpu_peak > 0 or gpu_avg > 0:
            avg, peak, device = gpu_avg, gpu_peak, "GPU"
        else:
            avg, peak, device = cpu_avg, cpu_peak, "CPU"

        rows.append({
            "step": name,
            "avg": avg,
            "peak": peak,
            "device": device,
        })

    # same behavior as other tables: sorted, stable
    rows.sort(key=lambda r: r["peak"], reverse=True)

    html = """
    <table style="width:100%; border-collapse: collapse; font-size:14px;">
        <thead style="position: sticky; top: 0; background: #f0f0f0; z-index:1;">
            <tr>
                <th style="text-align:left; padding:4px 8px;">Step</th>
                <th style="text-align:right; padding:4px 8px;">Avg</th>
                <th style="text-align:right; padding:4px 12px;">Peak</th>
                <th style="text-align:right; padding:4px 12px;">Device</th>
            </tr>
        </thead>
        <tbody>
    """

    for r in rows:
        html += f"""
        <tr>
            <td style="padding:4px 8px;">
                {r["step"]}
            </td>
            <td style="text-align:right; padding:4px 8px;">
                {fmt_time_run(r["avg"])}
            </td>
            <td style="text-align:right; padding:4px 12px;">
                {fmt_time_run(r["peak"])}
            </td>
            <td style="text-align:right; padding:4px 12px;">
                {r["device"]}
            </td>
        </tr>
        """

    html += """
        </tbody>
    </table>
    """

    panel["table"].content = html
