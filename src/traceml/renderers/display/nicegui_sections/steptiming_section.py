from nicegui import ui


def build_step_timing_table_section():
    card = ui.card().classes("m-2 p-4 w-full")
    card.style("""
        background: ffffff;
        backdrop-filter: blur(12px);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    """)

    with card:
        ui.label("Step Timings").classes("text-lg font-bold mb-2").style("color:#d47a00;")
        container = ui.html("", sanitize=False).style(
            "max-height: 350px; overflow-y: auto; width: 100%;"
        )

    return {"table": container}


def update_step_timing_table_section(panel, dashboard_data):
    """
    dashboard_data comes from StepTimerRenderer.get_dashboard_renderable()
    """

    if not dashboard_data:
        panel["table"].content = "<i>No step timings recorded</i>"
        return

    # Convert to rows with display values
    rows = []
    for name, vals in dashboard_data.items():
        gpu_avg = vals.get("gpu_avg_s", 0.0)
        gpu_peak = vals.get("gpu_max_s", 0.0)
        cpu_avg = vals.get("cpu_avg_s", 0.0)
        cpu_peak = vals.get("cpu_max_s", 0.0)

        if gpu_peak > 0 or gpu_avg > 0:
            avg = gpu_avg
            peak = gpu_peak
            device = "GPU"
        else:
            avg = cpu_avg
            peak = cpu_peak
            device = "CPU"

        rows.append({
            "event": name,
            "avg": avg,
            "peak": peak,
            "device": device,
        })

    # Sort by PEAK time (descending)
    rows.sort(key=lambda r: r["peak"], reverse=True)

    # Build HTML table
    html = """
    <table style="width:100%; border-collapse: collapse; font-size:14px;">
        <thead style="position: sticky; top: 0; background: #f0f0f0;">
            <tr>
                <th style="text-align:left;">Step</th>
                <th style="text-align:right;">Avg (s)</th>
                <th style="text-align:right;">Peak (s)</th>
                <th style="text-align:right;">Device</th>
            </tr>
        </thead>
        <tbody>
    """

    for r in rows:
        html += f"""
        <tr>
            <td>{r['event']}</td>
            <td style="text-align:right;">{r['avg']:.4f}</td>
            <td style="text-align:right; font-weight:600;">
                {r['peak']:.4f}
            </td>
            <td style="text-align:right; color:gray;">
                {r['device']}
            </td>
        </tr>
        """

    html += "</tbody></table>"
    panel["table"].content = html
