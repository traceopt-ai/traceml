from nicegui import ui
from traceml.renderers.utils import fmt_time_run


def build_step_timing_table_section():
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
        ui.label("Trace Timers").classes("text-l font-bold mb-2").style(
            "color:#d47a00;"
        )

        container = ui.html("", sanitize=False).style(
            """
            height: 350px;
            overflow-y: auto;
            width: 100%;
            padding-right: 12px;
            """
        )

    return {"table": container}


def _prepare_step_timing_rows(dashboard_data):
    """
    Normalize dashboard input to a stable, sorted list of rows.
    """
    if not dashboard_data:
        return []

    rows = [
        {
            "step": r["name"],
            "last": r["last"],
            "p50": r["p50_100"],
            "avg": r["avg_100"],
            "p95": r["p95_100"],
            "trend": r["trend"],
            "device": r["device"],
        }
        for r in dashboard_data
    ]
    rows.sort(key=lambda r: float(r["p95"]), reverse=True)
    return rows


def update_step_timing_table_section(panel, dashboard_data):
    """
    Update the Step Timing table using the new StepTimerRenderer output.
    """
    rows = _prepare_step_timing_rows(dashboard_data)

    if not rows:
        panel[
            "table"
        ].content = """
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

    html = """
    <table style="width:100%; border-collapse: collapse; font-size:14px;">
        <thead style="position: sticky; top: 0; background: #f0f0f0; z-index:1;">
            <tr>
            <th style="text-align:left; padding:4px 8px;">Step</th>
            <th style="text-align:right; padding:4px 8px;">Last</th>
            <th style="text-align:right; padding:4px 8px;">p50(100)</th>
            <th style="text-align:right; padding:4px 8px;">Avg(100)</th>
            <th style="text-align:right; padding:4px 12px;">p95(100)</th>
            <th style="text-align:center; padding:4px 12px;">Trend</th>
            <th style="text-align:center; padding:4px 12px;">Device</th>
            </tr>
        </thead>
        <tbody>
    """

    for r in rows:
        html += f"""
        <tr>
            <td style="padding:4px 8px;">{r["step"]}</td>
            <td style="text-align:right; padding:4px 8px;">
                {fmt_time_run(r["last"])}
            </td>
            <td style="text-align:right; padding:4px 8px;">
                {fmt_time_run(r["p50"])}
            </td>
            <td style="text-align:right; padding:4px 8px;">
                {fmt_time_run(r["avg"])}
            </td>
            <td style="text-align:right; padding:4px 12px;">
                {fmt_time_run(r["p95"])}
            </td>
            <td style="text-align:center; padding:4px 12px;">
                {r["trend"]}
            </td>
            <td style="text-align:center; padding:4px 12px;">
                {r["device"]}
            </td>
        </tr>
        """

    html += """
        </tbody>
    </table>
    """

    panel["table"].content = html
