"""
NiceGUI Step Timing Table (Trace Timers)

This section renders StepTimerRenderer output in a compact scrollable table.

Assumptions about input (`dashboard_data`):
  StepTimerRenderer.get_dashboard_renderable() returns:
    [
      {
        "name": str,
        "last": float,
        "p50_100": float,
        "p95_100": float,
        "avg_100": float,
        "trend": str,
        "device": str,
        "worst_rank": str,     # rank id (string) or "—"
        "coverage": str,       # e.g. "6/8" or "1/1" (optional)
        "min_samples": int,    # min samples among ranks that contributed (optional)
      },
      ...
    ]

Semantics:
  Values (last/p50/p95/avg) are aggregated using DDP "worst wins" rules
  in StepTimerRenderer. This UI only displays what the renderer computed.
"""

from typing import Any, Dict, List, Optional

from nicegui import ui

from traceml.renderers.utils import fmt_time_run

# -----------------------------
# Styling (match your dashboard)
# -----------------------------
CARD_STYLE = """
background: #ffffff;
backdrop-filter: blur(12px);
border-radius: 14px;
border: 1px solid rgba(255,255,255,0.25);
box-shadow: 0 4px 12px rgba(0,0,0,0.12);
"""

TITLE_STYLE = "color:#d47a00;"
EMPTY_STYLE = """
text-align:center;
padding:16px;
color:#888;
font-style:italic;
"""

# Table cell padding is intentionally tight to fit 350px cards well.
TH_L = "text-align:left; padding:4px 8px;"
TH_R = "text-align:right; padding:4px 8px;"
TH_C = "text-align:center; padding:4px 8px;"

TD_L = "padding:4px 8px;"
TD_R = "text-align:right; padding:4px 8px;"
TD_C = "text-align:center; padding:4px 8px;"


# Build
def build_step_timing_table_section():
    """
    Build the Step Timing card.

    Returns a dict of UI handles:
      - table: ui.html container that we update with rendered HTML
    """
    card = ui.card().classes("m-2 p-4 w-full")
    card.style(
        f"""
        height: 360px;
        display: flex;
        flex-direction: column;
        {CARD_STYLE}
        """
    )

    with card:
        # Header row: title + small hint
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Trace Timers").classes("text-l font-bold mb-2").style(
                TITLE_STYLE
            )
            ui.html(
                "stats: <b>worst-rank</b> over last 100",
                sanitize=False,
            ).classes("text-xs text-gray-500 mr-1")

        container = ui.html("", sanitize=False).style(
            """
            flex: 1;
            overflow-y: auto;
            width: 100%;
            padding-right: 12px;
            """
        )

    return {"table": container}


# Normalization / Sorting
def _normalize_rows(
    dashboard_data: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Convert StepTimerRenderer output into a stable list of table rows.

    We keep keys consistent and safe for rendering even if optional fields are missing.
    """
    if not dashboard_data:
        return []

    out: List[Dict[str, Any]] = []
    for r in dashboard_data:
        # Backward compatible: some older renderers used "step" instead of "name".
        name = r.get("name") or r.get("step") or "—"

        out.append(
            {
                "name": str(name),
                "last": float(r.get("last", 0.0) or 0.0),
                "p50": float(r.get("p50_100", 0.0) or 0.0),
                "avg": float(r.get("avg_100", 0.0) or 0.0),
                "p95": float(r.get("p95_100", 0.0) or 0.0),
                "trend": str(r.get("trend", "") or ""),
                "device": str(r.get("device", "—") or "—"),
                "worst_rank": str(r.get("worst_rank", "—") or "—"),
                "ranks": str(r.get("coverage", "—") or "—"),
            }
        )

    # Sort: slowest first (p95 is a good bottleneck proxy for training)
    out.sort(key=lambda x: float(x["p95"]), reverse=True)
    return out


# -----------------------------
# HTML Rendering
# -----------------------------
def _render_empty() -> str:
    """Empty state HTML."""
    return f"""
    <div style="{EMPTY_STYLE}">
        No step timing data detected.<br/>
        Run at least one training step.
    </div>
    """


def _render_table(rows: List[Dict[str, Any]]) -> str:
    """
    Render rows as a compact HTML table.

    Columns:
      Event | Last | p50(100) | Avg(100) | p95(100) | Trend | Device | Worst | Cov | MinN
    """
    # Sticky header: keeps labels visible while scrolling.
    html = f"""
    <table style="width:100%; border-collapse: collapse; font-size:13px;">
        <thead style="position: sticky; top: 0; background: #f0f0f0; z-index:1;">
            <tr>
                <th style="{TH_L}">Event</th>
                <th style="{TH_R}">Last</th>
                <th style="{TH_R}">p50(100)</th>
                <th style="{TH_R}">Avg(100)</th>
                <th style="{TH_R}">p95(100)</th>
                <th style="{TH_C}">Trend</th>
                <th style="{TH_C}">Device</th>
                <th style="{TH_C}">Worst</th>
                <th style="{TH_C}">Ranks</th>
            </tr>
        </thead>
        <tbody>
    """

    for r in rows:
        # Trend formatting is intentionally minimal: the renderer already encodes meaning
        # (e.g. "+1.2%", "-0.8%", "≈0%", "! ..."). Keep it readable and stable.
        trend = r["trend"] if r["trend"] else "—"

        html += f"""
        <tr>
            <td style="{TD_L}">{r["name"]}</td>
            <td style="{TD_R}">{fmt_time_run(r["last"])}</td>
            <td style="{TD_R}">{fmt_time_run(r["p50"])}</td>
            <td style="{TD_R}">{fmt_time_run(r["avg"])}</td>
            <td style="{TD_R}">{fmt_time_run(r["p95"])}</td>
            <td style="{TD_C}">{trend}</td>
            <td style="{TD_C}">{r["device"]}</td>
            <td style="{TD_C}">{r["worst_rank"]}</td>
            <td style="{TD_C}">{r["ranks"]}</td>
        </tr>
        """

    html += """
        </tbody>
    </table>
    """
    return html


def update_step_timing_table_section(panel, dashboard_data):
    """
    Update the Step Timing table using StepTimerRenderer output.

    `panel` is the dict returned by build_step_timing_table_section().
    `dashboard_data` is StepTimerRenderer.get_dashboard_renderable().
    """
    rows = _normalize_rows(dashboard_data)

    if not rows:
        panel["table"].content = _render_empty()
        return

    panel["table"].content = _render_table(rows)
