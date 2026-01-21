from nicegui import ui
import plotly.graph_objects as go
from traceml.utils.formatting import fmt_mem_new
from traceml.renderers.display.nicegui_sections.helper import extract_time_axis


METRIC_TEXT = "text-sm leading-normal text-gray-700"
METRIC_TITLE = "text-l font-bold mb-1 ml-1 break-words whitespace-normal"

# --- compact metric tile (same style as System section) ---
LABEL = "text-[11px] font-semibold tracking-wide leading-tight"
VAL   = "text-[12.5px] text-gray-700 leading-tight"
SUB   = "text-[11px] text-gray-500 leading-tight"


def _tile(title, title_color=None, shaded=False):
    """
    Compact tile: title + one value line + optional subtle subline.
    Designed to fit 2 rows x 3 cols under a 150px graph inside 350px card.
    """
    cls = "w-full px-1 py-1"
    box = ui.column().classes(cls).style("min-height: 46px;")  # keeps grid aligned but not tall

    if shaded:
        box.classes("rounded-md").style("background: rgba(0,0,0,0.02);")

    with box:
        t = ui.html(title, sanitize=False).classes(LABEL)
        if title_color:
            t.style(f"color:{title_color};")
        v = ui.html("–", sanitize=False).classes(VAL)
        s = ui.html("", sanitize=False).classes(SUB)

    return box, v, s


def _build_graph_section():
    """
    Graph: RAM % on left axis, GPU Mem % on right axis.
    """
    fig = go.Figure()
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=2, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(
            range=[0, 100],
            title=dict(text="RAM (%)", font=dict(color="#4caf50")),
            tickfont=dict(color="#4caf50"),
        ),
        yaxis2=dict(
            range=[0, 100],
            overlaying="y",
            side="right",
            title=dict(text="GPU Mem (%)", font=dict(color="#ff9800")),
            tickfont=dict(color="#ff9800"),
        ),
        showlegend=False,
    )
    return ui.plotly(fig).classes("w-full")



def build_process_section():
    card = ui.card().classes("m-2 p-2 w-full")
    card.style(
        """
        background: ffffff;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        height: 350px;
        overflow: hidden;
        """
    )

    with card:
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Process Metrics").classes(METRIC_TITLE).style("color:#d47a00;")
            window_text = ui.html("window: –", sanitize=False).classes("text-xs text-gray-500 mr-1")

        graph = _build_graph_section()

        # 2 rows x 3 columns: A B C / D E F
        with ui.grid(columns=3).classes("w-full gap-1 mt-1"):
            # A
            _, cpu_v, cpu_s = _tile("CPU (now/p50/p95)", title_color="#ff9800")
            # B
            _, ram_v, ram_s = _tile("RAM (now/p95/headroom)", title_color="#ff9800")
            # C
            _, gmem_v, gmem_s = _tile("GPU Mem (now/p95) (total)", title_color="#ff9800")

            # D
            _, ranks_v, ranks_s = _tile("Ranks (seen/stale)", title_color="#ff9800")
            # E
            _, oomh_v, oomh_s = _tile("OOM Headroom (worst rank)", title_color="#ff9800")
            # F
            _, imb_v, imb_s = _tile("GPU Mem Imbalance (max-min)", title_color="#ff9800")

    return {
        "window_text": window_text,
        "graph": graph,
        "cpu_v": cpu_v, "cpu_s": cpu_s,
        "ram_v": ram_v, "ram_s": ram_s,
        "gmem_v": gmem_v, "gmem_s": gmem_s,
        "ranks_v": ranks_v, "ranks_s": ranks_s,
        "oomh_v": oomh_v, "oomh_s": oomh_s,
        "imb_v": imb_v, "imb_s": imb_s,
    }


# Rollups (last N samples)
def _percentile(values, p: float, default=0.0) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return default
    vals.sort()
    if len(vals) == 1:
        return float(vals[0])
    k = (len(vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if c == f:
        return float(vals[f])
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return float(d0 + d1)


def _last_n(table, n=100):
    if not table:
        return []
    return table[-n:] if len(table) > n else table


def _compute_rollups_from_process_table(process_table, n=100):
    """
    Process table is local history (rank 0). This function computes rolling stats
    for the time-series tiles + graph (now/p50/p95 + headroom).
    """
    window = _last_n(process_table, n=n)
    last = window[-1] if window else {}

    # CPU (already percent)
    cpu_hist = [float(r.get("cpu_percent", 0.0) or 0.0) for r in window]
    cpu_now = float(last.get("cpu_percent", 0.0) or 0.0)
    cpu_p50 = _percentile(cpu_hist, 50, default=0.0)
    cpu_p95 = _percentile(cpu_hist, 95, default=0.0)

    # RAM (RSS bytes)
    ram_total = float(last.get("ram_total", 0.0) or 0.0)
    ram_used_hist = [float(r.get("ram_used", 0.0) or 0.0) for r in window]
    ram_now = float(last.get("ram_used", 0.0) or 0.0)
    ram_p95 = _percentile(ram_used_hist, 95, default=0.0)
    ram_headroom = max(ram_total - ram_now, 0.0) if ram_total else 0.0

    # GPU mem (process-local, single device)
    gpu_available = bool(last.get("gpu_available", False))
    g_total = float(last.get("gpu_mem_total", 0.0) or 0.0)
    g_used_hist = [float(r.get("gpu_mem_used", 0.0) or 0.0) for r in window]
    g_used_now = float(last.get("gpu_mem_used", 0.0) or 0.0)
    g_used_p95 = _percentile(g_used_hist, 95, default=0.0)
    g_headroom = max(g_total - g_used_now, 0.0) if g_total else 0.0

    return {
        "cpu": {"now": cpu_now, "p50": cpu_p50, "p95": cpu_p95},
        "ram": {"used_now": ram_now, "used_p95": ram_p95, "total": ram_total, "headroom": ram_headroom},
        "gpu": {"available": gpu_available, "used_now": g_used_now, "used_p95": g_used_p95, "total": g_total, "headroom": g_headroom},
    }


# -------------------------
# Update
# -------------------------
def update_process_section(panel, data, window_n=100, stale_after_s=5.0):
    """
    Update Process section from ProcessRenderer dashboard payload.

    Expected `data` from ProcessRenderer.get_dashboard_renderable():
      {
        "cpu_used": ..., "ram_used": ..., "ram_total": ...,
        "gpu_used": ..., "gpu_reserved": ..., "gpu_total": ...,
        "gpu_used_imbalance": ...,
        "n_ranks": ...,
        "table": <deque of local samples>,
        "remote_last_seen": {rank: last_seen_epoch, ...}   # optional
      }
    """
    data = data or {}
    process_table = data.get("table", None) or []
    if not process_table:
        panel["window_text"].content = "window: –"
        return

    # Window label (for the graph + now/p50/p95 rollups)
    panel["window_text"].content = f"window: last {min(window_n, len(process_table))} samples"

    # Rolling (rank-0 local history)
    roll = _compute_rollups_from_process_table(process_table, n=window_n)

    # A: CPU now/p50/p95 (use rolling series; more stable than single snapshot)
    cpu = roll["cpu"]
    panel["cpu_v"].content = f"<b>{cpu['now']:.0f}%</b> / {cpu['p50']:.0f}% / {cpu['p95']:.0f}%"
    panel["cpu_s"].content = ""

    # B: RAM RSS now/p95/headroom (total)
    ram = roll["ram"]
    if ram["total"] > 0:
        panel["ram_v"].content = (
            f"<b>{fmt_mem_new(ram['used_now'])}</b>/"
            f"{fmt_mem_new(ram['used_p95'])} "
            f"({fmt_mem_new(ram['total'])})"
        )
        panel["ram_s"].content = ""
    else:
        panel["ram_v"].content = "–"
        panel["ram_s"].content = ""

    # C: GPU mem now/p95/headroom (total) — local history
    g = roll["gpu"]
    if not g["available"] or g["total"] <= 0:
        panel["gmem_v"].content = "Not available"
        panel["gmem_s"].content = ""
    else:
        panel["gmem_v"].content = (
            f"<b>{fmt_mem_new(g['used_now'])}</b>/"
            f"{fmt_mem_new(g['used_p95'])} "
            f"({fmt_mem_new(g['total'])})"
        )
        panel["gmem_s"].content = ""

    # D: ranks (seen / stale)
    n_ranks = int(data.get("n_ranks", 1) or 1)
    remote_last_seen = data.get("remote_last_seen", {}) or {}

    # remote_last_seen excludes local rank 0, so "seen" should include local.
    seen = 1 + len(remote_last_seen) if n_ranks > 1 else 1

    stale = 0
    if remote_last_seen:
        now = __import__("time").time()
        stale = sum(1 for _, ts in remote_last_seen.items() if (now - float(ts)) > float(stale_after_s))

    panel["ranks_v"].content = f"<b>{seen}</b> seen"
    panel["ranks_s"].content = f"stale: {stale}" if n_ranks > 1 else ""

    # E: OOM headroom (worst rank) — use aggregated snapshot fields from ProcessRenderer
    # data["gpu_used"] / ["gpu_total"] are "max across ranks" when DDP.
    used_worst = data.get("gpu_used", None)
    total_worst = data.get("gpu_total", None)
    if used_worst is None or total_worst is None or float(total_worst) <= 0:
        panel["oomh_v"].content = "–"
        panel["oomh_s"].content = ""
    else:
        headroom = max(float(total_worst) - float(used_worst), 0.0)
        panel["oomh_v"].content = f"<b>{fmt_mem_new(headroom)}</b>"
        panel["oomh_s"].content = "max rank headroom"

    # F: GPU mem imbalance (max-min) — aggregated field from ProcessRenderer
    imb = data.get("gpu_used_imbalance", None)
    if imb is None:
        panel["imb_v"].content = "–"
        panel["imb_s"].content = ""
    else:
        panel["imb_v"].content = f"<b>{fmt_mem_new(float(imb))}</b>"
        panel["imb_s"].content = "across ranks"

    # Graph update (rolling window, local process table)
    _update_process_graph(panel, process_table)


def _update_process_graph(panel, process_table):
    if not process_table:
        return

    process_table = _last_n(process_table, n=100)
    x_hist = extract_time_axis(process_table)

    # RAM % (relative to ram_total)
    ram_total = float(process_table[-1].get("ram_total", 0.0) or 0.0) or 1.0
    ram_pct = [min(max((float(r.get("ram_used", 0.0) or 0.0) / ram_total) * 100.0, 0.0), 100.0) for r in process_table]

    # GPU mem % (relative to gpu_mem_total)
    gpu_available = bool(process_table[-1].get("gpu_available", False))
    gpu_total = float(process_table[-1].get("gpu_mem_total", 0.0) or 0.0) or 1.0
    if gpu_available:
        gpu_pct = [
            min(max((float(r.get("gpu_mem_used", 0.0) or 0.0) / gpu_total) * 100.0, 0.0), 100.0)
            for r in process_table
        ]
    else:
        gpu_pct = []

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=ram_pct,
            x=x_hist,
            mode="lines",
            name="RAM",
            yaxis="y",
            line=dict(color="#4caf50"),
        )
    )

    if gpu_available:
        fig.add_trace(
            go.Scatter(
                y=gpu_pct,
                x=x_hist,
                mode="lines",
                name="GPU Mem",
                yaxis="y2",
                line=dict(color="#ff9800"),
            )
        )

    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=2, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        xaxis=dict(showgrid=False, tickangle=-30, tickmode="auto", nticks=8),
        showlegend=False,
        yaxis=dict(
            range=[0, 100],
            title=dict(text="RAM (%)", font=dict(color="#4caf50")),
            tickfont=dict(color="#4caf50"),
        ),
    )

    if gpu_available:
        fig.update_layout(
            yaxis2=dict(
                range=[0, 100],
                overlaying="y",
                side="right",
                title=dict(text="GPU Mem (%)", font=dict(color="#ff9800")),
                tickfont=dict(color="#ff9800"),
            )
        )

    panel["graph"].update_figure(fig)
