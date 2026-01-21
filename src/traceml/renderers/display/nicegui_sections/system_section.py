from nicegui import ui
import plotly.graph_objects as go
from traceml.utils.formatting import fmt_mem_new
from traceml.renderers.display.nicegui_sections.helper import extract_time_axis

METRIC_TEXT = "text-sm leading-normal text-gray-700"
METRIC_TITLE = "text-l font-bold mb-1 ml-1 break-words whitespace-normal"


# --- compact metric tile ---
LABEL = "text-[11px] font-semibold tracking-wide leading-tight"
VAL   = "text-[12.5px] text-gray-700 leading-tight"
SUB   = "text-[11px] text-gray-500 leading-tight"

def _tile(title, title_color=None, shaded=False):
    """
    Tighter tile: minimal padding + tight line-heights.
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

        # keep subline but make it compact (and it can be left empty)
        s = ui.html("", sanitize=False).classes(SUB)

    return box, v, s

# --- graph ---
def _build_graph_section():
    fig = go.Figure()
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=2, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.05)",
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(
            range=[0, 100],
            title=dict(text="CPU (%)", font=dict(color="#4caf50")),
            tickfont=dict(color="#4caf50"),
        ),
        yaxis2=dict(
            range=[0, 100],
            overlaying="y",
            side="right",
            title=dict(text="GPU (%)", font=dict(color="#ff9800")),
            tickfont=dict(color="#ff9800"),
        ),
        showlegend=False,
    )
    return ui.plotly(fig).classes("w-full")



def build_system_section():
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
            ui.label("System Metrics").classes(METRIC_TITLE).style("color:#d47a00;")
            window_text = ui.html("window: –", sanitize=False).classes("text-xs text-gray-500 mr-1")

        graph = _build_graph_section()

        # 2 rows x 3 columns: A B C / D E F
        with ui.grid(columns=3).classes("w-full gap-1 mt-1"):
            # A
            _, cpu_v, cpu_s = _tile("CPU (now/p50/p95)", title_color="#ff9800")
            # B
            _, gpu_v, gpu_s = _tile("GPU Util (now/p50/p95)", title_color="#ff9800")
            # C
            _, imb_v, imb_s = _tile("GPU Util Imbalance (now/p95)", title_color="#ff9800")
            # D
            _, ram_v, ram_s = _tile("RAM (now/p95/headroom) (total)", title_color="#ff9800")
            # E
            _, gmem_v, gmem_s = _tile("GPU Mem (now/p95/headroom) (total)", title_color="#ff9800")
            # F
            _, temp_v, temp_s = _tile("Temp (max GPU)", title_color="#ff9800")

    return {
        "window_text": window_text,
        "graph": graph,
        "cpu_v": cpu_v, "cpu_s": cpu_s,
        "gpu_v": gpu_v, "gpu_s": gpu_s,
        "imb_v": imb_v, "imb_s": imb_s,
        "ram_v": ram_v, "ram_s": ram_s,
        "gmem_v": gmem_v, "gmem_s": gmem_s,
        "temp_v": temp_v, "temp_s": temp_s,
    }

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

def _gpu_utils_from_rec(rec):
    gpu_raw = rec.get("gpu_raw", {}) or {}
    return [float(v.get("util", 0.0) or 0.0) for v in gpu_raw.values()]

def _gpu_mem_used_from_rec(rec):
    gpu_raw = rec.get("gpu_raw", {}) or {}
    return [float(v.get("mem_used", 0.0) or 0.0) for v in gpu_raw.values()]

def _gpu_mem_total_from_rec(rec):
    gpu_raw = rec.get("gpu_raw", {}) or {}
    return [float(v.get("mem_total", 0.0) or 0.0) for v in gpu_raw.values()]

def _gpu_temp_from_rec(rec):
    gpu_raw = rec.get("gpu_raw", {}) or {}
    return [float(v.get("temperature", 0.0) or 0.0) for v in gpu_raw.values()]

def _compute_rollups(system_table, n=100):
    window = _last_n(system_table, n=n)
    last = window[-1] if window else {}

    # CPU
    cpu_hist = [float(r.get("cpu_percent", 0.0) or 0.0) for r in window]
    cpu_now = float(last.get("cpu_percent", 0.0) or 0.0)
    cpu_p50 = _percentile(cpu_hist, 50, default=0.0)
    cpu_p95 = _percentile(cpu_hist, 95, default=0.0)

    # RAM
    ram_total = float(last.get("ram_total", 0.0) or 0.0)
    ram_used_hist = [float(r.get("ram_used", 0.0) or 0.0) for r in window]
    ram_used_now = float(last.get("ram_used", 0.0) or 0.0)
    ram_used_p95 = _percentile(ram_used_hist, 95, default=0.0)
    ram_headroom = max(ram_total - ram_used_now, 0.0) if ram_total else 0.0

    # GPU availability
    gpu_available = bool(last.get("gpu_available", False))
    gpu_count = int(last.get("gpu_count", 0) or 0)

    # GPU util avg + imbalance
    gpu_avg_hist, gpu_delta_hist = [], []
    for r in window:
        utils = _gpu_utils_from_rec(r)
        if utils:
            gpu_avg_hist.append(sum(utils) / max(len(utils), 1))
            gpu_delta_hist.append(max(utils) - min(utils))
        else:
            gpu_avg_hist.append(0.0)
            gpu_delta_hist.append(0.0)

    gpu_util_now = gpu_avg_hist[-1] if gpu_avg_hist else 0.0
    gpu_util_p50 = _percentile(gpu_avg_hist, 50, default=0.0)
    gpu_util_p95 = _percentile(gpu_avg_hist, 95, default=0.0)
    gpu_delta_now = gpu_delta_hist[-1] if gpu_delta_hist else 0.0
    gpu_delta_p95 = _percentile(gpu_delta_hist, 95, default=0.0)

    # GPU memory (worst GPU) + headroom
    gpu_worst_mem_hist = []
    gpu_total_now = 0.0
    for r in window:
        mems = _gpu_mem_used_from_rec(r)
        tots = _gpu_mem_total_from_rec(r)
        if tots:
            gpu_total_now = max(tots)  # safe
        gpu_worst_mem_hist.append(max(mems) if mems else 0.0)

    gpu_mem_worst_now = gpu_worst_mem_hist[-1] if gpu_worst_mem_hist else 0.0
    gpu_mem_worst_p95 = _percentile(gpu_worst_mem_hist, 95, default=0.0)
    gpu_mem_headroom = max(gpu_total_now - gpu_mem_worst_now, 0.0) if gpu_total_now else 0.0

    # Temp (max GPU)
    temp_max_hist = []
    for r in window:
        temps = _gpu_temp_from_rec(r)
        temp_max_hist.append(max(temps) if temps else 0.0)
    temp_now = temp_max_hist[-1] if temp_max_hist else 0.0
    temp_p95 = _percentile(temp_max_hist, 95, default=0.0)

    if temp_now >= 85:
        temp_status = "Hot"
    elif temp_now >= 80:
        temp_status = "Warm"
    else:
        temp_status = "OK"

    return {
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "cpu": {"now": cpu_now, "p50": cpu_p50, "p95": cpu_p95},
        "gpu_util": {"now": gpu_util_now, "p50": gpu_util_p50, "p95": gpu_util_p95},
        "gpu_util_delta": {"now": gpu_delta_now, "p95": gpu_delta_p95},
        "ram": {"used_now": ram_used_now, "used_p95": ram_used_p95, "total": ram_total, "headroom": ram_headroom},
        "gpu_mem_worst": {
            "used_now": gpu_mem_worst_now,
            "used_p95": gpu_mem_worst_p95,
            "total": gpu_total_now,
            "headroom": gpu_mem_headroom,
        },
        "temp": {"now": temp_now, "p95": temp_p95, "status": temp_status},
    }


def update_system_section(panel, data, window_n=100):
    system_table = (data or {}).get("table", None) or []
    if not system_table:
        panel["window_text"].content = "window: –"
        return

    roll = _compute_rollups(system_table, n=window_n)
    panel["window_text"].content = f"window: last {min(window_n, len(system_table))} samples"

    # A: CPU
    cpu = roll["cpu"]
    panel["cpu_v"].content = f"<b>{cpu['now']:.0f} / {cpu['p50']:.0f}% / {cpu['p95']:.0f}%"
    panel["cpu_s"].content = ""

    # D: RAM
    ram = roll["ram"]
    if ram["total"] > 0:
        pct_now = (ram["used_now"] * 100.0) / ram["total"]
        panel["ram_v"].content = (
            f"<b>{fmt_mem_new(ram['used_now'])}</b>/"
            f"{fmt_mem_new(ram['used_p95'])}/"
            f"{fmt_mem_new(ram['headroom'])} "
            f"({fmt_mem_new(ram['total'])})"
        )
        panel["ram_s"].content = ""
    else:
        panel["ram_v"].content = "–"
        panel["ram_s"].content = ""

    # GPU tiles (B, C, E, F)
    if not roll["gpu_available"]:
        panel["gpu_v"].content = "Not available"
        panel["gpu_s"].content = ""
        panel["imb_v"].content = "Not available"
        panel["imb_s"].content = ""
        panel["gmem_v"].content = "Not available"
        panel["gmem_s"].content = ""
        panel["temp_v"].content = "Not available"
        panel["temp_s"].content = ""
    else:
        # B: GPU util
        gu = roll["gpu_util"]
        panel["gpu_v"].content = f"<b>{gu['now']:.0f}%</b> / {gu['p50']:.0f}% / {gu['p95']:.0f}%"
        panel["gpu_s"].content = ""

        # C: Imbalance
        gd = roll["gpu_util_delta"]
        panel["imb_v"].content = f"{gd['now']:.0f}%</b> / {gd['p95']:.0f}%"
        panel["imb_s"].content = ""

        # E: GPU mem (worst)
        gm = roll["gpu_mem_worst"]
        if gm["total"] > 0:
            panel["gmem_v"].content = (
                f"<b>{fmt_mem_new(gm['used_now'])}</b>/"
                f"{fmt_mem_new(gm['used_p95'])}/"
                f"{fmt_mem_new(gm['headroom'])}  "
                f"({fmt_mem_new(gm['total'])})"
            )
            panel["gmem_s"].content = ""
        else:
            panel["gmem_v"].content = "–"
            panel["gmem_s"].content = ""

        # F: Temp (max) — keep for throttling
        t = roll["temp"]
        panel["temp_v"].content = f"Now <b>{t['now']:.0f}°C</b> · P95 {t['p95']:.0f}°C"
        panel["temp_s"].content = f"Status: {t['status']}"

    # Graph update
    _update_graph_section(panel, system_table)

def _update_graph_section(panel, system_table):
    if not system_table:
        return

    system_table = _last_n(system_table, n=100)
    x_hist = extract_time_axis(system_table)

    fig = go.Figure()

    # CPU
    cpu_hist = [rec.get("cpu_percent", 0) or 0 for rec in system_table]
    fig.add_trace(
        go.Scatter(
            y=cpu_hist,
            x=x_hist,
            mode="lines",
            name="CPU",
            yaxis="y",
            line=dict(color="#4caf50"),
        )
    )

    # GPU (avg util)
    gpu_available = bool(system_table[-1].get("gpu_available", False))
    if gpu_available:
        gpu_avg_hist = []
        for rec in system_table:
            utils = _gpu_utils_from_rec(rec)
            gpu_avg_hist.append((sum(utils) / max(len(utils), 1)) if utils else 0.0)

        fig.add_trace(
            go.Scatter(
                y=gpu_avg_hist,
                x=x_hist,
                mode="lines",
                name="GPU",
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
            title=dict(text="CPU (%)", font=dict(color="#4caf50")),
            tickfont=dict(color="#4caf50"),
        ),
    )
    if gpu_available:
        fig.update_layout(
            yaxis2=dict(
                range=[0, 100],
                overlaying="y",
                side="right",
                title=dict(text="GPU (%)", font=dict(color="#ff9800")),
                tickfont=dict(color="#ff9800"),
            )
        )

    panel["graph"].update_figure(fig)
