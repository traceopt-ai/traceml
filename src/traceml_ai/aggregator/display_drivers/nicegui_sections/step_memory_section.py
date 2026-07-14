"""Step Memory analysis — peak/worst-vs-median GB over steps (PR2 revamp).

Reuses the original metric normalization + stats; renders with glass + ECharts.
Renderer payload (metrics[].series + summary + coverage) is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nicegui import ui

from traceml_ai.diagnostics.trends import compute_trend_pct

from . import theme

BYTES_PER_GB = 1e9


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_list(v: Any) -> List[Any]:
    if v is None:
        return []
    try:
        return list(v)
    except Exception:
        return []


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(float(x) for x in xs)
    if len(ys) == 1:
        return ys[0]
    pos = (q / 100.0) * (len(ys) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    frac = pos - lo
    return ys[lo] * (1.0 - frac) + ys[hi] * frac


def _avg_last_k(xs: List[float], k: int = 100) -> float:
    if not xs:
        return 0.0
    tail = xs[-k:] if len(xs) > k else xs
    return float(sum(float(x) for x in tail) / max(1, len(tail)))


def _gb_list(values: List[float]) -> List[float]:
    return [float(v) / BYTES_PER_GB for v in values] if values else []


@dataclass(frozen=True)
class _Stats:
    last: float
    p50: float
    p95: float
    avg100: float
    trend: float


@dataclass(frozen=True)
class _View:
    steps: List[int]
    worst_gb: List[float]
    median_gb: List[float]
    worst: _Stats
    median: _Stats
    skew_pct: float
    worst_rank: Optional[int]
    single_rank: bool


def _mem_options() -> Dict[str, Any]:
    grid = "rgba(17,24,39,0.05)"
    axis = "rgba(17,24,39,0.14)"
    txt = "#9aa3af"

    def ln(name, col, dash=False):
        return {
            "name": name,
            "type": "line",
            "smooth": True,
            "showSymbol": False,
            "lineStyle": {
                "width": 2.2 if not dash else 1.6,
                "color": col,
                "type": "dashed" if dash else "solid",
                "shadowColor": "rgba(17,24,39,0.16)",
                "shadowBlur": 4,
                "shadowOffsetY": 2,
            },
            "endLabel": {
                "show": not dash,
                "color": col,
                "fontFamily": "Geist Mono",
                "fontSize": 10,
                "fontWeight": 600,
                ":formatter": "p=>p.value.toFixed(2)+' GB'",
            },
            "areaStyle": (
                {
                    "color": {
                        "type": "linear",
                        "x": 0,
                        "y": 0,
                        "x2": 0,
                        "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "rgba(255,140,0,0.20)"},
                            {"offset": 1, "color": "rgba(255,255,255,0)"},
                        ],
                    }
                }
                if not dash
                else None
            ),
            "data": [],
        }

    return {
        "backgroundColor": "transparent",
        "animationDuration": 600,
        "color": [theme.ORANGE, theme.C_CPU],
        "grid": {
            "left": 4,
            "right": 34,
            "top": 12,
            "bottom": 4,
            "containLabel": True,
        },
        "tooltip": {
            "trigger": "axis",
            "backgroundColor": "rgba(255,253,250,0.97)",
            "borderColor": theme.BORDER,
            "textStyle": {
                "color": theme.INK,
                "fontFamily": "Geist Mono",
                "fontSize": 11,
            },
            "axisPointer": {
                "type": "line",
                "lineStyle": {"color": axis, "type": "dashed"},
            },
        },
        "legend": {"show": False},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": [],
            "axisLabel": {
                "color": txt,
                "fontFamily": "Geist Mono",
                "fontSize": 10,
            },
            "axisLine": {"lineStyle": {"color": axis, "opacity": 0.5}},
            "axisTick": {"show": False},
            "splitLine": {"show": False},
        },
        "yAxis": {
            "type": "value",
            "scale": True,
            "axisLabel": {
                "color": txt,
                "fontFamily": "Geist Mono",
                "fontSize": 10,
                ":formatter": "v=>v.toFixed(1)",
            },
            "axisLine": {"show": False},
            "axisTick": {"show": False},
            "splitLine": {"lineStyle": {"color": grid}},
        },
        "series": [
            ln("Peak", theme.ORANGE),
            ln("Median", theme.C_CPU, dash=True),
        ],
    }


def build_step_memory_section() -> Dict[str, Any]:
    kpis: Dict[str, Any] = {}
    card = ui.element("div").classes("glass reveal")
    card.style(
        "padding:18px 20px; width:100%; height:100%; "
        "display:flex; flex-direction:column; overflow:hidden;"
    )
    with card:
        with (
            ui.row()
            .classes("w-full items-center")
            .style("margin-bottom:8px; gap:12px;")
        ):
            ui.label("Step memory").classes("ctitle")
            ui.element("div").style("flex:1;")
            win = ui.label("waiting for memory").classes("cmeta")
        chart = ui.echart(_mem_options()).style(
            "height:200px; width:100%; flex:1; min-height:160px;"
        )
        hint = ui.label("").style(
            "font-family:var(--mono); font-size:11px; color:var(--muted); text-align:center;"
        )
        with (
            ui.row()
            .classes("w-full")
            .style("gap:9px; margin-top:12px; flex-wrap:wrap;")
        ):
            for key, lab in [
                ("peak", "PEAK"),
                ("p95", "PEAK p95"),
                ("avg", "AVG(100)"),
                ("trend", "TREND"),
            ]:
                with (
                    ui.element("div")
                    .classes("kpi")
                    .style(f"--acc:{theme.ORANGE}; min-width:100px;")
                ):
                    ui.label(lab).classes("klab")
                    kpis[key] = ui.html("—", sanitize=False).classes("kval")
    return {
        "chart": chart,
        "win": win,
        "hint": hint,
        "kpis": kpis,
        "_last": None,
    }


def update_step_memory_section(
    panel: Dict[str, Any],
    payload: Any,
    *,
    prefer_metric: str = "peak_allocated",
) -> None:
    try:
        view = _normalize(_select_metric(payload, prefer_metric))
        if view is None:
            msg = getattr(payload, "status_message", None)
            if not isinstance(msg, str) and isinstance(payload, dict):
                msg = payload.get("status_message")
            if isinstance(msg, str) and "No GPU detected" in msg:
                panel["win"].text = "no GPU"
                panel["hint"].text = (
                    "No GPU present, so step memory is unavailable."
                )
                return
            view = panel.get("_last")
        if view is None:
            return
        panel["_last"] = view
        panel["hint"].text = ""

        chart = panel["chart"]
        cats = [str(s) for s in view.steps]
        chart.options["xAxis"]["data"] = cats
        chart.options["series"][0]["name"] = (
            "Peak" if view.single_rank else "Worst"
        )
        chart.options["series"][0]["data"] = view.worst_gb
        chart.options["series"][1]["data"] = (
            [] if view.single_rank else view.median_gb
        )
        chart.update()

        panel["win"].text = f"{len(view.steps)} aligned steps"
        k = panel["kpis"]
        k["peak"].content = theme.kval(
            f"{view.worst.last / BYTES_PER_GB:.2f}", "GB"
        )
        k["p95"].content = theme.kval(
            f"{view.worst.p95 / BYTES_PER_GB:.2f}", "GB"
        )
        k["avg"].content = theme.kval(
            f"{view.worst.avg100 / BYTES_PER_GB:.2f}", "GB"
        )
        k["trend"].content = theme.kval(f"{view.worst.trend:+.0f}", "%")
    except Exception:
        pass


def _select_metric(payload: Any, preferred: str) -> Optional[Any]:
    metrics = getattr(payload, "metrics", None)
    if metrics is None and isinstance(payload, dict):
        metrics = payload.get("metrics")
    if not metrics:
        return None
    for m in metrics:
        key = (
            m.get("metric")
            if isinstance(m, dict)
            else getattr(m, "metric", None)
        )
        if key == preferred:
            return m
    return metrics[0]


def _stats(values_bytes: List[float]) -> _Stats:
    if not values_bytes:
        return _Stats(0.0, 0.0, 0.0, 0.0, 0.0)
    trend = compute_trend_pct(values_bytes)
    return _Stats(
        float(values_bytes[-1]),
        _percentile(values_bytes, 50.0),
        _percentile(values_bytes, 95.0),
        _avg_last_k(values_bytes, 100),
        float(trend if trend is not None else 0.0),
    )


def _normalize(metric: Any) -> Optional[_View]:
    if metric is None:
        return None
    series = (
        metric.get("series", {})
        if isinstance(metric, dict)
        else getattr(metric, "series", None)
    )
    summary = (
        metric.get("summary", {})
        if isinstance(metric, dict)
        else getattr(metric, "summary", None)
    )

    def _g(obj, attr):
        return (
            obj.get(attr)
            if isinstance(obj, dict)
            else getattr(obj, attr, None)
        )

    steps = _to_list(_g(series, "steps"))
    worst_b = _to_list(_g(series, "worst"))
    median_b = _to_list(_g(series, "median"))
    if not steps or not worst_b or not median_b:
        return None
    n = min(len(steps), len(worst_b), len(median_b))
    if n <= 0:
        return None
    steps = [int(_safe_float(v)) for v in steps[:n]]
    worst_b = [_safe_float(v) for v in worst_b[:n]]
    median_b = [_safe_float(v) for v in median_b[:n]]

    coverage = (
        metric.get("coverage", {})
        if isinstance(metric, dict)
        else getattr(metric, "coverage", None)
    )
    ws = _g(coverage, "world_size")
    rp = _g(coverage, "ranks_present")
    single = bool(_safe_float(ws) <= 1.0 or _safe_float(rp) <= 1.0)
    wr = _g(summary, "worst_rank")
    return _View(
        steps=steps,
        worst_gb=_gb_list(worst_b),
        median_gb=_gb_list(median_b),
        worst=_stats(worst_b),
        median=_stats(median_b),
        skew_pct=_safe_float(_g(summary, "skew_pct")),
        worst_rank=(None if wr is None else int(_safe_float(wr))),
        single_rank=single,
    )
