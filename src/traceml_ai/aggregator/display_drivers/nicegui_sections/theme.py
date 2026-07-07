# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
TraceOpt design-system theme for the NiceGUI dashboard (PR2 revamp).

Adopts the shared brand contract (traceopt_website/DESIGN.md,
traceopt-design-system-2026-06-05): light, warm-white, glass surfaces, orange
accent, Geist + Geist Mono. Single source of truth for the dashboard's chrome.
Functional data-viz colors (phase + severity) are kept as-is (they encode
meaning) and pulled from the canonical phase keys, not re-hued.

This module is pure styling + ECharts option builders + small format helpers;
the per-section modules construct their UI with the CSS classes defined here.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

# --- Brand chrome tokens (DESIGN.md) -------------------------------------
BG = "#fffdfa"
INK = "#1f2937"
MUTED = "#6b7280"
ORANGE = "#FF8C00"
ORANGE_STRONG = "#D2691E"
ORANGE_TOP = "#FFAF38"
BORDER = "rgba(17,24,39,0.10)"

# --- Functional data-viz palette (do NOT re-hue) -------------------------
# Canonical phase keys + colors, aligned with the TraceOpt viewer's functional
# palette (traceopt-viewer/frontend/src/index.css) so phase colors match across
# the live dashboard and the viewer. H2D = #ff9800 (the viewer's value), which
# reconciles the prior dashboard-only divergence (it used teal #00897b).
PHASES: List[Tuple[str, str, str]] = [
    ("IW", "input_wait", "#d32f2f"),
    ("H2D", "h2d", "#ff9800"),
    ("FWD", "forward", "#1976d2"),
    ("BWD", "backward", "#512da8"),
    ("OPT", "optimizer_step", "#2e7d32"),
    ("RESIDUAL", "residual_proxy", "#f9a825"),
]
SEV = {
    "crit": "#c62828",
    "warn": "#ef6c00",
    "info": "#b45309",
    "healthy": "#2e7d32",
    "neutral": "#6b7280",
}
# CPU/GPU traces: GPU = brand orange (hero metric), CPU = clean blue.
C_CPU = "#2563eb"
C_GPU = ORANGE

_FONT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts")
_FONT_ROUTE = "/_traceml_fonts"


def register_static_fonts(app: Any) -> None:
    """Serve the bundled Geist woff2 offline (idempotent, best-effort)."""
    try:
        path = os.path.normpath(_FONT_DIR)
        if os.path.isdir(path):
            app.add_static_files(_FONT_ROUTE, path)
    except Exception:
        pass  # best-effort: fonts fall back to the system stack


def head_html() -> str:
    """Full <style> block: offline @font-face + brand chrome + components."""
    return f"""
<style>
@font-face{{font-family:'Geist';font-style:normal;font-weight:100 900;
  src:url('{_FONT_ROUTE}/Geist.woff2') format('woff2');font-display:swap;}}
@font-face{{font-family:'Geist Mono';font-style:normal;font-weight:100 900;
  src:url('{_FONT_ROUTE}/GeistMono.woff2') format('woff2');font-display:swap;}}
:root{{
  --bg:{BG}; --ink:{INK}; --muted:{MUTED}; --orange:{ORANGE};
  --orange-strong:{ORANGE_STRONG}; --tint:rgba(255,140,0,0.10); --ring:rgba(255,140,0,0.20);
  --sans:'Geist',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --mono:'Geist Mono','SF Mono',Menlo,Consolas,monospace;
}}
html,body,.nicegui-content{{padding:0!important;margin:0!important;width:100%!important;max-width:100%!important;}}
body{{
  font-family:var(--sans); color:var(--ink); min-height:100vh; background-color:var(--bg);
  background-image:
    radial-gradient(rgba(17,24,39,0.032) 1px, transparent 1px),
    radial-gradient(1000px 520px at 5% -12%, rgba(255,160,30,0.26), transparent 56%),
    radial-gradient(820px 560px at 104% -4%, rgba(255,120,0,0.15), transparent 52%),
    radial-gradient(900px 640px at 60% 118%, rgba(255,170,60,0.13), transparent 60%);
  background-size:24px 24px,100% 100%,100% 100%,100% 100%; background-attachment:fixed;
}}
/* edge-lit glass: dark hairline OUTER border + inset top lip + grounded base + 3-layer shadow */
.glass{{
  background:linear-gradient(177deg, rgba(255,255,255,0.80), rgba(255,255,255,0.58));
  backdrop-filter:blur(26px) saturate(155%); -webkit-backdrop-filter:blur(26px) saturate(155%);
  border:1px solid {BORDER}; border-radius:20px;
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.68), inset 0 -1px 0 rgba(17,24,39,0.04),
    0 1px 2px rgba(17,24,39,0.05), 0 10px 24px rgba(17,24,39,0.06), 0 26px 54px rgba(17,24,39,0.05);
  transition:box-shadow .3s ease, transform .3s ease;
}}
.glass:hover{{box-shadow:inset 0 1px 0 rgba(255,255,255,0.8), 0 16px 36px rgba(17,24,39,0.10), 0 34px 70px rgba(17,24,39,0.07); transform:translateY(-2px);}}
.eyebrow{{font-family:var(--mono); font-style:italic; font-size:12px; color:var(--orange-strong); background:var(--tint); border:1px solid var(--ring); padding:3px 11px; border-radius:999px;}}
.wm-trace{{font-family:var(--sans); font-weight:700; font-size:25px; color:var(--ink); letter-spacing:-.015em;}}
.wm-ml{{font-family:var(--sans); font-weight:700; font-size:25px; color:var(--orange); letter-spacing:-.015em;}}
.meta-l{{font-family:var(--mono); font-size:9px; letter-spacing:.12em; color:var(--muted);}}
.meta-v{{font-family:var(--mono); font-size:14px; font-weight:600; color:var(--ink); font-variant-numeric:tabular-nums;}}
.livedot{{width:8px;height:8px;border-radius:999px;background:#16a34a;animation:tml-pulse 2.4s infinite;}}
@keyframes tml-pulse{{0%{{box-shadow:0 0 0 0 rgba(22,163,74,.5);}}70%{{box-shadow:0 0 0 6px rgba(22,163,74,0);}}100%{{box-shadow:0 0 0 0 rgba(22,163,74,0);}}}}
@keyframes tml-rise{{from{{opacity:0;transform:translateY(18px);}}to{{opacity:1;transform:none;}}}}
.reveal{{animation:tml-rise .7s cubic-bezier(.2,.7,.2,1) both;}}
.d1{{animation-delay:.07s;}} .d2{{animation-delay:.13s;}} .d3{{animation-delay:.19s;}} .d4{{animation-delay:.25s;}}
.ctitle{{font-family:var(--sans); font-weight:600; font-size:17px; color:var(--ink);}}
.cmeta{{font-family:var(--mono); font-size:11px; color:var(--muted);}}
/* phase ribbon (the hero signature) */
.ribbon{{display:flex; width:100%; height:34px; border-radius:13px; overflow:hidden; border:1px solid rgba(17,24,39,0.08);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.5), inset 0 -1px 0 rgba(17,24,39,.10), 0 3px 10px rgba(17,24,39,.10);}}
.pseg{{height:100%; transition:width .65s cubic-bezier(.4,0,.2,1); display:flex; align-items:center; justify-content:center; min-width:0; overflow:hidden; box-shadow:inset 0 1px 0 rgba(255,255,255,.32);}}
.seglab{{font-family:var(--mono); font-size:10px; font-weight:600; color:rgba(255,255,255,.96); white-space:nowrap; text-shadow:0 1px 1px rgba(0,0,0,.18);}}
.legchip{{font-family:var(--mono); font-size:10.5px; color:var(--muted); display:flex; align-items:center; gap:5px;}}
.legdot{{width:9px;height:9px;border-radius:3px;}}
.verdict{{font-family:var(--sans); font-size:21px; font-weight:500; color:var(--ink); letter-spacing:-.01em;}}
.sevpill{{font-family:var(--mono); font-size:10.5px; font-weight:600; padding:3px 9px; border-radius:999px; text-transform:uppercase; letter-spacing:.06em;}}
/* KPI tiles */
.kpi{{position:relative; background:rgba(255,255,255,0.4); border:1px solid rgba(17,24,39,0.08); border-radius:13px; padding:11px 13px 10px; min-width:118px; transition:background .2s, transform .2s, box-shadow .2s;}}
.kpi:hover{{background:rgba(255,255,255,0.72); transform:translateY(-2px); box-shadow:0 8px 20px rgba(17,24,39,0.07);}}
.kpi::before{{content:''; position:absolute; left:0; top:0; height:100%; width:3px; background:var(--acc,var(--orange)); opacity:.85;}}
.klab{{font-family:var(--mono); font-size:10px; letter-spacing:.09em; text-transform:uppercase; color:var(--orange-strong); font-weight:600;}}
.kq{{display:block; margin-top:2px; text-transform:none; letter-spacing:0; color:var(--muted); font-weight:500; font-size:9px;}}
.kval{{font-family:var(--mono); font-size:19px; font-weight:600; color:var(--ink); font-variant-numeric:tabular-nums; margin-top:4px; line-height:1.1;}}
.kunit{{font-size:0.62em; color:var(--muted); font-weight:500; margin-left:2px;}}
.ksub{{font-family:var(--mono); font-size:10px; color:var(--muted); margin-top:2px;}}
.diagrow{{display:flex; align-items:flex-start; gap:10px; padding:10px 0; border-top:1px solid rgba(17,24,39,0.07);}}
.diagdot{{width:9px; height:9px; border-radius:999px; margin-top:5px; flex:none;}}
.staleband{{font-family:var(--mono); font-size:11px; color:#b45309; background:rgba(239,108,0,0.10); border:1px solid rgba(239,108,0,0.22); padding:2px 9px; border-radius:999px;}}
</style>
"""


# --- format helpers ------------------------------------------------------
def kval(num: str, unit: str = "") -> str:
    """KPI value HTML with de-emphasized unit (big number, small muted unit)."""
    u = f"<span class='kunit'>{unit}</span>" if unit else ""
    return f"{num}{u}"


def gb(b: Any) -> Optional[float]:
    try:
        return float(b) / 1e9
    except Exception:
        return None


def nice_ymax(values: Any, floor: float = 10.0) -> float:
    """A 'nice' rounded y-axis ceiling for the given values: ~20% headroom
    above the peak, never below `floor`, snapped to a 5/10/25 step. Anchors
    the axis at 0 so low usage still reads as low, while a narrow band still
    fills the chart instead of hugging the bottom."""
    vals = [float(v) for v in values if v is not None]
    m = (max(vals) * 1.2) if vals else 0.0
    m = max(m, floor)
    step = 5.0 if m <= 50 else (10.0 if m <= 120 else 25.0)
    n = int(m / step)
    return float((n + 1) * step) if m > n * step else float(n * step)


# --- ECharts option builders ---------------------------------------------
_GRID = "rgba(17,24,39,0.05)"
_AXIS = "rgba(17,24,39,0.14)"
_TXT = "#9aa3af"


def _area(c1: str, c2: str) -> Dict[str, Any]:
    return {
        "type": "linear",
        "x": 0,
        "y": 0,
        "x2": 0,
        "y2": 1,
        "colorStops": [
            {"offset": 0, "color": c1},
            {"offset": 0.85, "color": c2},
            {"offset": 1, "color": "rgba(255,255,255,0)"},
        ],
    }


def _time_axis() -> Dict[str, Any]:
    return {
        "type": "time",
        "boundaryGap": False,
        "axisLabel": {
            "hideOverlap": True,
            "color": _TXT,
            "fontFamily": "Geist Mono",
            "fontSize": 10,
            ":formatter": "v=>{const d=new Date(v);return ('0'+d.getHours()).slice(-2)+':'+('0'+d.getMinutes()).slice(-2)+':'+('0'+d.getSeconds()).slice(-2);}",
        },
        "axisLine": {"lineStyle": {"color": _AXIS, "opacity": 0.5}},
        "axisTick": {"show": False},
        "splitLine": {"show": False},
    }


def dual_line_options(
    left_name: str,
    right_name: str,
    left_col: str = C_CPU,
    right_col: str = C_GPU,
    unit: str = "%",
    ymax: Optional[float] = None,
) -> Dict[str, Any]:
    """Dual-y-axis smooth area time-series (e.g. CPU/GPU or RAM/GPU-mem)."""

    def yax(side: str, col: str) -> Dict[str, Any]:
        ax = {
            "type": "value",
            "position": side,
            "axisLabel": {
                "color": _TXT,
                "fontFamily": "Geist Mono",
                "fontSize": 10,
                ":formatter": f"v=>v+'{unit}'",
            },
            "axisLine": {
                "show": True,
                "lineStyle": {"color": col, "opacity": 0.4},
            },
            "axisTick": {"show": False},
            "splitLine": {
                "show": side == "left",
                "lineStyle": {"color": _GRID},
            },
        }
        ax["min"] = 0
        if ymax is not None:
            ax["max"] = ymax
        return ax

    def ln(name: str, col: str, idx: int, c1: str, c2: str) -> Dict[str, Any]:
        return {
            "name": name,
            "type": "line",
            "smooth": True,
            "showSymbol": False,
            "yAxisIndex": idx,
            "lineStyle": {
                "width": 2.4,
                "color": col,
                "shadowColor": "rgba(17,24,39,0.18)",
                "shadowBlur": 4,
                "shadowOffsetY": 2,
            },
            "endLabel": {
                "show": True,
                "color": col,
                "fontFamily": "Geist Mono",
                "fontSize": 10,
                "fontWeight": 600,
                ":formatter": f"p=>Math.round(p.value[1])+'{unit}'",
            },
            "areaStyle": {"color": _area(c1, c2)},
            "data": [],
        }

    return {
        "backgroundColor": "transparent",
        "animationDuration": 600,
        "color": [left_col, right_col],
        "grid": {
            "left": 4,
            "right": 32,
            "top": 12,
            "bottom": 4,
            "containLabel": True,
        },
        "tooltip": {
            "trigger": "axis",
            "backgroundColor": "rgba(255,253,250,0.97)",
            "borderColor": BORDER,
            "textStyle": {
                "color": INK,
                "fontFamily": "Geist Mono",
                "fontSize": 11,
            },
            "axisPointer": {
                "type": "line",
                "lineStyle": {"color": _AXIS, "type": "dashed"},
            },
            ":valueFormatter": f"v=>(v==null?'-':Math.round(v)+'{unit}')",
        },
        "xAxis": _time_axis(),
        "yAxis": [yax("left", left_col), yax("right", right_col)],
        "series": [
            ln(
                left_name,
                left_col,
                0,
                "rgba(37,99,235,0.16)",
                "rgba(37,99,235,0.04)",
            ),
            ln(
                right_name,
                right_col,
                1,
                "rgba(255,140,0,0.24)",
                "rgba(255,140,0,0.06)",
            ),
        ],
    }


def gauge_options() -> Dict[str, Any]:
    """Single orange progress-ring gauge (0-100), big mono center value."""
    return {
        "backgroundColor": "transparent",
        "series": [
            {
                "type": "gauge",
                "startAngle": 218,
                "endAngle": -38,
                "min": 0,
                "max": 100,
                "radius": "94%",
                "center": ["50%", "58%"],
                "progress": {
                    "show": True,
                    "width": 13,
                    "roundCap": True,
                    "itemStyle": {
                        "color": {
                            "type": "linear",
                            "x": 0,
                            "y": 1,
                            "x2": 1,
                            "y2": 0,
                            "colorStops": [
                                {"offset": 0, "color": ORANGE_TOP},
                                {"offset": 1, "color": ORANGE},
                            ],
                        },
                        "shadowColor": "rgba(255,140,0,0.3)",
                        "shadowBlur": 8,
                    },
                },
                "axisLine": {
                    "lineStyle": {
                        "width": 13,
                        "color": [[1, "rgba(17,24,39,0.07)"]],
                    }
                },
                "pointer": {"show": False},
                "axisTick": {"show": False},
                "splitLine": {"show": False},
                "axisLabel": {"show": False},
                "anchor": {"show": False},
                "title": {"show": False},
                "detail": {
                    "valueAnimation": True,
                    "offsetCenter": [0, "2%"],
                    "fontFamily": "Geist Mono",
                    "fontSize": 32,
                    "fontWeight": 600,
                    "color": INK,
                    ":formatter": "v=>Math.round(v)+'%'",
                },
                "data": [{"value": 0}],
            }
        ],
    }


def single_line_options(
    name: str, col: str = ORANGE, unit: str = " GB"
) -> Dict[str, Any]:
    """Single smooth area line over an x index/step axis (e.g. step memory)."""
    return {
        "backgroundColor": "transparent",
        "animationDuration": 600,
        "color": [col],
        "grid": {
            "left": 4,
            "right": 30,
            "top": 14,
            "bottom": 4,
            "containLabel": True,
        },
        "tooltip": {
            "trigger": "axis",
            "backgroundColor": "rgba(255,253,250,0.97)",
            "borderColor": BORDER,
            "textStyle": {
                "color": INK,
                "fontFamily": "Geist Mono",
                "fontSize": 11,
            },
            "axisPointer": {
                "type": "line",
                "lineStyle": {"color": _AXIS, "type": "dashed"},
            },
        },
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "axisLabel": {
                "color": _TXT,
                "fontFamily": "Geist Mono",
                "fontSize": 10,
            },
            "axisLine": {"lineStyle": {"color": _AXIS, "opacity": 0.5}},
            "axisTick": {"show": False},
            "splitLine": {"show": False},
            "data": [],
        },
        "yAxis": {
            "type": "value",
            "scale": True,
            "axisLabel": {
                "color": _TXT,
                "fontFamily": "Geist Mono",
                "fontSize": 10,
            },
            "axisLine": {"show": False},
            "axisTick": {"show": False},
            "splitLine": {"lineStyle": {"color": _GRID}},
        },
        "series": [
            {
                "name": name,
                "type": "line",
                "smooth": True,
                "showSymbol": False,
                "lineStyle": {
                    "width": 2.4,
                    "color": col,
                    "shadowColor": "rgba(17,24,39,0.18)",
                    "shadowBlur": 4,
                    "shadowOffsetY": 2,
                },
                "endLabel": {
                    "show": True,
                    "color": col,
                    "fontFamily": "Geist Mono",
                    "fontSize": 10,
                    "fontWeight": 600,
                    ":formatter": f"p=>p.value.toFixed(2)+'{unit}'",
                },
                "areaStyle": {
                    "color": _area(
                        "rgba(255,140,0,0.22)", "rgba(255,140,0,0.05)"
                    )
                },
                "data": [],
            }
        ],
    }
