"""Generate the study figures from results.json.

Deterministic. Colorblind-safe Okabe-Ito palette (validated with the dataviz
skill's checker; every bar carries a direct value label as secondary encoding).
Writes PNG (dpi 200) + SVG for each figure into ./figures/.

Figures:
  1 overhead_by_mode      - cost %, split into continuous vs one-shot modes
  2 data_written          - artifact bytes per tool, log scale
  3 collection_time       - wall time each tool needs to produce its output
  4 capability_matrix     - tools x capabilities (qualitative, honest)
  5 cost_of_insight       - overhead vs interpretation effort positioning

Run AFTER parse_results.py. Missing numbers render as gaps, not crashes.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.environ.get("FIGDIR", os.path.join(HERE, "figures"))
RESULTS = os.environ.get("RESULTS_JSON", os.path.join(HERE, "results.json"))

# Okabe-Ito, assigned by entity (fixed order, never by rank).
C = {
    "bare": "#999999",
    "traceml": "#0072B2",
    "torch_profiler": "#D55E00",
    "cprofile": "#009E73",
    "combo": "#CC79A7",
}
LABEL = {
    "bare": "bare",
    "traceml": "TraceML",
    "torch_profiler": "torch.profiler",
    "cprofile": "cProfile",
    "combo": "combo",
}
STATUS = {"yes": "#009E73", "partial": "#E69F00", "no": "#CFCFCF"}
GLYPH = {"yes": "✓", "partial": "~", "no": "✗"}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#666666",
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "svg.fonttype": "none",
    }
)


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(
            os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight", dpi=200
        )
    plt.close(fig)
    print("wrote", name)


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load():
    with open(RESULTS) as f:
        return json.load(f)["configs"]


def fig_overhead(cfg):
    """Fig 1: overhead %, honestly split by mode."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    items = [
        ("traceml", cfg["traceml"].get("overhead_pct_vs_bare"), "continuous"),
        (
            "cprofile",
            cfg["cprofile"].get("overhead_pct_vs_bare"),
            "continuous",
        ),
        (
            "torch_profiler",
            cfg["torch_profiler"].get("perstep_overhead_pct_vs_bare"),
            "one-shot / step",
        ),
    ]
    xs = range(len(items))
    vals = [(v if v is not None else 0) for _, v, _ in items]
    colors = [C[k] for k, _, _ in items]
    bars = ax.bar(
        list(xs),
        vals,
        color=colors,
        width=0.6,
        edgecolor="white",
        linewidth=1.5,
    )
    for b, (k, v, mode) in zip(bars, items):
        txt = f"+{v:.1f}%" if v is not None else "n/a"
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            txt,
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{LABEL[k]}\n({mode})" for k, _, mode in items])
    ax.set_ylabel("overhead vs bare run (%)")
    ax.set_title("Overhead: two different modes, not one number")
    ax.axhline(0, color="#666666", linewidth=0.8)
    _clean(ax)
    fig.text(
        0.5,
        -0.02,
        "TraceML & cProfile: whole-run continuous.  "
        "torch.profiler: per-step inside its profiling window.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    _save(fig, "fig1_overhead_by_mode")


def fig_data(cfg):
    """Fig 2: data written per tool, log scale."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    order = ["traceml", "cprofile", "torch_profiler", "combo"]
    order = [k for k in order if cfg[k].get("artifact_bytes")]
    vals = [cfg[k]["artifact_bytes"] / 1e6 for k in order]
    colors = [C[k] for k in order]
    bars = ax.bar(
        range(len(order)),
        vals,
        color=colors,
        width=0.6,
        edgecolor="white",
        linewidth=1.5,
    )
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.1f} MB" if v >= 1 else f"{v*1000:.0f} KB",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_yscale("log")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABEL[k] for k in order])
    ax.set_ylabel("data written (MB, log scale)")
    ax.set_title("Footprint: the profiler trace dwarfs everything")
    _clean(ax)
    _save(fig, "fig2_data_written")


def fig_time(cfg):
    """Fig 3: wall time each tool needs to produce its output."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    items = [
        (
            "traceml",
            cfg["traceml"].get("training_duration_s"),
            "whole run (300 steps)",
        ),
        ("cprofile", cfg["cprofile"].get("wall_s"), "whole run (300 steps)"),
        (
            "torch_profiler",
            cfg["torch_profiler"].get("collect_wall_s"),
            "window (25 steps)",
        ),
    ]
    vals = [(v if v is not None else 0) for _, v, _ in items]
    colors = [C[k] for k, _, _ in items]
    bars = ax.bar(
        range(len(items)),
        vals,
        color=colors,
        width=0.6,
        edgecolor="white",
        linewidth=1.5,
    )
    for b, (k, v, sub) in zip(bars, items):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.0f}s" if v else "n/a",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([f"{LABEL[k]}\n{sub}" for k, _, sub in items])
    ax.set_ylabel("wall time to produce output (s)")
    ax.set_title("Time to a result (mind the different windows)")
    _clean(ax)
    _save(fig, "fig3_collection_time")


def fig_capability():
    """Fig 4: qualitative capability matrix (honest)."""
    caps = [
        "Sees GPU-idle\ndirectly",
        "Always-on /\ncontinuous",
        "One-line\nverdict",
        "Deep per-op\n'why'",
        "Usable at\n1 step",
    ]
    rows = ["traceml", "torch_profiler", "cprofile"]
    M = {
        "traceml": ["yes", "yes", "yes", "no", "no"],
        "torch_profiler": ["partial", "no", "no", "yes", "yes"],
        "cprofile": ["no", "no", "no", "partial", "yes"],
    }
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, r in enumerate(rows):
        for j, cap in enumerate(caps):
            s = M[r][j]
            ax.add_patch(
                plt.Rectangle(
                    (j, len(rows) - 1 - i),
                    0.94,
                    0.94,
                    facecolor=STATUS[s],
                    edgecolor="white",
                    linewidth=2,
                )
            )
            ax.text(
                j + 0.47,
                len(rows) - 1 - i + 0.47,
                GLYPH[s],
                ha="center",
                va="center",
                fontsize=15,
                color="white",
                fontweight="bold",
            )
    ax.set_xlim(0, len(caps))
    ax.set_ylim(0, len(rows))
    ax.set_xticks([j + 0.47 for j in range(len(caps))])
    ax.set_xticklabels(caps, fontsize=9)
    ax.set_yticks([len(rows) - 1 - i + 0.47 for i in range(len(rows))])
    ax.set_yticklabels([LABEL[r] for r in rows], fontweight="bold")
    ax.set_title("What each tool actually gives you")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    legend = [
        Patch(facecolor=STATUS["yes"], label="yes"),
        Patch(facecolor=STATUS["partial"], label="partial"),
        Patch(facecolor=STATUS["no"], label="no"),
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )
    _save(fig, "fig4_capability_matrix")


def fig_cost(cfg):
    """Fig 5: overhead vs interpretation-effort positioning."""
    # interpretation effort: honest 1(read one line)..5(read a trace)
    pts = {
        "traceml": (
            cfg["traceml"].get("overhead_pct_vs_bare"),
            1,
            "one-line verdict",
        ),
        "cprofile": (
            cfg["cprofile"].get("overhead_pct_vs_bare"),
            3,
            "CPU fns, GPU-blind",
        ),
        "torch_profiler": (
            cfg["torch_profiler"].get("perstep_overhead_pct_vs_bare"),
            5,
            "read the trace",
        ),
    }
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for k, (x, y, note) in pts.items():
        if x is None:
            continue
        ax.scatter(
            [x],
            [y],
            s=520,
            color=C[k],
            edgecolor="white",
            linewidth=2,
            zorder=3,
        )
        ax.annotate(
            f"{LABEL[k]}\n{note}",
            (x, y),
            textcoords="offset points",
            xytext=(14, 0),
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xlabel("overhead vs bare (%)  -> cheaper is left")
    ax.set_ylabel("interpretation effort  -> easier is down")
    ax.set_yticks([1, 3, 5])
    ax.set_yticklabels(["one line", "some reading", "read a trace"])
    ax.set_title("Cost of insight: cheap tripwire vs deep scalpel")
    ax.margins(x=0.25, y=0.25)
    _clean(ax)
    fig.text(
        0.5,
        -0.02,
        "Lower-left = cheap + easy (tripwire).  "
        "Upper-right = costly + deep (scalpel).",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    _save(fig, "fig5_cost_of_insight")


def main():
    cfg = load()
    fig_overhead(cfg)
    fig_data(cfg)
    fig_time(cfg)
    fig_capability()
    fig_cost(cfg)
    print("all figures written to", FIGDIR)


if __name__ == "__main__":
    main()
