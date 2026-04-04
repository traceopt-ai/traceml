"""
Step Memory Combined Renderer

This renderer presents a **window-peak, rank-agnostic** summary of
step-level peak memory metrics computed by `StepMemoryCombinedComputer`.

Table semantics
---------------
Columns:
- One column per memory metric:
    - Peak Allocated
    - Peak Reserved

Rows (over last K fully completed steps, aligned across ranks):
- Median Peak (max/K) : median rank's *peak* over the window
- Worst Peak (max/K)  : worst rank's *peak* over the window
- Worst Rank          : rank responsible for worst peak
- Skew (%)            : (worst − median) / median

Optional (low-noise):
- Worst Trend (Δ)     : (last_worst − first_worst) over the aligned series window

This table is intentionally stable and low-noise.
Per-step volatility belongs in plots, not summaries.
"""

import shutil
from typing import Optional

from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from traceml.aggregator.display_drivers.layout import MODEL_MEMORY_LAYOUT
from traceml.renderers.base_renderer import BaseRenderer
from traceml.utils.formatting import fmt_mem_new

from .computer import StepMemoryMetricsComputer
from .diagnostics import build_step_memory_diagnosis, format_cli_diagnosis
from .schema import StepMemoryCombinedResult


class StepMemoryRenderer(BaseRenderer):
    """
    CLI renderer for step-level combined memory summary.

    This renderer shows a **window-peak summary table** where:
    - columns = memory metrics (allocated, reserved)
    - rows    = median peak, worst peak, worst-rank, skew (+ optional trend)

    It is designed to surface **OOM risk and rank imbalance quickly**
    without overwhelming the user with per-step noise.
    """

    def __init__(self, db_path: str):
        super().__init__(
            name="Model Step Memory",
            layout_section_name=MODEL_MEMORY_LAYOUT,
        )
        self._computer = StepMemoryMetricsComputer(db_path=db_path)
        self._cached: Optional[StepMemoryCombinedResult] = None

    def _payload(self) -> Optional[StepMemoryCombinedResult]:
        """
        Fetch latest computed payload.

        Uses a simple cache to avoid flicker when data is temporarily
        incomplete (e.g., ranks slightly out of sync).
        """
        payload = self._computer.compute_cli()
        if payload and payload.metrics:
            self._cached = payload
        return self._cached

    def get_panel_renderable(self) -> Panel:
        """
        Render the CLI panel containing the memory summary table.

        Returns
        -------
        rich.panel.Panel
            Renderer-ready panel for CLI display.
        """
        payload = self._payload()

        if payload is None or not payload.metrics:
            return Panel(
                "Waiting for first fully completed step across all ranks…",
                title="Model Step Memory",
            )

        metrics = payload.metrics

        diag = build_step_memory_diagnosis(metrics)
        diag_text = format_cli_diagnosis(diag)

        # Stable order: allocated then reserved (if present)
        def _sort_key(m) -> int:
            if m.metric == "peak_allocated":
                return 0
            if m.metric == "peak_reserved":
                return 1
            return 99

        metrics = sorted(metrics, key=_sort_key)

        # All metrics share the same window size by construction
        K = metrics[0].summary.steps_used

        table = Table(
            show_header=True,
            header_style="bold blue",
            box=None,
            expand=False,
        )

        table.add_column("Metric", style="magenta")

        for m in metrics:
            if m.metric == "peak_allocated":
                title = "Peak Allocated"
            elif m.metric == "peak_reserved":
                title = "Peak Reserved"
            else:
                title = m.metric.replace("_", " ").title()

            table.add_column(title, justify="right")

        # Window-peak rows (bytes formatted via fmt_mem_new)
        table.add_row(
            f"Median Peak (max/{K})",
            *[fmt_mem_new(m.summary.median_peak) for m in metrics],
        )

        table.add_row(
            f"Worst Peak (max/{K})",
            *[fmt_mem_new(m.summary.worst_peak) for m in metrics],
        )

        table.add_row(
            "Worst Rank",
            *[
                (
                    f"r{m.summary.worst_rank}"
                    if m.summary.worst_rank is not None
                    else "—"
                )
                for m in metrics
            ],
        )

        table.add_row(
            "Skew (%)",
            *[f"+{m.summary.skew_pct * 100:.1f}%" for m in metrics],
        )

        # Optional: low-noise trend (use worst series delta)
        # This is helpful for spotting monotonic growth / fragmentation.
        table.add_row("")
        table.add_row(
            "Head/Tail Delta (worst)",
            *[self._format_worst_trend_delta(m) for m in metrics],
        )

        subtitle = (
            f"Peaks over last {K} fully completed steps"
            if K > 0
            else "Waiting for first fully completed step"
        )

        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 120)

        footer = "\n\n[dim]Peaks = per-rank max over last K; median/worst = across ranks.[/dim]"

        return Panel(
            Group(
                diag_text,
                "",
                table,
                footer,
            ),
            title=f"Model Step Memory ({subtitle})",
            border_style="cyan",
            width=width,
        )

    @staticmethod
    def _format_worst_trend_delta(m) -> str:
        """
        Format a stable head-vs-tail delta for the worst series.

        This mirrors the memory diagnosis more closely than raw last-minus-first.
        """
        series = m.series
        if not series.steps or not series.worst:
            return "—"

        values = [float(v) for v in series.worst]
        n = len(values)
        if n < 2:
            return "—"

        segment = max(4, int(round(n * 0.20)))
        segment = min(segment, max(1, n // 2))

        head_avg = sum(values[:segment]) / segment
        tail_avg = sum(values[-segment:]) / segment
        delta = tail_avg - head_avg

        if delta == 0.0:
            return fmt_mem_new(0.0)

        sign = "+" if delta > 0.0 else "-"
        return f"{sign}{fmt_mem_new(abs(delta))}"

    def get_dashboard_renderable(self) -> StepMemoryCombinedResult:
        """
        Return the typed compute result directly to dashboard consumers.
        """
        return self._computer.compute_dashboard()
