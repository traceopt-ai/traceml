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

from ipywidgets import HTML
from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from traceml.database.remote_database_store import RemoteDBStore
from traceml.renderers.base_renderer import BaseRenderer
from traceml.aggregator.display_drivers.layout import (
    MODEL_MEMORY_LAYOUT,
)
from traceml.utils.formatting import fmt_mem_new

from .compute import StepMemoryCombinedComputer
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

    def __init__(self, remote_store: RemoteDBStore):
        super().__init__(
            name="Model Step Memory",
            layout_section_name=MODEL_MEMORY_LAYOUT,
        )
        self._computer = StepMemoryCombinedComputer(remote_store)
        self._cached: Optional[StepMemoryCombinedResult] = None

    def _payload(self) -> Optional[StepMemoryCombinedResult]:
        """
        Fetch latest computed payload.

        Uses a simple cache to avoid flicker when data is temporarily
        incomplete (e.g., ranks slightly out of sync).
        """
        payload = self._computer.compute()
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
            "Worst Trend (Δ)",
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
        Format worst-series delta (last - first) as a compact trend hint.

        Uses bytes; fmt_mem_new handles scaling to MB/GB.
        """
        series = m.series
        if not series.steps or not series.worst:
            return "—"

        if len(series.worst) < 2:
            return "—"

        delta = float(series.worst[-1]) - float(series.worst[0])
        sign = "+" if delta >= 0 else ""
        return (
            f"{sign}{fmt_mem_new(abs(delta))}"
            if delta != 0
            else fmt_mem_new(0.0)
        )

    def get_dashboard_renderable(self) -> StepMemoryCombinedResult:
        """
        Return the typed compute result directly to dashboard consumers.
        """
        return self._payload()

    def get_notebook_renderable(self) -> HTML:
        pass

    def log_summary(self, path: Optional[str] = None) -> None:
        pass
