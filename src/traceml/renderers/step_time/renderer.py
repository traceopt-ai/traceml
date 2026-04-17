"""
Step Combined Renderer

CLI renderer for window-summed, rank-agnostic step-time metrics.

Behavior:
- If world_size == 1 (single-rank run): show only SUM rows (no skew/worst)
- Else: show Median/Worst/Worst Rank/Skew rows
"""

import shutil
from typing import Optional

from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from traceml.aggregator.display_drivers.layout import MODEL_COMBINED_LAYOUT
from traceml.diagnostics.trends import (
    DEFAULT_TREND_CONFIG,
    compute_trend_pct,
    format_trend_pct,
)
from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.utils import fmt_time_run

from .compute import StepCombinedComputer
from .diagnostics import build_step_diagnosis, format_cli_diagnosis
from .schema import StepCombinedTimeResult


class StepCombinedRenderer(BaseRenderer):
    """
    CLI renderer for step combined time summary.
    """

    def __init__(self, db_path):
        super().__init__(
            name="Model Step Summary",
            layout_section_name=MODEL_COMBINED_LAYOUT,
        )
        self._computer = StepCombinedComputer(db_path=db_path)
        self._cached: Optional[StepCombinedTimeResult] = None

    def _payload(self) -> Optional[StepCombinedTimeResult]:
        """
        CLI compute is summary-only (cheap).
        Cache to avoid flicker on transient incompleteness.
        """
        payload = self._computer.compute_cli()
        if payload and payload.metrics:
            self._cached = payload
        return self._cached

    def get_panel_renderable(self) -> Panel:
        payload = self._payload()

        if payload is None or not payload.metrics:
            return Panel(
                "Waiting for first fully completed step across all ranks…",
                title="Model Step Summary",
            )

        metrics = payload.metrics
        diag = build_step_diagnosis(metrics)
        diag_text = format_cli_diagnosis(diag)

        step_metric = next(
            (m for m in metrics if m.metric == "step_time"), None
        )
        wait_metric = next(
            (m for m in metrics if m.metric == "wait_proxy"), None
        )

        # Put wait_proxy last
        metrics = sorted(metrics, key=lambda m: (m.metric == "wait_proxy"))

        # All metrics share the same window size by construction
        K = metrics[0].summary.steps_used
        world_size = metrics[0].coverage.world_size
        ranks_present = metrics[0].coverage.ranks_present
        single_rank = (world_size <= 1) or (ranks_present <= 1)

        table = Table(
            show_header=True, header_style="bold blue", box=None, expand=False
        )
        table.add_column("Metric", style="magenta")

        for m in metrics:
            title = (
                "Wait*"
                if m.metric == "wait_proxy"
                else m.metric.replace("_", " ").title()
            )
            table.add_column(title, justify="right")

        subtitle = (
            f"Summed over last {K} fully completed steps"
            if K > 0
            else "Waiting for first fully completed step"
        )

        if single_rank:
            table.add_row(
                f"Sum (Σ {K})",
                *[
                    fmt_time_run(m.summary.worst_total) for m in metrics
                ],  # worst_total==sum in single-rank mode
            )
        else:
            table.add_row(
                f"Median (Σ {K})",
                *[fmt_time_run(m.summary.median_total) for m in metrics],
            )
            table.add_row(
                f"Worst (Σ {K})",
                *[fmt_time_run(m.summary.worst_total) for m in metrics],
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

        table.add_row("")
        if K >= DEFAULT_TREND_CONFIG.min_points:
            table.add_row(
                "Trend",
                *[_metric_trend_label(m, single_rank) for m in metrics],
            )
            table.add_row("")

        # Optional WAIT share line (still meaningful in both modes)
        if step_metric and wait_metric and step_metric.summary.worst_total > 0:
            denom = (
                step_metric.summary.median_total
                if not single_rank
                else step_metric.summary.worst_total
            )
            wait_share = (
                wait_metric.summary.median_total / denom if denom > 0 else 0.0
            )

            table.add_row(
                "WAIT Share (%)",
                *[
                    (
                        f"[red]{wait_share * 100:.1f}%[/red]"
                        if m.metric == "wait_proxy"
                        else ""
                    )
                    for m in metrics
                ],
            )

        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 120)

        footer = "\n\n[dim]* WAIT = step time − model execution (mixed CPU/GPU proxy)[/dim]"
        return Panel(
            Group(diag_text, "", table, footer),
            title=f"Model Step Summary ({subtitle})",
            border_style="cyan",
            width=width,
        )

    def get_dashboard_renderable(self) -> StepCombinedTimeResult:
        """
        Dashboard gets a richer payload (cheap summaries + rank heatmap).
        """
        return self._computer.compute_dashboard()


def _metric_trend_label(metric, single_rank: bool) -> str:
    if metric is None or metric.series is None:
        return "—"
    series = metric.series.worst if single_rank else metric.series.median
    return format_trend_pct(
        compute_trend_pct(series, config=DEFAULT_TREND_CONFIG),
        deadband_pct=DEFAULT_TREND_CONFIG.deadband_pct,
    )
