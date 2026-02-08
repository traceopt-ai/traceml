"""
Step Combined Renderer

This renderer presents a **window-summed, rank-agnostic** summary of
step-level execution time metrics computed by `StepCombinedComputer`.

Table semantics
---------------
Columns:
- One column per metric (dataloader, forward, backward, optimizer, …)

Rows:
- Median (ΣK)     : typical rank total over last K steps
- Worst (ΣK)      : worst rank total over last K steps
- Worst Rank      : rank responsible for worst total
- Skew (%)        : (worst − median) / median

This table is intentionally **stable and low-noise**.
Per-step volatility belongs in plots, not summaries.
"""

from typing import Optional
import shutil

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from traceml.database.remote_database_store import RemoteDBStore
from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.managers.cli_display_manager import MODEL_COMBINED_LAYOUT
from traceml.renderers.utils import fmt_time_run

from .compute import StepCombinedComputer
from .schema import StepCombinedTimeResult


class StepCombinedRenderer(BaseRenderer):
    """
    CLI renderer for step combined time summary.

    This renderer shows a **window-summed summary table** where:
    - columns = metrics
    - rows    = median, worst, worst-rank, skew

    It is designed to surface **stragglers and imbalance quickly**
    without overwhelming the user with per-step noise.
    """

    def __init__(self, remote_store: RemoteDBStore):
        super().__init__(
            name="Model Step Summary",
            layout_section_name=MODEL_COMBINED_LAYOUT,
        )
        self._computer = StepCombinedComputer(remote_store)
        self._cached: Optional[StepCombinedTimeResult] = None


    def _payload(self) -> Optional[StepCombinedTimeResult]:
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
        Render the CLI panel containing the summary table.

        Returns
        -------
        rich.panel.Panel
            Renderer-ready panel for CLI display.
        """
        payload = self._payload()

        if payload is None or not payload.metrics:
            return Panel(
                "Waiting for first fully completed step across all ranks…",
                title="Model Step Summary",
            )

        metrics = payload.metrics
        step_metric = next(
            (m for m in metrics if m.metric == "step_time_ms"),
            None,
        )
        wait_metric = next(
            (m for m in metrics if m.metric == "wait_proxy"),
            None,
        )

        metrics = sorted(
            metrics,
            key=lambda m: (m.metric == "wait_proxy")
        )
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
            if m.metric == "wait_proxy":
                title = "Wait*"
            else:
                title = m.metric.replace("_", " ").title()

            table.add_column(title, justify="right")


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
                f"r{m.summary.worst_rank}"
                if m.summary.worst_rank is not None
                else "—"
                for m in metrics
            ],
        )

        table.add_row(
            "Skew (%)",
            *[f"+{m.summary.skew_pct * 100:.1f}%" for m in metrics],
        )

        subtitle = (
            f"Summed over last {K} fully completed steps"
            if K > 0
            else "Waiting for first fully completed step"
        )

        table.add_row("")
        if step_metric and wait_metric and step_metric.summary.median_total > 0:
            wait_share = (
                    wait_metric.summary.median_total
                    / step_metric.summary.median_total
            )

            table.add_row(
                "WAIT Share (%)",
                *[
                    f"[red]{wait_share * 100:.1f}%[/red]" if m.metric == "wait_proxy" else ""
                    for m in metrics
                ],
            )


        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 100)
        footer = "\n\n[dim]* WAIT = step time − GPU compute (mixed CPU/GPU proxy)[/dim]"

        return Panel(
            Group(
                table,
                footer,
            ),
            title=f"Model Step Summary ({subtitle})",
            border_style="cyan",
            width=width,
        )

    def get_dashboard_renderable(self) -> StepCombinedTimeResult:
        """
        Return the typed compute result directly to dashboard consumers.
        """
        return self._payload()


    def log_summary(self, path: Optional[str] = None) -> None:
        pass
