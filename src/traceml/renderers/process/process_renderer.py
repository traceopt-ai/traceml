"""
Process renderer.

This module contains all presentation logic for process-level telemetry.

Responsibilities
----------------
- CLI rendering (Rich)
- Notebook rendering (HTML)
- Dashboard payload adaptation
- Summary logging

All aggregation and synchronization logic is delegated to
`ProcessMetricsComputer`.
"""

import shutil
from typing import Dict, Any, Optional

from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import PROCESS_LAYOUT
from traceml.utils.formatting import fmt_percent, fmt_mem_new
from traceml.database.remote_database_store import RemoteDBStore
from traceml.loggers.error_log import get_error_logger

from .process_compute import ProcessMetricsComputer
from traceml.renderers.utils import CARD_STYLE


class ProcessRenderer(BaseRenderer):
    """
    Renderer for process-level telemetry.
    """

    TABLE_NAME = "process"

    def __init__(self, remote_store: Optional[RemoteDBStore] = None):
        super().__init__(name="Process", layout_section_name=PROCESS_LAYOUT)

        self._remote_store = remote_store
        self._computer = ProcessMetricsComputer(remote_store)
        self._logger = get_error_logger("ProcessRenderer")

    # CLI rendering
    def get_panel_renderable(self) -> Panel:
        snap = self._computer.compute_live_snapshot()

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left")
        table.add_column(justify="left")

        table.add_row(
            "[bold green]CPU (worst rank)[/bold green] "
            f"{fmt_percent(snap.get('cpu_used', 0.0))}",
            "",
        )

        if snap.get("gpu_total") is not None:
            gpu_str = (
                f"{fmt_mem_new(snap['gpu_used'])}/"
                f"{fmt_mem_new(snap['gpu_reserved'])}/"
                f"{fmt_mem_new(snap['gpu_total'])}"
                f" [dim](rank {snap.get('gpu_rank')})[/dim]"
            )
        else:
            gpu_str = "[red]Not available[/red]"

        table.add_row(
            "[bold green]GPU MEM (used/reserved/total)[/bold green]",
            gpu_str,
        )

        if snap.get("gpu_used_imbalance", 0.0) > 0.0:
            table.add_row(
                "[bold green]GPU used imbalance[/bold green]",
                fmt_mem_new(snap["gpu_used_imbalance"]),
            )

        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            table,
            title="[bold cyan]Process Metrics[/bold cyan]",
            border_style="cyan",
            width=width,
        )

    # Dashboard payload
    def get_dashboard_renderable(self) -> Dict[str, Any]:
        self._computer.update_dashboard()
        snap = self._computer.compute_live_snapshot()
        snap["history"] = self._computer.get_dashboard_history()
        return snap


    # Notebook rendering
    def get_notebook_renderable(self) -> HTML:
        snap = self._computer.compute_live_snapshot()

        gpu_html = (
            f"""
            <div>
                <b>GPU MEM (worst rank {snap.get("gpu_rank")}):</b>
                {fmt_mem_new(snap['gpu_used'])} /
                {fmt_mem_new(snap['gpu_reserved'])} /
                {fmt_mem_new(snap['gpu_total'])}
            </div>
            """
            if snap.get("gpu_total") is not None
            else "<div><b>GPU:</b> <span style='color:red;'>Not available</span></div>"
        )

        html = f"""
        <div style="{CARD_STYLE}">
            <h4 style="color:#d47a00; margin-top:0;">Process Metrics</h4>

            <div>
                <b>CPU (worst rank):</b>
                {fmt_percent(snap.get('cpu_used', 0.0))}
            </div>

            {gpu_html}
        </div>
        """

        return HTML(html)


    # Summary logging
    def log_summary(self, path=None) -> None:
        """
        Log process summary statistics.

        Notes
        -----
        Summary semantics will be revisited. This method is kept for
        completeness and API stability.
        """
        if not self._table:
            return

        summary = self._computer.compute_summary(list(self._table))
        console = Console()

        t = Table.grid(padding=(0, 1))
        t.add_column(style="magenta")
        t.add_column(style="dim", no_wrap=True)
        t.add_column(style="white")

        t.add_row(
            "TOTAL PROCESS SAMPLES",
            "[magenta]|[/magenta]",
            str(summary.get("total_samples", 0)),
        )

        console.print(
            Panel(
                t,
                title="[bold magenta]Process - Summary[/bold magenta]",
                border_style="magenta",
            )
        )
