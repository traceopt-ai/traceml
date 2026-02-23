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
from typing import Any, Dict, Optional

from IPython.display import HTML
from rich.panel import Panel
from rich.table import Table

from traceml.database.remote_database_store import RemoteDBStore
from traceml.loggers.error_log import get_error_logger
from traceml.renderers.base_renderer import BaseRenderer
from traceml.aggregator.display_drivers.layout import (
    PROCESS_LAYOUT,
)
from traceml.utils.formatting import fmt_mem_new, fmt_percent, fmt_mem_triple

from .compute import ProcessMetricsComputer


class ProcessRenderer(BaseRenderer):
    """
    Renderer for process-level telemetry.
    """

    NAME = "Process"

    def __init__(self, remote_store: Optional[RemoteDBStore] = None):
        super().__init__(name=self.NAME, layout_section_name=PROCESS_LAYOUT)

        self._remote_store = remote_store
        self._computer = ProcessMetricsComputer(remote_store)
        self._logger = get_error_logger("ProcessRenderer")

    # CLI rendering
    def get_panel_renderable(self) -> Panel:
        snap = self._computer.compute_live_snapshot()

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="bright_white", no_wrap=True)
        table.add_column(justify="right",style="bright_white", no_wrap=True)
        table.add_column(justify="left", style="bright_white", no_wrap=True)

        table.add_row(
            "[bold green]CPU (worst rank)[/bold green] "
            f"{fmt_percent(snap.get('cpu_used', 0.0))}",
            "",
        )

        if snap.get("gpu_total") is not None:
            gpu_str = (
                f"{fmt_mem_triple(snap['gpu_used'], snap['gpu_reserved'], snap['gpu_total'])}"
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
        pass

    def log_summary(self, path=None) -> None:
        pass
