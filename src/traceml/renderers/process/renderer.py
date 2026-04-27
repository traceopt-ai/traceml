"""
Process renderer.

This module contains all presentation logic for process-level telemetry.
"""

import shutil
from typing import Any, Dict

from rich.panel import Panel
from rich.table import Table

from traceml.aggregator.display_drivers.layout import PROCESS_LAYOUT
from traceml.loggers.error_log import get_error_logger
from traceml.renderers.base_renderer import BaseRenderer
from traceml.utils.formatting import fmt_mem_new, fmt_mem_triple

from .computer import ProcessMetricsComputer


class ProcessRenderer(BaseRenderer):
    """
    Renderer for process-level telemetry.
    """

    NAME = "Process"

    def __init__(self, db_path: str):
        super().__init__(name=self.NAME, layout_section_name=PROCESS_LAYOUT)
        self.db_path = db_path
        self._computer = ProcessMetricsComputer(db_path=self.db_path)
        self._logger = get_error_logger(self.NAME + "Renderer")

    def get_panel_renderable(self) -> Panel:
        """
        Build the Rich panel for process telemetry.

        The snapshot is already aggregated by ProcessMetricsComputer:
        - CPU is worst-rank CPU at latest committed seq
        - GPU memory is taken from the least-headroom rank
        """
        snap = self._computer.compute_cli()

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="bright_white", no_wrap=True)
        table.add_column(justify="right", style="bright_white", no_wrap=True)

        cpu_used = float(snap.get("cpu_used") or 0.0)
        cpu_cores = cpu_used / 100.0
        table.add_row(
            "[bold green]CPU (worst rank)[/bold green]",
            f"{cpu_cores:.2f} cores",
        )

        gpu_used = snap.get("gpu_used")
        gpu_reserved = snap.get("gpu_reserved")
        gpu_total = snap.get("gpu_total")
        gpu_rank = snap.get("gpu_rank")

        if (
            gpu_used is not None
            and gpu_reserved is not None
            and gpu_total is not None
        ):
            gpu_str = fmt_mem_triple(gpu_used, gpu_reserved, gpu_total)
            if gpu_rank is not None:
                gpu_str += f" [dim](rank {gpu_rank})[/dim]"
        else:
            gpu_str = "[red]Not available[/red]"

        table.add_row(
            "[bold green]GPU MEM (used/reserved/total)[/bold green]",
            gpu_str,
        )

        gpu_imbalance = snap.get("gpu_used_imbalance")
        if gpu_imbalance is not None and gpu_imbalance > 0.0:
            table.add_row(
                "[bold green]GPU used imbalance[/bold green]",
                fmt_mem_new(gpu_imbalance),
            )

        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            table,
            title="[bold cyan]Process Metrics[/bold cyan]",
            border_style="cyan",
            width=width,
        )

    def get_dashboard_renderable(self) -> Dict[str, Any]:
        """
        Return the dashboard/UI payload.
        """
        return self._computer.compute_dashboard()
