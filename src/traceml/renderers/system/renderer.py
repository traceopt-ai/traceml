"""
System renderer.

This module contains all presentation logic for system-level telemetry,
including:
- CLI rendering (Rich)
- Notebook rendering (HTML)
- Dashboard payload adaptation
- Summary logging

All metric computation is delegated to `SystemMetricsComputer`.
"""

import shutil
from typing import Any, Dict, Optional

from IPython.display import HTML
from rich.panel import Panel
from rich.table import Table

from traceml.database.remote_database_store import RemoteDBStore
from traceml.loggers.error_log import get_error_logger
from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.managers.cli_display_manager import (
    SYSTEM_LAYOUT,
)
from traceml.utils.formatting import fmt_mem_new, fmt_percent

from .compute import SystemMetricsComputer


class SystemRenderer(BaseRenderer):
    """
    Renderer for system-level telemetry.

    This class is presentation-focused and delegates all aggregation
    logic to `SystemMetricsComputer`.

    Outputs
    -------
    - CLI Rich panels
    - Jupyter notebook HTML cards
    - Dashboard-compatible payloads
    - Text summaries for logging
    """

    NAME = "System"

    def __init__(self, remote_store: RemoteDBStore):
        super().__init__(name=self.NAME, layout_section_name=SYSTEM_LAYOUT)
        self._store = remote_store
        self._logger = get_error_logger(self.NAME + "Renderer")

    def _get_table(self) -> Optional[Any]:
        """
        Retrieve the system table from the RemoteDBStore.
        Returns None if:
        - no data has arrived yet
        - DB or table has not been created yet
        """
        try:
            db = self._store.get_db(rank=0, sampler_name=self.NAME + "Sampler")
            if db is None:
                return None
            return db.get_table(self.NAME + "Table")
        except Exception as e:
            self._logger.error(f"[TraceML] Failed to fetch system table: {e}")
            return None

    # Snapshot computation (latest state)
    def _compute_snapshot(self) -> Dict[str, Any]:
        table = self._get_table()
        self._computer = SystemMetricsComputer(table)
        return self._computer.compute_snapshot()

    def _get_panel_cpu_row(self, table, data):
        ram_pct_str = ""
        if data["ram_total"]:
            ram_pct = data["ram_used"] * 100.0 / data["ram_total"]
            ram_pct_str = f" ({ram_pct:.1f}%)"

        table.add_row(
            f"[bold green]CPU[/bold green] {fmt_percent(data['cpu'])}",
            f"[bold green]RAM[/bold green] {fmt_mem_new(data['ram_used'])}/{fmt_mem_new(data['ram_total'])}{ram_pct_str}",
        )

    def _get_panel_gpu_row(self, table, data):
        """Render GPU metrics into a Rich table."""
        if not data["gpu_available"]:
            table.add_row(
                "[bold green]GPU[/bold green]", "[red]Not available[/red]"
            )
            return

        table.add_row(
            f"[bold green]GPU UTIL[/bold green] {fmt_percent(data['gpu_util_total'])}",
            f"[bold green]GPU MEM[/bold green] "
            f"{fmt_mem_new(data['gpu_mem_used'])}/"
            f"{fmt_mem_new(data['gpu_mem_total'])}",
        )

        # second GPU row: temperature + power
        temp = data.get("gpu_temp_max")
        pu = data.get("gpu_power_usage")
        pl = data.get("gpu_power_limit")

        temp_str = (
            f"[bold green]GPU TEMP[/bold green] {temp:.1f}°C"
            if temp is not None
            else "[bold green]GPU TEMP[/bold green] N/A"
        )
        power_str = (
            f"[bold green]GPU POWER[/bold green] {pu:.1f}W / {pl:.1f}W"
            if pu is not None and pl is not None
            else "[bold green]GPU POWER[/bold green] N/A"
        )
        table.add_row(temp_str, power_str)

    def get_panel_renderable(self) -> Panel:
        """
        Return a Rich Panel for CLI display.
        """
        data = self._compute_snapshot()

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="white")
        table.add_column(justify="left", style="white")

        self._get_panel_cpu_row(table, data)
        self._get_panel_gpu_row(table, data)

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            table,
            title="[bold cyan]System Metrics[/bold cyan]",
            title_align="center",
            border_style="cyan",
            width=panel_width,
        )

    def _get_notebook_cpu_row(self, data):
        ram_pct = ""
        if data["ram_total"]:
            try:
                rp = data["ram_used"] * 100.0 / data["ram_total"]
                ram_pct = f" ({rp:.1f}%)"
            except Exception:
                pass

        cpu_ram_html = f"""
            <div style="display:flex; justify-content:space-between;
                align-items:center; margin-bottom:12px;
            ">
                <div><b>CPU:</b> {fmt_percent(data['cpu'])}</div>
                <div>
                    <b>RAM:</b>
                    {fmt_mem_new(data['ram_used'])}/{fmt_mem_new(data['ram_total'])}{ram_pct}
                </div>
            </div>
        """
        return cpu_ram_html

    def get_dashboard_renderable(self):
        """
        Return an object suitable for dashboard rendering.:
        """
        data = self._compute_snapshot()
        data["table"] = self._get_table()
        return data

    def get_notebook_renderable(self) -> HTML:
        pass

    def log_summary(self, path) -> None:
        pass
