"""
System renderer.

Presentation logic for system-level telemetry:
- CLI rendering (Rich)
- Dashboard payload (dict)
- Summary logging (optional)

All metric computation is delegated to SystemMetricsComputer.
"""

import shutil
from typing import Any, Dict, Optional

from rich.panel import Panel
from rich.table import Table

from traceml.aggregator.display_drivers.layout import SYSTEM_LAYOUT
from traceml.database.remote_database_store import RemoteDBStore
from traceml.loggers.error_log import get_error_logger
from traceml.renderers.base_renderer import BaseRenderer
from traceml.utils.formatting import fmt_mem_new, fmt_percent

from .compute import SystemMetricsComputer


class SystemRenderer(BaseRenderer):
    """
    Renderer for system-level telemetry.

    Driver expectations:
    - CLI driver calls get_panel_renderable()
    - Dashboard driver calls get_dashboard_renderable()
    """

    NAME = "System"

    def __init__(self, remote_store: RemoteDBStore) -> None:
        super().__init__(name=self.NAME, layout_section_name=SYSTEM_LAYOUT)
        self._store = remote_store
        self._logger = get_error_logger(self.NAME + "Renderer")

    def _get_table(self) -> Optional[Any]:
        """
        Fetch the system table from RemoteDBStore.

        Returns None if:
        - DB doesn't exist yet
        - table isn't created yet
        - store read fails
        """
        try:
            db = self._store.get_db(rank=0, sampler_name=self.NAME + "Sampler")
            if db is None:
                return None
            return db.get_table(self.NAME + "Table")
        except Exception as e:
            self._logger.error(f"[TraceML] Failed to fetch system table: {e}")
            return None

    def _compute_cli(self) -> Dict[str, Any]:
        table = self._get_table() or []
        return SystemMetricsComputer(table).compute_cli()

    def _compute_dashboard(self, window_n: int = 100) -> Dict[str, Any]:
        table = self._get_table() or []
        return SystemMetricsComputer(table).compute_dashboard(window_n=window_n)



    def get_panel_renderable(self) -> Panel:
        """Return a Rich Panel for CLI display (latest sample)."""
        data = self._compute_cli()

        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="left", style="white")
        grid.add_column(justify="left", style="white")

        # CPU + RAM row
        ram_pct_str = ""
        if data["ram_total"]:
            ram_pct = data["ram_used"] * 100.0 / data["ram_total"]
            ram_pct_str = f" ({ram_pct:.1f}%)"

        grid.add_row(
            f"[bold green]CPU[/bold green] {fmt_percent(data['cpu'])}",
            f"[bold green]RAM[/bold green] "
            f"{fmt_mem_new(data['ram_used'])}/{fmt_mem_new(data['ram_total'])}{ram_pct_str}",
        )

        # GPU rows
        if not data["gpu_available"]:
            grid.add_row("[bold green]GPU[/bold green]", "[red]Not available[/red]")
        else:
            grid.add_row(
                f"[bold green]GPU UTIL[/bold green] {fmt_percent(data['gpu_util_total'])}",
                f"[bold green]GPU MEM[/bold green] "
                f"{fmt_mem_new(data['gpu_mem_used'])}/{fmt_mem_new(data['gpu_mem_total'])}",
            )

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
            grid.add_row(temp_str, power_str)

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            grid,
            title="[bold cyan]System Metrics[/bold cyan]",
            title_align="center",
            border_style="cyan",
            width=panel_width,
        )

    def get_dashboard_renderable(self) -> Dict[str, Any]:
        """
        Return a compact dashboard payload.

        This matches the semantics of your existing system dashboard card, but
        avoids shipping the raw system table to the UI.
        """
        return self._compute_dashboard(window_n=100)

    def log_summary(self, path) -> None:
        """
        Optional: write a run summary (p95/avg over full run).
        You can implement later by adding compute_summary() back if needed.
        """
        pass