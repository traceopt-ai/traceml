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


from typing import Dict, Any
import shutil
import numpy as np

from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.database.database import Database
from traceml.renderers.display.cli_display_manager import SYSTEM_LAYOUT
from traceml.utils.formatting import fmt_percent, fmt_mem_new
from traceml.renderers.utils import append_text, CARD_STYLE
from .system_compute import SystemMetricsComputer


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

    def __init__(self, database: Database):
        super().__init__(name="System", layout_section_name=SYSTEM_LAYOUT)
        self.db = database
        self._table = database.create_or_get_table("system")
        self._computer = SystemMetricsComputer(self._table)

    # Snapshot computation (latest state)
    def _compute_snapshot(self) -> Dict[str, Any]:
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
            table.add_row("[bold green]GPU[/bold green]", "[red]Not available[/red]")
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

    def get_notebook_renderable(self) -> HTML:
        """
        Return an HTML renderable for Jupyter notebooks.
        """
        data = self._compute_snapshot()

        # --- CPU + RAM ---
        cpu_ram_html = self._get_notebook_cpu_row(data)

        # --- GPU ---
        if data["gpu_available"]:
            gpu_util_html = f"{fmt_percent(data['gpu_util_total'])}"
            gpu_mem_html = f"{fmt_mem_new(data['gpu_mem_used'])} / {fmt_mem_new(data['gpu_mem_total'])}"

            temp = data.get("gpu_temp_max")
            pu = data.get("gpu_power_usage")
            pl = data.get("gpu_power_limit")

            temp_html = f"{temp:.1f}°C" if temp is not None else "N/A"
            power_html = (
                f"{pu:.1f}W / {pl:.1f}W" if pu is not None and pl is not None else "N/A"
            )

            gpu_section = f"""
                <div style="
                    display:flex; 
                    flex-direction:column; 
                    gap:6px;
                ">
                    <div>
                        <b>GPU Util:</b> {gpu_util_html}<br>
                        <b>GPU Mem:</b> {gpu_mem_html}
                    </div>
                    <div>
                        <b>GPU Temp:</b> {temp_html}<br>
                        <b>GPU Power:</b> {power_html}
                    </div>
                </div>
            """
        else:
            gpu_section = """
                <div><b>GPU:</b> <span style='color:red;'>Not available</span></div>
            """

        # --- Final card ---
        html = f"""
        <div style="{CARD_STYLE}">
            <h4 style="color:#d47a00;"; margin-top:0;">System Metrics</h4>

            {cpu_ram_html}

            {gpu_section}
        </div>
        """

        return HTML(html)

    def get_dashboard_renderable(self):
        """
        Return an object suitable for dashboard rendering.:
        """
        data = self._compute_snapshot()
        data["table"] = self._table
        return data

    def _compute_summary(self) -> Dict[str, Any]:
        return self._computer.compute_summary()

    def _cpu_summary(self, t, s):
        t.add_row(
            "CPU (avg / p95)",
            "[cyan]|[/cyan]",
            f"{s['cpu_avg_percent']:.1f}%  / {s['cpu_p95_percent']:.1f}%",
        )

    def _ram_summary(self, t, s):
        t.add_row(
            "RAM (avg / peak / total)",
            "[cyan]|[/cyan]",
            f"{fmt_mem_new(s['ram_avg_used'])} / "
            f"{fmt_mem_new(s['ram_peak_used'])} / "
            f"{fmt_mem_new(s['ram_total'])}",
        )

    def _gpu_summary(self, t, s):
        if not s["gpu_available"]:
            t.add_row("GPU", "[cyan]|[/cyan]", "[red]Not available[/red]")
            return

        t.add_row("GPU COUNT", "[cyan]|[/cyan]", str(s["gpu_total_count"]))
        t.add_row(
            "GPU UTIL (avg / peak)",
            "|",
            f"{s['gpu_util_total_avg']:.1f}% / " f"{s['gpu_util_total_peak']:.1f}%",
        )

        t.add_row(
            "GPU MEMORY (p95 / peak / capacity)",
            "|",
            f"{fmt_mem_new(s['gpu_mem_total_p95'])} / "
            f"{fmt_mem_new(s['gpu_mem_total_peak'])}  / "
            f"{fmt_mem_new(s['gpu_mem_total_capacity'])}",
        )
        t.add_row(
            "GPU MEMORY SINGLE DEVICE (peak)",
            "[cyan]|[/cyan]",
            f"{fmt_mem_new(s['gpu_mem_single_peak'])}",
        )

        t.add_row(
            "GPU TEMP (peak)",
            "[cyan]|[/cyan]",
            f"{s['gpu_temp_peak']:.1f}°C",
        )

    def log_summary(self, path) -> None:
        s = self._compute_summary()
        console = Console(record=True)

        t = Table.grid(padding=(0, 1))
        t.add_column(style="cyan")
        t.add_column(style="dim", no_wrap=True)
        t.add_column(style="white")

        t.add_row("TOTAL SYSTEM SAMPLES", "[cyan]|[/cyan]", str(s["total_samples"]))

        if s["total_samples"]:
            self._cpu_summary(t, s)
            self._ram_summary(t, s)
            self._gpu_summary(t, s)

        console.print(
            Panel(
                t, title="[bold cyan]System - Summary[/bold cyan]", border_style="cyan"
            )
        )
        if path:
            text = console.export_text(clear=False)
            append_text(path, "\n" + text + "\n")
