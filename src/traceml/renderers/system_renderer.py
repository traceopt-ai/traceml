from typing import Dict, Any
import shutil
import numpy as np
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.database.database import Database
from traceml.renderers.display.cli_display_manager import SYSTEM_LAYOUT_NAME
from traceml.utils.formatting import fmt_percent, fmt_mem_new


class SystemRenderer(BaseRenderer):
    """
    Renderer that reads from the 'system' table (list of samples)
    and computes live + summary statistics for CPU, RAM, and GPU.
    """

    def __init__(self, database: Database):
        super().__init__(name="System", layout_section_name=SYSTEM_LAYOUT_NAME)
        self.db = database
        self._table = database.create_or_get_table("system")

    def _compute_snapshot(self) -> Dict[str, Any]:
        """Return latest system info for live display."""
        latest = self.db.get_record_at_index("system", -1)
        if not latest:
            return {
                "cpu": 0.0,
                "ram_used": 0.0,
                "ram_total": 0.0,
                "gpu_available": False,
                "gpu_util_total": None,
                "gpu_mem_used": None,
                "gpu_mem_total": None,
            }

        gpu_raw = latest.get("gpu_raw", {}) or {}

        # total util, mem & mem_total across all GPUs
        util_total = sum(v["util"] for v in gpu_raw.values()) if gpu_raw else None
        mem_used_total = (
            sum(v["mem_used"] for v in gpu_raw.values()) if gpu_raw else None
        )
        mem_total_total = (
            sum(v["mem_total"] for v in gpu_raw.values()) if gpu_raw else None
        )

        return {
            "cpu": latest.get("cpu_percent", 0.0),
            "ram_used": latest.get("ram_used", 0.0),
            "ram_total": latest.get("ram_total", 0.0),
            "gpu_available": latest.get("gpu_available", False),
            "gpu_util_total": util_total,
            "gpu_mem_used": mem_used_total,
            "gpu_mem_total": mem_total_total,
        }

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute averages and peaks from the entire table."""
        if not self._table:
            return {"error": "no data", "total_samples": 0}

        gpu_available = self._table[-1].get("gpu_available", False)
        gpu_count = self._table[-1].get("gpu_count", 0)

        cpu_vals = [x.get("cpu_percent", 0.0) for x in self._table]
        ram_vals = [x.get("ram_used", 0.0) for x in self._table]
        ram_total = self._table[-1].get("ram_total", 0.0)

        summary = {
            "total_samples": len(self._table),
            "cpu_average_percent": round(float(np.mean(cpu_vals)), 2),
            "cpu_peak_percent": round(float(np.max(cpu_vals)), 2),
            "ram_average_used": round(float(np.mean(ram_vals)), 2),
            "ram_peak_used": round(float(np.max(ram_vals)), 2),
            "ram_total": ram_total,
            "gpu_available": gpu_available,
            "gpu_total_count": gpu_count,
        }

        util_totals = []
        mem_used_totals = []
        mem_total_totals = []

        for x in self._table:
            gpu_raw = x.get("gpu_raw", {}) or {}
            if gpu_raw:
                util_totals.append(sum(v["util"] for v in gpu_raw.values()))
                mem_used_totals.append(sum(v["mem_used"] for v in gpu_raw.values()))
                mem_total_totals.append(sum(v["mem_total"] for v in gpu_raw.values()))

        if gpu_available and util_totals:
            summary.update(
                {
                    "gpu_average_util_total": round(float(np.mean(util_totals)), 2),
                    "gpu_peak_util_total": round(float(np.max(util_totals)), 2),
                    "gpu_memory_average_used": round(
                        float(np.mean(mem_used_totals)), 2
                    ),
                    "gpu_memory_peak_used": round(float(np.max(mem_used_totals)), 2),
                    "gpu_memory_total": round(float(np.mean(mem_total_totals)), 2),
                }
            )
        return summary

    def get_panel_renderable(self) -> Panel:
        data = self._compute_snapshot()

        ram_pct_str = ""
        if data["ram_total"]:
            try:
                ram_pct_str = f" ({data['ram_used'] * 100.0 / data['ram_total']:.1f}%)"
            except Exception:
                pass

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="white")
        table.add_column(justify="left", style="white")

        sys_info = [
            f"[bold green]CPU[/bold green] {fmt_percent(data['cpu'])}",
            f"[bold green]RAM[/bold green] {fmt_mem_new(data['ram_used'])}/{fmt_mem_new(data['ram_total'])}{ram_pct_str}",
        ]

        table.add_row(*sys_info)
        table.add_row("")

        if data["gpu_available"]:
            gpu_str = [
                f"[bold green]GPU[/bold green] {fmt_percent(data['gpu_util_total'])}",
                f"[bold green]GPU MEM [/bold green] {fmt_mem_new(data['gpu_mem_used'])}/{fmt_mem_new(data['gpu_mem_total'])}",
            ]
        else:
            gpu_str = "[bold green]GPU[/bold green] [red]Not available[/red]"

        table.add_row(gpu_str)

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            table,
            title="[bold cyan]System[/bold cyan]",
            title_align="center",
            border_style="cyan",
            width=panel_width,
        )

    def get_notebook_renderable(self) -> HTML:
        data = self._compute_snapshot()

        # CPU + RAM first row
        cpu_ram_html = f"""
            <p>
                <b>CPU:</b> {fmt_percent(data['cpu'])}<br>
                <b>RAM:</b> {fmt_mem_new(data['ram_used'])} / {fmt_mem_new(data['ram_total'])}
                {"(" + str(round(data['ram_used'] * 100 / data['ram_total'], 1)) + "%)" if data['ram_total'] else ""}
            </p>
        """

        # GPU second row
        if data["gpu_available"]:
            gpu_html = f"""
                <p>
                    <b>GPU Util:</b> {fmt_percent(data['gpu_util_total'])}<br>
                    <b>GPU Memory:</b> {fmt_mem_new(data['gpu_mem_used'])} /
                    {fmt_mem_new(data['gpu_mem_total'])}
                </p>
            """
        else:
            gpu_html = """
                <p><b>GPU:</b> <span style='color:red;'>Not available</span></p>
            """

        html = f"""
        <div style="border:2px solid #00bcd4; border-radius:8px; padding:12px;">
            <h4 style="color:#00bcd4; margin-top:0;">System</h4>

            <!-- CPU + RAM -->
            {cpu_ram_html}

            <!-- GPU -->
            {gpu_html}
        </div>
        """

        return HTML(html)

    def _cpu_summary(self, t, s):
        t.add_row("CPU AVG", "[cyan]|[/cyan]", f"{s['cpu_average_percent']:.1f}%")
        t.add_row("CPU PEAK", "[cyan]|[/cyan]", f"{s['cpu_peak_percent']:.1f}%")

    def _ram_summary(self, t, s):
        t.add_row(
            "RAM AVG",
            "[cyan]|[/cyan]",
            f"{fmt_mem_new(s['ram_average_used'])} / {fmt_mem_new(s['ram_total'])}",
        )
        t.add_row(
            "RAM PEAK",
            "[cyan]|[/cyan]",
            f"{fmt_mem_new(s['ram_peak_used'])} / {fmt_mem_new(s['ram_total'])}",
        )

    def _gpu_summary(self, t, s):
        if not s["gpu_available"]:
            t.add_row("GPU", "[cyan]|[/cyan]", "[red]Not available[/red]")
            return

        t.add_row("GPU COUNT", "[cyan]|[/cyan]", str(s["gpu_total_count"]))
        t.add_row(
            "GPU UTIL AVG", "[cyan]|[/cyan]", f"{s['gpu_average_util_total']:.1f}%"
        )
        t.add_row("GPU UTIL PEAK", "[cyan]|[/cyan]", f"{s['gpu_peak_util_total']:.1f}%")
        t.add_row(
            "GPU MEM AVG",
            "[cyan]|[/cyan]",
            f"{fmt_mem_new(s['gpu_memory_average_used'])} / "
            f"{fmt_mem_new(s['gpu_memory_total'])}",
        )
        t.add_row(
            "GPU MEM PEAK",
            "[cyan]|[/cyan]",
            f"{fmt_mem_new(s['gpu_memory_peak_used'])} / "
            f"{fmt_mem_new(s['gpu_memory_total'])}",
        )

    def log_summary(self) -> None:
        s = self._compute_summary()
        console = Console()

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
