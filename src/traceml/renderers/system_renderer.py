from typing import Dict, Any, List
import shutil
import numpy as np
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import SYSTEM_LAYOUT_NAME
from traceml.utils.formatting import fmt_percent, fmt_mem_new


class SystemRenderer(BaseRenderer):
    """
    Renderer that reads from the 'system' table (list of samples)
    and computes live + summary statistics for CPU, RAM, and GPU.
    """

    def __init__(self, table: List[Dict[str, Any]]):
        super().__init__(name="System", layout_section_name=SYSTEM_LAYOUT_NAME)
        self._table = table

    def _compute_snapshot(self) -> Dict[str, Any]:
        """Return latest system info for live display."""
        if not self._table:
            return {
                "cpu": 0.0,
                "ram_used": 0.0,
                "ram_total": 0.0,
                "gpu_available": False,
                "gpu_util_avg": None,
                "gpu_mem_used": None,
                "gpu_mem_total": None,
            }

        latest = self._table[-1]
        return {
            "cpu": latest.get("cpu_percent", 0.0),
            "ram_used": latest.get("ram_used", 0.0),
            "ram_total": latest.get("ram_total", 0.0),
            "gpu_available": latest.get("gpu_available", False),
            "gpu_util_avg": latest.get("gpu_util_avg"),
            "gpu_mem_used": latest.get("gpu_mem_sum_used"),
            "gpu_mem_total": latest.get("gpu_mem_total"),
        }

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute averages and peaks from the entire table."""
        if not self._table:
            return {"error": "no data", "total_samples": 0}

        cpu_vals = [x.get("cpu_percent", 0.0) for x in self._table]
        ram_vals = [x.get("ram_used", 0.0) for x in self._table]
        ram_total = self._table[-1].get("ram_total", 0.0)

        gpu_util_avg = [x.get("gpu_util_avg") for x in self._table if x.get("gpu_util_avg") is not None]
        gpu_util_max = [x.get("gpu_util_max") for x in self._table if x.get("gpu_util_max") is not None]
        gpu_mem_sum_used = [x.get("gpu_mem_sum_used") for x in self._table if x.get("gpu_mem_sum_used") is not None]
        gpu_mem_max_used = [x.get("gpu_mem_max_used") for x in self._table if x.get("gpu_mem_max_used") is not None]
        gpu_mem_total = [x.get("gpu_mem_total") for x in self._table if x.get("gpu_mem_total") is not None]

        gpu_available = self._table[-1].get("gpu_available", False)
        gpu_count = self._table[-1].get("gpu_count", 0)

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

        if gpu_available and gpu_util_avg:
            summary.update({
                "gpu_average_util_percent": round(float(np.mean(gpu_util_avg)), 2),
                "gpu_peak_util_percent": round(float(np.max(gpu_util_max)), 2),
                "gpu_memory_peak_used": round(float(np.max(gpu_mem_max_used)), 2),
                "gpu_memory_average_used": round(float(np.mean(gpu_mem_sum_used)), 2),
                "gpu_memory_total": round(float(np.mean(gpu_mem_total)), 2),
            })

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
        table.add_column(justify="left", style="white")

        sys_info = [
            f"[bold green]CPU[/bold green] {fmt_percent(data['cpu'])}",
            f"[bold green]RAM[/bold green] {fmt_mem_new(data['ram_used'])}/{fmt_mem_new(data['ram_total'])}{ram_pct_str}",
        ]

        if data["gpu_available"]:
            sys_info.append(f"[bold green]GPU[/bold green] {fmt_percent(data['gpu_util_avg'])}")
            sys_info.append(
                f"[bold green]GPU MEM[/bold green] {fmt_mem_new(data['gpu_mem_used'])}/{fmt_mem_new(data['gpu_mem_total'])}"
            )

        table.add_row("   ".join(sys_info))

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

        gpu_html = ""
        if data["gpu_available"]:
            gpu_html = f"""
            <p><b>GPU:</b> {fmt_percent(data['gpu_util_avg'])}</p>
            <p><b>GPU MEM:</b> {fmt_mem_new(data['gpu_mem_used'])}/{fmt_mem_new(data['gpu_mem_total'])}</p>
            """

        html = f"""
        <div style="border:2px solid #00bcd4; border-radius:8px; padding:10px;">
            <h4 style="color:#00bcd4; margin:0;">System</h4>
            <p><b>CPU:</b> {fmt_percent(data['cpu'])}</p>
            <p><b>RAM:</b> {fmt_mem_new(data['ram_used'])}/{fmt_mem_new(data['ram_total'])}</p>
            {gpu_html}
        </div>
        """
        return HTML(html)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        summary = self._compute_summary()
        console = Console()
        t = Table.grid(padding=(0, 1))
        t.add_column(justify="left", style="cyan")
        t.add_column(justify="center", style="dim", no_wrap=True)
        t.add_column(justify="right", style="white")

        t.add_row("TOTAL SYSTEM SAMPLES", "[cyan]|[/cyan]", str(summary["total_samples"]))
        if summary["total_samples"]:
            t.add_row("CPU AVG", "[cyan]|[/cyan]", f"{summary['cpu_average_percent']:.1f}%")
            t.add_row("CPU PEAK", "[cyan]|[/cyan]", f"{summary['cpu_peak_percent']:.1f}%")
            t.add_row(
                "RAM AVG",
                "[cyan]|[/cyan]",
                f"{fmt_mem_new(summary['ram_average_used'])}/{fmt_mem_new(summary['ram_total'])}",
            )
            t.add_row(
                "RAM PEAK",
                "[cyan]|[/cyan]",
                f"{fmt_mem_new(summary['ram_peak_used'])}/{fmt_mem_new(summary['ram_total'])}",
            )

            if summary["gpu_available"]:
                t.add_row("GPU COUNT", "[cyan]|[/cyan]", str(summary["gpu_total_count"]))
                t.add_row("GPU UTIL AVG", "[cyan]|[/cyan]", f"{summary['gpu_average_util_percent']:.1f}%")
                t.add_row("GPU UTIL PEAK", "[cyan]|[/cyan]", f"{summary['gpu_peak_util_percent']:.1f}%")
                t.add_row(
                    "GPU MEM AVG",
                    "[cyan]|[/cyan]",
                    f"{fmt_mem_new(summary['gpu_memory_average_used'])}/{fmt_mem_new(summary['gpu_memory_total'])}",
                )
                t.add_row(
                    "GPU MEM PEAK",
                    "[cyan]|[/cyan]",
                    f"{fmt_mem_new(summary['gpu_memory_peak_used'])}/{fmt_mem_new(summary['gpu_memory_total'])}",
                )
            else:
                t.add_row("GPU", "[cyan]|[/cyan]", "[red]Not available[/red]")

        console.print(Panel(t, title="[bold cyan]System - Summary[/bold cyan]", border_style="cyan"))
