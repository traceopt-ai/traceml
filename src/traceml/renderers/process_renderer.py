from typing import Dict, Any
import shutil

from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.database.database import Database
from traceml.renderers.display.cli_display_manager import (
    PROCESS_LAYOUT,
)
from traceml.utils.formatting import fmt_percent, fmt_mem_new


class ProcessRenderer(BaseRenderer):
    """
    Process panel.
    Expects BaseStdoutLogger.log() to receive a dict:
    """

    def __init__(self, database: Database):
        super().__init__(name="Process", layout_section_name=PROCESS_LAYOUT)
        self.db = database
        self._table = database.create_or_get_table("process")

    def _compute_snapshot(self) -> Dict[str, Any]:
        """Compute latest process metrics from the shared table."""
        latest = self.db.get_record_at_index("process", -1)
        if not latest:
            return {
                "cpu_used": 0.0,
                "cpu_logical_core_count": 0.0,
                "ram_used": 0.0,
                "ram_total": 0.0,
                "gpu_used": None,
                "gpu_reserved": None,
                "gpu_total": None,
            }

        gpu = latest.get("gpu_raw", {}) or {}
        if gpu:
            used_sum = sum(v.get("used", 0) for v in gpu.values())
            reserved_sum = sum(v.get("reserved", 0) for v in gpu.values())
            total_sum = sum(v.get("total", 0) for v in gpu.values())
        else:
            used_sum = reserved_sum = total_sum = None

        return {
            "cpu_used": latest.get("cpu_percent", 0.0),
            "cpu_logical_core_count": latest.get("cpu_logical_core_count", 0),
            "ram_used": latest.get("ram_used", 0.0),
            "ram_total": latest.get("ram_total", 0.0),
            "gpu_used": used_sum,
            "gpu_reserved": reserved_sum,
            "gpu_total": total_sum,
        }

    ## CLI rendering in Terminal
    def get_panel_renderable(self) -> Panel:
        proc = self._compute_snapshot()

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="white")
        table.add_column(justify="left", style="white")

        # process row
        proc_info = (
            f"[bold green]CPU[/bold green] {fmt_percent(proc['cpu_used'])}",
            f"[bold green]RAM[/bold green] {fmt_mem_new(proc['ram_used'])}",
        )
        table.add_row(*proc_info)

        if proc["gpu_total"]:
            gpu_str = (
                f"{fmt_mem_new(proc['gpu_used'])}/"
                f"{fmt_mem_new(proc['gpu_reserved'])}/"
                f"{fmt_mem_new(proc['gpu_total'])}"
            )
        else:
            gpu_str = "[red]Not available[/red]"

        table.add_row(" ")
        gpu_str = f"[bold green]GPU MEM (used/reserved/total)[/bold green] {gpu_str}"
        table.add_row(gpu_str)

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            table,
            title="[bold cyan]Process[/bold cyan]",
            title_align="center",
            border_style="cyan",
            width=panel_width,
        )

    # Notebook rendering
    def get_notebook_renderable(self) -> HTML:
        data = self._compute_snapshot()

        # GPU formatting
        if data["gpu_total"]:
            gpu_html = f"""
                <p><b>GPU MEM:</b>
                    {fmt_mem_new(data['gpu_used'])} /
                    {fmt_mem_new(data['gpu_reserved'])} /
                    {fmt_mem_new(data['gpu_total'])}
                </p>
            """
        else:
            gpu_html = """
                <p><b>GPU MEM:</b> <span style="color:red;">Not available</span></p>
            """

        html = f"""
        <div style="flex:1; border:2px solid #00bcd4; border-radius:8px; padding:10px;">
            <h4 style="color:#00bcd4; margin:0;">Process</h4>

            <p><b>CPU:</b> {fmt_percent(data['cpu_used'])}</p>
            <p><b>RAM:</b> {fmt_mem_new(data['ram_used'])}</p>

            {gpu_html}
        </div>
        """

        return HTML(
            f"<div style='display:flex; gap:20px; margin-top:10px;'>{html}</div>"
        )

    def get_dashboard_renderable(self):
        data = self._compute_snapshot()
        data["table"] = self._table
        return data

    def compute_summary(self) -> Dict[str, Any]:
        """
        Compute averages + peaks from table.
        Matches old ProcessSampler.get_summary() keys.
        """

        if not self._table:
            return {"total_samples": 0}

        cpu_vals, cpu_logical_cores = [], []
        ram_vals, ram_total = [], []
        gpu_used_vals, gpu_reserved_vals, gpu_total_vals = [], [], []

        for row in self._table:
            cpu_vals.append(row.get("cpu_percent", 0.0))
            ram_vals.append(row.get("ram_used", 0.0))
            cpu_logical_cores.append(row.get("cpu_logical_core_count", 0.0))
            ram_total.append(row.get("ram_total", 0.0))

            # GPU
            g = row.get("gpu_raw", {}) or {}
            if g:
                used = sum(v.get("used", 0) for v in g.values())
                reserved = sum(v.get("reserved", 0) for v in g.values())
                total = sum(v.get("total", 0) for v in g.values())
                gpu_used_vals.append(used)
                gpu_reserved_vals.append(reserved)
                gpu_total_vals.append(total)

        summary = {
            "total_samples": len(self._table),
            "cpu_average_percent": float(sum(cpu_vals)) / len(cpu_vals),
            "cpu_peak_percent": max(cpu_vals),
            "cpu_logical_core_count": max(
                cpu_logical_cores
            ),  # ProcessSampler no longer gives this 'live'
            "ram_average_used": float(sum(ram_vals)) / len(ram_vals),
            "ram_peak_used": max(ram_vals),
            "ram_total": max(ram_total),
            "is_GPU_available": bool(gpu_used_vals),
        }

        if gpu_used_vals:
            summary.update(
                {
                    "gpu_average_memory_used": float(sum(gpu_used_vals))
                    / len(gpu_used_vals),
                    "gpu_peak_memory_used": max(gpu_used_vals),
                    "gpu_average_memory_reserved": float(sum(gpu_reserved_vals))
                    / len(gpu_reserved_vals),
                    "gpu_peak_memory_reserved": max(gpu_reserved_vals),
                    "gpu_memory_total": float(max(gpu_total_vals)),
                }
            )

        return summary

    def _proc_cpu_summary(self, t: Table, block: dict) -> None:
        average_cpu_percent = block["cpu_average_percent"]
        peak_cpu_percent = block["cpu_peak_percent"]
        avg_cores_used = round(average_cpu_percent / 100, 2)
        peak_cores_used = round(peak_cpu_percent / 100, 2)

        t.add_row(
            "CPU AVG",
            "[magenta]|[/magenta]",
            f"{average_cpu_percent:.1f}% (~{avg_cores_used:.1f} cores of "
            f"{block['cpu_logical_core_count']:.1f} cores)",
        )
        t.add_row(
            "CPU PEAK",
            "[magenta]|[/magenta]",
            f"{peak_cpu_percent:.1f}% (~{peak_cores_used:.1f} cores of "
            f"{block['cpu_logical_core_count']:.1f} cores)",
        )

    def _proc_ram_summary(self, t, block: dict) -> None:
        # RAM Summary
        avg_ram_used = block["ram_average_used"]
        peak_ram_used = block["ram_peak_used"]
        total_ram = block["ram_total"]

        t.add_row(
            "RAM AVG",
            "[magenta]|[/magenta]",
            f"{fmt_mem_new(avg_ram_used)} / {fmt_mem_new(total_ram)} "
            f"({(avg_ram_used / total_ram) * 100:.1f}%)",
        )
        t.add_row(
            "RAM PEAK",
            "[magenta]|[/magenta]",
            f"{fmt_mem_new(peak_ram_used)} / {fmt_mem_new(total_ram)} "
            f"({(peak_ram_used / total_ram) * 100:.1f}%)",
        )

    def _proc_gpu_memory(self, t, block: dict) -> None:
        if block.get("is_GPU_available", False):
            total_gpu = block.get("gpu_memory_total", 0)

            t.add_row(
                "GPU MEM AVG (Used/Reserved/Total)",
                "[magenta]|[/magenta]",
                f"{fmt_mem_new(block['gpu_average_memory_used'])} / "
                f"{fmt_mem_new(block['gpu_average_memory_reserved'])} / "
                f"{fmt_mem_new(total_gpu)}",
            )
            t.add_row(
                "GPU MEM PEAK (Used/Reserved/Total)",
                "[magenta]|[/magenta]",
                f"{fmt_mem_new(block['gpu_peak_memory_used'])} /"
                f"({fmt_mem_new(block['gpu_peak_memory_reserved'])} / "
                f" {fmt_mem_new(total_gpu)}",
            )
        else:
            t.add_row("GPU", "[magenta]|[/magenta]", "[red]Not available[/red]")

    def log_summary(self) -> None:
        """Render the computed process summary to console (same style as before)."""
        console = Console()
        summary = self.compute_summary()

        t = Table.grid(padding=(0, 1))
        t.add_column(justify="left", style="magenta")
        t.add_column(justify="center", style="dim", no_wrap=True)
        t.add_column(justify="right", style="white")

        t.add_row(
            "TOTAL PROCESS SAMPLES",
            "[magenta]|[/magenta]",
            str(summary.get("total_samples", 0)),
        )

        if summary.get("total_samples", 0) > 0:
            self._proc_cpu_summary(t, summary)
            self._proc_ram_summary(t, summary)
            self._proc_gpu_memory(t, summary)

        console.print(
            Panel(
                t,
                title="[bold magenta]Process - Summary[/bold magenta]",
                border_style="magenta",
            )
        )
