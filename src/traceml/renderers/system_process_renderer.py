from typing import Dict, Any
import shutil

from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from IPython.display import HTML


from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import (
    SYSTEM_PROCESS_LAYOUT_NAME,
)
from traceml.utils.formatting import fmt_percent, fmt_mem_new


class SystemProcessRenderer(BaseRenderer):
    """
    Combined System + Process panel.
    Expects BaseStdoutLogger.log() to receive a dict:
      {
        "SystemSampler":  { "data": {...}, ... },
        "ProcessSampler": { "data": {...}, ... },
      }
    """

    def __init__(self):
        super().__init__(
            name="System/Process", layout_section_name=SYSTEM_PROCESS_LAYOUT_NAME
        )
        self._latest_snapshot: Dict[str, Any] = {}

    def get_data(self) -> Dict[str, Any]:
        snaps = self._latest_snapshot or {}
        sysd = (snaps.get("SystemSampler") or {}).get("data") or {}
        procd = (snaps.get("ProcessSampler") or {}).get("data") or {}
        return {
            "system": {
                "cpu": sysd.get("cpu_percent", 0.0),
                "ram_used": sysd.get("ram_used", 0.0),
                "ram_total": sysd.get("ram_total", 0.0),
                "gpu": {
                    "available": sysd.get("gpu_available", False),
                    "util_avg": sysd.get("gpu_util_avg"),
                    "mem_used": sysd.get("gpu_mem_sum_used"),
                    "mem_total": sysd.get("gpu_mem_total"),
                },
            },
            "process": {
                "cpu": procd.get("process_cpu_percent", 0.0),
                "ram": procd.get("process_ram", 0.0),
                "gpu_mem": procd.get("process_gpu_memory", None),
            },
        }

    ## CLI rendering in Terminal
    def get_panel_renderable(self) -> Panel:
        data = self.get_data()
        sys = data["system"]
        proc = data["process"]

        ram_pct_str = ""
        if sys["ram_total"]:
            try:
                ram_pct_str = f" ({sys['ram_used'] * 100.0 / sys['ram_total']:.1f}%)"
            except Exception:
                pass

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="white")
        table.add_column(justify="left", style="white")
        table.add_column(justify="left", style="white")

        # system row
        sys_info = [
            "[bold cyan]System[/bold cyan]",
            f"[bold green]CPU[/bold green] {fmt_percent(sys['cpu'])}",
            f"[bold green]RAM[/bold green] {fmt_mem_new(sys['ram_used'])}/{fmt_mem_new(sys['ram_total'])}{ram_pct_str}",
        ]
        if sys["gpu"]["available"]:
            sys_info.append(
                f"[bold green]GPU[/bold green] {fmt_percent(sys['gpu']['util_avg'])}"
            )
            sys_info.append(
                f"[bold green]GPU MEM[/bold green] {fmt_mem_new(sys['gpu']['mem_used'])}/{fmt_mem_new(sys['gpu']['mem_total'])}"
            )
        table.add_row("   ".join(sys_info))

        table.add_row("")  # gap

        # process row
        proc_info = [
            "[bold cyan]Process[/bold cyan]",
            f"[bold green]CPU[/bold green] {fmt_percent(proc['cpu'])}",
            f"[bold green]RAM[/bold green] {fmt_mem_new(proc['ram'])}",
        ]
        if proc["gpu_mem"] is not None:
            proc_info.append(
                f"[bold green]GPU MEM[/bold green] {fmt_mem_new(proc['gpu_mem'])}"
            )
        table.add_row("   ".join(proc_info))

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            table,
            title="[bold cyan]System + Process[/bold cyan]",
            title_align="center",
            border_style="cyan",
            width=panel_width,
        )

    # Notebook rendering
    def get_notebook_renderable(self) -> HTML:
        data = self.get_data()
        sys = data["system"]
        proc = data["process"]

        sys_html = f"""
        <div style="flex:1; border:2px solid #00bcd4; border-radius:8px; padding:10px;">
            <h4 style="color:#00bcd4; margin:0;">System</h4>
            <p><b>CPU:</b> {fmt_percent(sys['cpu'])}</p>
            <p><b>RAM:</b> {fmt_mem_new(sys['ram_used'])}/{fmt_mem_new(sys['ram_total'])}</p>
        </div>
        """

        proc_html = f"""
        <div style="flex:1; border:2px solid #00bcd4; border-radius:8px; padding:10px;">
            <h4 style="color:#00bcd4; margin:0;">Process</h4>
            <p><b>CPU:</b> {fmt_percent(proc['cpu'])}</p>
            <p><b>RAM:</b> {fmt_mem_new(proc['ram'])}</p>
        </div>
        """

        if proc["gpu_mem"] is not None:
            proc_html = proc_html.replace(
                "</div>",
                f"<p><b>GPU MEM:</b> {fmt_mem_new(proc['gpu_mem'])}</p></div>",
            )

        combined = f"""
        <div style="display:flex; gap:20px; margin-top:10px;">
            {sys_html}
            {proc_html}
        </div>
        """

        return HTML(combined)

    def _sys_cpu_summary(self, t, block):
        t.add_row(
            "CPU AVERAGE",
            "[cyan]|[/cyan]",
            f"{block['cpu_average_percent']:.1f}% of {block['cpu_logical_core_count']} cores",
        )
        t.add_row(
            "CPU PEAK",
            "[cyan]|[/cyan]",
            f"{block['cpu_peak_percent']:.1f}% of {block['cpu_logical_core_count']} cores",
        )

    def _sys_ram_summary(self, t, block):
        total_ram = block["ram_total"]
        avg_ram_used = block["ram_average_used"]
        peak_ram_used = block["ram_peak_used"]

        t.add_row(
            "RAM AVERAGE",
            "[cyan]|[/cyan]",
            f"{fmt_mem_new(avg_ram_used)} / {fmt_mem_new(total_ram)} "
            f"({(avg_ram_used / total_ram) * 100:.1f}%)",
        )

        t.add_row(
            "RAM PEAK",
            "[cyan]|[/cyan]",
            f"{fmt_mem_new(peak_ram_used)} / {fmt_mem_new(total_ram)} "
            f"({(peak_ram_used / total_ram) * 100:.1f}%)",
        )

    def _sys_gpu_memory(self, t, block):
        if block.get("is_GPU_available", False) and block.get("gpu_total_count", 0) > 0:
            total_gpu_mem = block.get("gpu_memory_total", 0)
            t.add_row(
                "GPU COUNT",
                "[cyan]|[/cyan]",
                str(block["gpu_total_count"]),
            )
            t.add_row(
                "GPU UTIL AVERAGE",
                "[cyan]|[/cyan]",
                f"{block['gpu_average_util_percent']:.1f}%",
            )
            t.add_row(
                "GPU UTIL PEAK",
                "[cyan]|[/cyan]",
                f"{block['gpu_peak_util_percent']:.1f}%",
            )
            t.add_row(
                "GPU MEMORY AVERAGE",
                "[cyan]|[/cyan]",
                f"{fmt_mem_new(block['gpu_memory_average_used'])} / {fmt_mem_new(total_gpu_mem)}",
            )
            t.add_row(
                "GPU MEMORY PEAK",
                "[cyan]|[/cyan]",
                f"{fmt_mem_new(block['gpu_memory_peak_used'])} / {fmt_mem_new(total_gpu_mem)}",
            )
        else:
            t.add_row("GPU", "[cyan]|[/cyan]", "[red]Not available[/red]")

    def _render_system_summary(self, block: Dict[str, Any], console) -> None:
        t = Table.grid(padding=(0, 1))
        t.add_column(justify="left", style="cyan")
        t.add_column(justify="center", style="dim", no_wrap=True)
        t.add_column(justify="right", style="white")

        t.add_row("TOTAL SYSTEM SAMPLES", "[cyan]|[/cyan]", str(block["total_samples"]))
        if block["total_samples"]:
            self._sys_cpu_summary(t, block)
            self._sys_ram_summary(t, block)
            self._sys_gpu_memory(t, block)
        console.print(
            Panel(
                t, title="[bold cyan]System - Summary[/bold cyan]", border_style="cyan"
            )
        )

    def _proc_cpu_summary(self, t: Table, block: dict) -> None:
        average_cpu_percent = block["average_cpu_percent"]
        peak_cpu_percent = block["peak_cpu_percent"]
        avg_cores_used = round(average_cpu_percent / 100, 2)
        peak_cores_used = round(peak_cpu_percent / 100, 2)

        t.add_row(
            "CPU AVERAGE",
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
        avg_ram_used = block["average_ram"]
        peak_ram_used = block["peak_ram"]
        total_ram = block["total_ram"]

        t.add_row(
            "RAM AVERAGE",
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
            total_gpu = block.get("total_gpu_memory_used", 0)

            t.add_row(
                "GPU MEMORY AVERAGE",
                "[magenta]|[/magenta]",
                f"{fmt_mem_new(block['average_gpu_memory_used'])} / {fmt_mem_new(total_gpu)}",
            )
            t.add_row(
                "GPU MEMORY PEAK",
                "[magenta]|[/magenta]",
                f"{fmt_mem_new(block['peak_gpu_memory_used'])} / {fmt_mem_new(total_gpu)}",
            )
        else:
            t.add_row("GPU", "[magenta]|[/magenta]", "[red]Not available[/red]")

    def _render_process_summary(self, block: Dict[str, Any], console) -> None:
        t = Table.grid(padding=(0, 1))
        t.add_column(justify="left", style="magenta")
        t.add_column(justify="center", style="dim", no_wrap=True)
        t.add_column(justify="right", style="white")

        # CPU summary
        t.add_row(
            "TOTAL PROCESS SAMPLES", "[magenta]|[/magenta]", str(block["total_samples"])
        )
        if block["total_samples"]:
            self._proc_cpu_summary(t, block)
            self._proc_ram_summary(t, block)
            self._proc_gpu_memory(t, block)

        console.print(
            Panel(
                t,
                title="[bold magenta]Process - Summary[/bold magenta]",
                border_style="magenta",
            )
        )

    def log_summary(self, summary: Dict[str, Any]) -> None:
        console = Console()

        sys_summary = (summary or {}).get("SystemSampler") or {}
        proc_summary = (summary or {}).get("ProcessSampler") or {}
        if sys_summary:
            self._render_system_summary(sys_summary, console)
        if proc_summary:
            self._render_process_summary(proc_summary, console)
