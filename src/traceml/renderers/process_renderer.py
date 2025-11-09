from typing import Dict, Any
import shutil

from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from IPython.display import HTML


from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import (
    PROCESS_LAYOUT_NAME,
)
from traceml.utils.formatting import fmt_percent, fmt_mem_new


class ProcessRenderer(BaseRenderer):
    """
    Process panel.
    Expects BaseStdoutLogger.log() to receive a dict:
    """

    def __init__(self):
        super().__init__(
            name="Process", layout_section_name=PROCESS_LAYOUT_NAME
        )
        self._latest_snapshot: Dict[str, Any] = {}

    def get_data(self) -> Dict[str, Any]:
        snaps = self._latest_snapshot or {}
        procd = (snaps.get("ProcessSampler") or {}).get("data") or {}
        return {
            "process": {
                "cpu": procd.get("process_cpu_percent", 0.0),
                "ram": procd.get("process_ram", 0.0),
                "gpu_mem": procd.get("process_gpu_memory", None),
            },
        }

    ## CLI rendering in Terminal
    def get_panel_renderable(self) -> Panel:
        data = self.get_data()
        proc = data["process"]

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="white")
        table.add_column(justify="left", style="white")
        table.add_column(justify="left", style="white")

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
        proc = data["process"]

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
            {proc_html}
        </div>
        """

        return HTML(combined)

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
            total_gpu = block.get("total_gpu_memory", 0)

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
        proc_summary = (summary or {}).get("ProcessSampler") or {}
        if proc_summary:
            self._render_process_summary(proc_summary, console)
