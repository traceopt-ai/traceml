from rich.panel import Panel
from rich.table import Table
from typing import Dict, Any
from rich.console import Console
import shutil
import os
import psutil

from .base_logger import BaseStdoutLogger
from .display_manager import PROCESS_LAYOUT_NAME
from traceml.utils.formatting import fmt_mem, fmt_percent


class ProcessStdoutLogger(BaseStdoutLogger):
    """
    Compact process-level logger showing CPU%, RAM, and GPU memory for the current PID.
    """

    def __init__(self):
        super().__init__(name="Process", layout_section_name=PROCESS_LAYOUT_NAME)

        self._latest_snapshot: Dict[str, Any] = {
            "process_cpu_percent": 0.0,
            "process_ram": 0.0,  # MB
            "process_gpu_memory": None,  # MB or None
        }

        # CPU topology
        self.pid = os.getpid()
        self.logical_cores = psutil.cpu_count(logical=True) or 0
        self.physical_cores = psutil.cpu_count(logical=False) or 0
        self.hyperthreaded = (
            self.physical_cores > 0 and self.logical_cores > self.physical_cores
        )
        self.threads_per_core = (
            (self.logical_cores // self.physical_cores) if self.physical_cores else 1
        )

    def _get_panel_renderable(self) -> Panel:
        d = self._latest_snapshot or {}
        cpu_val = d.get("process_cpu_percent", 0.0)
        ram_mb = d.get("process_ram", 0.0)
        gpu_mem_mb = d.get("process_gpu_memory", None)

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="white")

        # CPU row
        cores_str = f"{self.logical_cores}" if self.logical_cores else "n/a"
        table.add_row(
            f"[cyan]CPU[/cyan]: {fmt_percent(cpu_val)}   "
            f"[cyan]Threads[/cyan]: {cores_str}"
        )

        # RAM row
        table.add_row(f"[cyan]RAM[/cyan]: {fmt_mem(ram_mb)}")

        # GPU memory row (only if present)
        if gpu_mem_mb is not None:
            table.add_row(f"[magenta]GPU Mem[/magenta]: {fmt_mem(gpu_mem_mb)}")

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(50, int(cols * 0.5)), 70)

        return Panel(
            table,
            title=f"[bold cyan]Process (PID {self.pid})[/bold cyan]",
            title_align="center",
            border_style="cyan",
            width=panel_width,
        )

    def log_summary(self, summary: Dict[str, Any]):
        """
        Works with ProcessSampler.get_summary() output:
          - total_process_samples
          - cpu_average_percent, cpu_peak_percent
          - ram_average, ram_peak  (MB)
          - gpu_average_memory, gpu_peak_memory (MB)
        """
        console = Console()

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="cyan")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="white")

        def fmt(key: str, value: Any) -> str:
            if value is None:
                return "N/A"
            k = str(key)
            if "percent" in k:
                return fmt_percent(value)
            if any(s in k for s in ("ram", "gpu", "memory")):
                return fmt_mem(value)
            if "total" in k or "count" in k:
                try:
                    return str(int(value))
                except Exception:
                    return str(value)
            return str(value)

        for key, value in summary.items():
            table.add_row(
                key.replace("_", " ").upper(), "[cyan]|[/cyan]", fmt(key, value)
            )

        panel = Panel(
            table,
            title=f"[bold cyan]{self.name} - Summary[/bold cyan]",
            border_style="cyan",
        )
        console.print(panel)
