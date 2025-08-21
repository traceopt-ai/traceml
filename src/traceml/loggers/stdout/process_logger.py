from rich.panel import Panel
from rich.table import Table
from typing import Dict, Any
from rich.console import Console

import psutil

from .base_logger import BaseStdoutLogger
from .display_manager import PROCESS_LAYOUT_NAME


class ProcessStdoutLogger(BaseStdoutLogger):
    """
    Stdout logger for Process-level (self PID) CPU, RAM and GPU usage metrics.
    """

    def __init__(self):
        super().__init__(name="Process", layout_section_name=PROCESS_LAYOUT_NAME)

        self._latest_snapshot = {
            "process_cpu_percent": 0.0,
            "process_ram": 0.0,
            "process_gpu_memory": None,
        }

        # Detect system CPU topology at logger initialization
        self.logical_cores = psutil.cpu_count(logical=True)
        self.physical_cores = psutil.cpu_count(logical=False)
        self.hyperthreaded = self.logical_cores > self.physical_cores
        self.threads_per_core = (
            self.logical_cores // self.physical_cores if self.physical_cores else 1
        )

    def _fmt_percent(self, v: Any) -> str:
        try:
            return f"{float(v):.1f}%"
        except Exception:
            return "N/A"

    def _fmt_mem_mb(self, v: Any) -> str:
        try:
            return f"{float(v):.0f}MB"
        except Exception:
            return "N/A"

    def _get_panel_renderable(self) -> Panel:
        """
        Generates the Rich Panel for live display:
        """
        d = self._latest_snapshot or {}

        cpu_val = d.get("process_cpu_percent", 0.0)
        ram_val = d.get("process_ram", 0.0)
        gpu_mem = d.get("process_gpu_memory")

        table = Table(box=None, show_header=False, padding=(0, 2))
        table.add_column(justify="center", style="bold magenta")
        table.add_column(justify="center", style="bold cyan")

        cpu_display_str = (
            f"CPU ({self.logical_cores} cores): {self._fmt_percent(cpu_val)}"
        )
        ram_display_str = f"RAM: {self._fmt_mem_mb(ram_val)}"

        table.add_row(cpu_display_str, ram_display_str)

        if gpu_mem is not None:
            gpu_display_str = f"GPU Memory: {self._fmt_mem_mb(gpu_mem)}"
            table.add_row("", gpu_display_str)

        return Panel(
            table,
            title="Live Process Metrics",
            title_align="center",
            border_style="dim white",
            width=80,
        )

    def log_summary(self, summary: Dict[str, Any]):
        """
        Logs the final summary.
        Works with ProcessSampler.get_summary() output:
          - total_process_samples
          - cpu_average_percent, cpu_peak_percent
          - ram_average_mb, ram_peak_mb
          - gpu_average_memory_mb, gpu_peak_memory_mb
        """
        console = Console()

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold cyan")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="bold white")

        def fmt_pair(key: str, value: Any) -> str:
            if value is None:
                return "N/A"
            # Percent-style keys
            if "percent" in key:
                try:
                    return f"{float(value):.1f}%"
                except Exception:
                    return "N/A"
            # Memory in MB
            if any(sub in key for sub in ("ram_", "gpu_", "memory")) and key.endswith(
                "_mb"
            ):
                try:
                    return f"{float(value):.2f} MB"
                except Exception:
                    return "N/A"
            # Counts / totals
            if "total" in key or "count" in key:
                try:
                    return str(int(value))
                except Exception:
                    return str(value)
            # Fallback
            return str(value)

        for key, value in summary.items():
            display_key = key.replace("_", " ").upper()
            display_value = fmt_pair(key, value)
            table.add_row(display_key, "[cyan]|[/cyan]", display_value)

        panel = Panel(
            table, title=f"[bold cyan]{self.name} - Final Summary", border_style="cyan"
        )
        console.print(panel)
