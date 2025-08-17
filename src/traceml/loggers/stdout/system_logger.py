from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from typing import Dict, Any, Optional

from .base_logger import BaseStdoutLogger
from .display_manager import SYSTEM_LAYOUT_NAME
from .display_manager import StdoutDisplayManager


class SystemStdoutLogger(BaseStdoutLogger):
    """
    Stdout logger showing:
      - CPU on one row
      - RAM on one row
      - GPU UTILIZATION block: AVG / MIN / MAX / IMBALANCE
      - GPU MEMORY block: HIGHEST / LOWEST / HighP (over 90%) and total GPUs
    """

    def __init__(self):
        super().__init__(name="System", layout_section_name=SYSTEM_LAYOUT_NAME)
        self._latest_env: Optional[Dict[str, Any]] = None
        self._latest_snapshot: Dict[str, Any] = {}

    def _format_percent(self, value: Any) -> str:
        try:
            return f"{float(value):.1f}%"
        except Exception:
            return "N/A"

    def _format_ratio(self, value: Any) -> str:
        try:
            return f"{float(value):.2f}"
        except Exception:
            return "N/A"

    def _format_memory(self, mb: Any) -> str:
        try:
            mbf = float(mb)
            if mbf >= 1024:
                return f"{mbf / 1024:.1f}G"
            return f"{mbf:.0f}M"
        except Exception:
            return "N/A"

    def _get_panel_renderable(self) -> Panel:
        env = self._latest_env or {}
        d = self._latest_snapshot or {}

        # CPU and RAM
        cpu_val = d.get("cpu_percent", 0.0)
        ram_used = d.get("ram_used", 0.0)
        ram_total = d.get("ram_total", 0.0)

        # GPU utilization metrics
        gpu_util_avg = d.get("gpu_util_avg_percent")
        gpu_util_min = d.get("gpu_util_min_nonzero_percent")
        gpu_util_max = d.get("gpu_util_max_percent")
        imbalance = d.get("gpu_util_imbalance_ratio")

        # GPU memory metrics
        gpu_mem_high = d.get("gpu_memory_highest_used")
        gpu_mem_low = d.get("gpu_memory_lowest_nonzero_used")
        high_pressure = d.get("gpu_count_high_pressure", 0)
        total_gpus = d.get("gpu_total_count", None)

        # Build table
        table = Table(box=None, show_header=False, padding=(0, 2))
        table.add_column(justify="left", style="bold magenta")

        # CPU row
        table.add_row(f"CPU: [white]{self._format_percent(cpu_val)}[/white]")

        # RAM row
        ram_pct = ""
        try:
            if ram_total and ram_total > 0:
                ram_pct = f" ({float(ram_used) / float(ram_total) * 100:.1f}%)"
        except Exception:
            ram_pct = ""
        table.add_row(
            f"RAM: [white]{self._format_memory(ram_used)}/{self._format_memory(ram_total)}{ram_pct}[/white]"
        )

        # GPU UTILIZATION block
        if gpu_util_avg is not None:
            table.add_row("")  # spacer
            table.add_row("[bold magenta]GPU UTILIZATION:[/bold magenta]")
            table.add_row(f"  AVG: [white]{self._format_percent(gpu_util_avg)}[/white]")
            if gpu_util_min not in (None, 0):
                table.add_row(
                    f"  MIN: [white]{self._format_percent(gpu_util_min)}[/white]"
                )
            if gpu_util_max not in (None, 0):
                table.add_row(
                    f"  MAX: [white]{self._format_percent(gpu_util_max)}[/white]"
                )
            if imbalance not in (None, 0):
                table.add_row(
                    f"  IMBALANCE: [white]{self._format_ratio(imbalance)}[/white]"
                )

        # GPU MEMORY block
        if (
            gpu_mem_high is not None
            or gpu_mem_low not in (None, 0)
            or total_gpus is not None
        ):
            table.add_row("")  # spacer
            table.add_row("[bold yellow]GPU MEMORY:[/bold yellow]")
            if gpu_mem_high is not None:
                table.add_row(f"  HIGHEST: {self._format_memory(gpu_mem_high)}")
            if gpu_mem_low not in (None, 0):
                table.add_row(f"  LOWEST: {self._format_memory(gpu_mem_low)}")
            if total_gpus is not None:
                table.add_row(f"  HighP: {high_pressure}/{total_gpus} (GPUs >90%)")
                table.add_row(f"  TOTAL GPUs: {total_gpus}")
            else:
                table.add_row(f"  HighP: {high_pressure} (GPUs >90%)")

        panel = Panel(
            table,
            title="Live System Metrics",
            title_align="center",
            border_style="dim white",
            width=80,
        )
        return panel

    def log_summary(self, summary: Dict[str, Any]):
        console = Console()
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold bright_red")
        table.add_column(justify="center", style="bright_red", no_wrap=True)
        table.add_column(justify="right", style="bold white")

        def fmt(key: str, value: Any) -> str:
            if value is None:
                return "N/A"
            if "percent" in key:
                try:
                    return f"{float(value):.1f}%"
                except Exception:
                    return "N/A"
            if "imbalance" in key:
                try:
                    return f"{float(value):.2f}"
                except Exception:
                    return "N/A"
            if "count" in key or isinstance(value, int):
                return str(value)
            if any(sub in key for sub in ("memory", "used", "available", "total")):
                try:
                    return f"{float(value):.2f} MB"
                except Exception:
                    return "N/A"
            return str(value)

        for key, value in summary.items():
            display_key = key.replace("_", " ").upper()
            display_value = fmt(key, value)
            table.add_row(display_key, "[bright_red]|[/bright_red]", display_value)

        panel = Panel(
            table,
            title=f"[bold bright_red]{self.name} - Final Summary",
            border_style="bright_red",
        )
        console.print(panel)
