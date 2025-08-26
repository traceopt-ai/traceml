from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from typing import Dict, Any, Optional
import shutil

from .base_logger import BaseStdoutLogger
from .display_manager import SYSTEM_LAYOUT_NAME
from traceml.utils.formatting import fmt_mem, fmt_percent, fmt_ratio


class SystemStdoutLogger(BaseStdoutLogger):
    """
    Shows CPU, RAM, GPU Utilization, and GPU Memory in a clean panel.
    """

    def __init__(self):
        super().__init__(name="System", layout_section_name=SYSTEM_LAYOUT_NAME)
        self._latest_env: Optional[Dict[str, Any]] = None
        self._latest_snapshot: Dict[str, Any] = {}

    def _get_panel_renderable(self) -> Panel:
        d = self._latest_snapshot or {}

        # CPU & RAM
        cpu_val = d.get("cpu_percent", 0.0)
        ram_used = d.get("ram_used", 0.0)
        ram_total = d.get("ram_total", 0.0)

        ram_pct = ""
        try:
            if ram_total > 0:
                pct = (ram_used / ram_total) * 100
                ram_pct = f" ({pct:.1f}%)"
        except Exception:
            pass

        # GPU Utilization
        gpu_util_avg = d.get("gpu_util_avg_percent")
        gpu_util_min = d.get("gpu_util_min_nonzero_percent")
        gpu_util_max = d.get("gpu_util_max_percent")
        imbalance = d.get("gpu_util_imbalance_ratio")

        # GPU Memory
        gpu_mem_high = d.get("gpu_memory_highest_used")
        gpu_mem_low = d.get("gpu_memory_lowest_nonzero_used")
        high_pressure = d.get("gpu_count_high_pressure", 0)
        total_gpus = d.get("gpu_total_count", None)

        # Build compact table
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="white")

        # CPU + RAM row
        table.add_row(
            f"[cyan]CPU[/cyan]: {fmt_percent(cpu_val)}   "
            f"[cyan]RAM[/cyan]: {fmt_mem(ram_used)}/{fmt_mem(ram_total)}{ram_pct}"
        )

        # GPU Utilization block
        if gpu_util_avg is not None:
            table.add_row("")  # spacer
            util_parts = [f"AVG {fmt_percent(gpu_util_avg)}"]
            if gpu_util_min not in (None, 0):
                util_parts.append(f"MIN {fmt_percent(gpu_util_min)}")
            if gpu_util_max not in (None, 0):
                util_parts.append(f"MAX {fmt_percent(gpu_util_max)}")
            if imbalance not in (None, 0):
                util_parts.append(f"IMB {fmt_ratio(imbalance)}")

            table.add_row("[magenta]GPU Util[/magenta]: " + " | ".join(util_parts))

        # GPU Memory block
        if gpu_mem_high or gpu_mem_low or total_gpus is not None:
            table.add_row("")
            if gpu_mem_high is not None:
                table.add_row(f"[yellow]GPU Mem High[/yellow]: {fmt_mem(gpu_mem_high)}")
            if gpu_mem_low not in (None, 0):
                table.add_row(f"[yellow]GPU Mem Low[/yellow]: {fmt_mem(gpu_mem_low)}")
            if total_gpus is not None:
                table.add_row(f"[yellow]High Pressure[/yellow]: {high_pressure}/{total_gpus} (GPUs >90%)")
                table.add_row(f"[yellow]Total GPUs[/yellow]: {total_gpus}")

        # Adaptive width
        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(50, int(cols * 0.5)), 70)

        return Panel(
            table,
            title="[bold cyan]System Metrics[/bold cyan]",
            title_align="center",
            border_style="cyan",
            width=panel_width,
        )

    def log_summary(self, summary: Dict[str, Any]):
        console = Console()
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="cyan")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="white")

        def fmt(key: str, value: Any) -> str:
            if value is None:
                return "N/A"
            if "percent" in key:
                return fmt_percent(value)
            if "imbalance" in key:
                return fmt_ratio(value)
            if "count" in key or isinstance(value, int):
                return str(value)
            if any(sub in key for sub in ("memory", "used", "available", "total")):
                try:
                    return f"{float(value):.2f} MB"
                except Exception:
                    return "N/A"
            return str(value)

        for key, value in summary.items():
            table.add_row(key.replace("_", " ").upper(), "[cyan]|[/cyan]", fmt(key, value))

        panel = Panel(
            table,
            title=f"[bold cyan]{self.name} - Summary[/bold cyan]",
            border_style="cyan",
        )
        console.print(panel)
