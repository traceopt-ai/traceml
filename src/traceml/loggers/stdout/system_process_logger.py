from typing import Dict, Any, Optional
import shutil

from rich.panel import Panel
from rich.table import Table
from rich.console import Console
#

from .base_logger import BaseStdoutLogger
from .display_manager import SYSTEM_PROCESS_LAYOUT_NAME
from traceml.utils.formatting import fmt_mem, fmt_percent, fmt_ratio, fmt_mem_new


class SystemProcessStdoutLogger(BaseStdoutLogger):
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
        self._latest_env: Optional[Dict[str, Any]] = None
        self._latest_snapshot: Dict[str, Any] = {}

    def _get_panel_renderable(self) -> Panel:
        snaps = self._latest_snapshot or {}
        sysd = (snaps.get("SystemSampler") or {}).get("data") or {}
        procd = (snaps.get("ProcessSampler") or {}).get("data") or {}

        # ------- System (host) -------
        cpu_host = sysd.get("cpu_percent", 0.0)
        ram_used = sysd.get("ram_used", 0.0)
        ram_total = sysd.get("ram_total", 0.0)
        ram_pct_str = ""
        if (
            isinstance(ram_used, (int, float))
            and isinstance(ram_total, (int, float))
            and ram_total > 0
        ):
            try:
                ram_pct_str = f" ({ram_used * 100.0 / ram_total:.1f}%)"
            except Exception:
                ram_pct_str = ""

        # GPU (aggregate)
        gpu_util_avg = sysd.get("gpu_util_avg_percent")
        gpu_util_min = sysd.get("gpu_util_min_nonzero_percent")
        gpu_util_max = sysd.get("gpu_util_max_percent")
        imbalance = sysd.get("gpu_util_imbalance_ratio")

        gpu_mem_high = sysd.get("gpu_memory_highest_used")
        gpu_mem_low = sysd.get("gpu_memory_lowest_nonzero_used")
        high_pressure = sysd.get("gpu_count_high_pressure", 0)
        total_gpus = sysd.get("gpu_total_count")

        # ------- Process -------
        pid_cpu = procd.get("process_cpu_percent", 0.0)
        pid_ram = procd.get("process_ram", 0.0)  # MB
        pid_gpu_mem = procd.get("process_gpu_memory", None)  # MB or None

        # Build compact merged table
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="white")

        # Row 1: Host CPU/RAM vs Process CPU/RAM
        table.add_row(
            "[bold cyan]Host[/bold cyan] "
            f"CPU {fmt_percent(cpu_host)}   RAM {fmt_mem_new(ram_used)}/{fmt_mem_new(ram_total)}{ram_pct_str}     "
            "[bold cyan]Proc[/bold cyan] "
            f"CPU {fmt_percent(pid_cpu)}   RAM {fmt_mem(pid_ram)}"
        )

        # Row 2: GPU Util (if present)
        if total_gpus:
            util_bits = [f"AVG {fmt_percent(gpu_util_avg)}"]
            if gpu_util_min not in (None, 0):
                util_bits.append(f"MIN {fmt_percent(gpu_util_min)}")
            if gpu_util_max not in (None, 0):
                util_bits.append(f"MAX {fmt_percent(gpu_util_max)}")
            if imbalance not in (None, 0):
                util_bits.append(f"IMB {fmt_ratio(imbalance)}")
            table.add_row("[magenta]GPU Util[/magenta]: " + " | ".join(util_bits))

            # Row 3: GPU Mem (host aggregate + process)
            gpu_mem_parts = []
            gpu_mem_parts.append(f"High {fmt_mem(gpu_mem_high)}")
            gpu_mem_parts.append(f"Low {fmt_mem(gpu_mem_low)}")
            gpu_mem_parts.append(f">90% {high_pressure}/{total_gpus}")
            gpu_mem_parts.append(f"Proc {fmt_mem(pid_gpu_mem)}")

            if gpu_mem_parts:
                table.add_row("[yellow]GPU Mem[/yellow]: " + " | ".join(gpu_mem_parts))

        # Adaptive width
        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            table,
            title="[bold cyan]System + Process[/bold cyan]",
            title_align="center",
            border_style="cyan",
            width=panel_width,
        )

    def _render_system_summary(self, block: Dict[str, Any], console) -> None:
        t = Table.grid(padding=(0, 1))
        t.add_column(justify="left", style="cyan")
        t.add_column(justify="center", style="dim", no_wrap=True)
        t.add_column(justify="right", style="white")

        t.add_row("TOTAL SYSTEM SAMPLES", "[cyan]|[/cyan]", str(block["total_samples"]))
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

        # RAM summary
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

        console.print(
            Panel(
                t, title="[bold cyan]System - Summary[/bold cyan]", border_style="cyan"
            )
        )

    def _render_process_summary(self, block: Dict[str, Any], console) -> None:
        t = Table.grid(padding=(0, 1))
        t.add_column(justify="left", style="magenta")
        t.add_column(justify="center", style="dim", no_wrap=True)
        t.add_column(justify="right", style="white")

        for k, v in block.items():
            key = str(k).replace("_", " ").upper()
            if isinstance(v, (int, float)) and "percent" in k:
                val = fmt_percent(v)
            elif any(s in k for s in ("ram", "memory", "mb")) and isinstance(
                v, (int, float)
            ):
                val = fmt_mem(v)
            else:
                val = str(v)
            t.add_row(key, "[magenta]|[/magenta]", val)

        console.print(
            Panel(
                t,
                title="[bold magenta]Process - Summary[/bold magenta]",
                border_style="magenta",
            )
        )

    def log_summary(self, summary: Dict[str, Any]) -> None:
        sys_summary = (summary or {}).get("SystemSampler") or {}
        proc_summary = (summary or {}).get("ProcessSampler") or {}

        console = Console()

        if sys_summary:
            self._render_system_summary(sys_summary, console)
        if proc_summary:
            self._render_process_summary(proc_summary, console)
        if not sys_summary and not proc_summary:
            # fallback to generic
            self._render_system_summary(summary or {})
