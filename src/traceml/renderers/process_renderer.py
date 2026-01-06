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
from .utils import CARD_STYLE
import numpy as np


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
            title="[bold cyan]Process Metrics[/bold cyan]",
            title_align="center",
            border_style="cyan",
            width=panel_width,
        )

    # Notebook rendering
    def get_notebook_renderable(self) -> HTML:
        data = self._compute_snapshot()

        # --- GPU ---
        if data["gpu_total"]:
            gpu_html = f"""
                <div>
                    <b>GPU MEM:</b>
                    {fmt_mem_new(data['gpu_used'])} /
                    {fmt_mem_new(data['gpu_reserved'])} /
                    {fmt_mem_new(data['gpu_total'])}
                </div>
            """
        else:
            gpu_html = """
                <div><b>GPU MEM:</b>
                    <span style="color:red;">Not available</span>
                </div>
            """

        html = f"""
        <div style="{CARD_STYLE} ">
            <h4 style="color:#d47a00; margin-top:0;">Process Metrics</h4>

            <div><b>CPU:</b> {fmt_percent(data['cpu_used'])}</div>
            <div><b>RAM:</b> {fmt_mem_new(data['ram_used'])}</div>

            {gpu_html}
        </div>
        """

        return HTML(html)

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
        ram_vals, ram_total_vals = [], []
        gpu_used_vals, gpu_reserved_vals, gpu_total_vals = [], [], []
        gpu_single_used_peak_vals = []

        for row in self._table:

            cpu_vals.append(row.get("cpu_percent", 0.0))
            cpu_logical_cores.append(row.get("cpu_logical_core_count", 0.0))

            ram_vals.append(row.get("ram_used", 0.0))
            ram_total_vals.append(row.get("ram_total", 0.0))

            # GPU
            g = row.get("gpu_raw", {}) or {}
            if g:
                used = [v.get("used", 0) for v in g.values()]
                reserved = [v.get("reserved", 0) for v in g.values()]
                total = [v.get("total", 0) for v in g.values()]
                gpu_used_vals.append(sum(used))
                gpu_reserved_vals.append(sum(reserved))
                gpu_total_vals.append(sum(total))

                # failure-critical: single-device max
                gpu_single_used_peak_vals.append(max(used))

        cpu_vals = [v / 100.0 for v in cpu_vals]
        summary = {
            "total_samples": len(self._table),

            "cpu_cores_p50": round(float(np.median(cpu_vals)), 2),
            "cpu_cores_p95": round(float(np.percentile(cpu_vals, 95)), 2),
            "cpu_logical_core_count": float(np.max(cpu_logical_cores)),

            "ram_used_p95": round(float(np.percentile(ram_vals, 95)), 2),
            "ram_used_peak": round(float(np.max(ram_vals)), 2),
            "ram_total": float(np.max(ram_total_vals)),

            "is_GPU_available": bool(gpu_used_vals),
        }

        if gpu_used_vals:
            summary.update(
                {
                    # GPU memory used: sustained pressure + absolute risk
                    "gpu_mem_used_p95_total": round(float(np.percentile(gpu_used_vals, 95)), 2),
                    "gpu_mem_used_peak_total": round(float(np.max(gpu_used_vals)), 2),

                    # OOM risk (single device)
                    "gpu_mem_used_peak_single": round(float(np.max(gpu_single_used_peak_vals)), 2),

                    # GPU memory reserved
                    "gpu_mem_reserved_peak_total": round(float(np.max(gpu_reserved_vals)), 2),
                    "gpu_mem_total_capacity": float(np.max(gpu_total_vals)),

                }
            )

        return summary

    def _proc_cpu_summary(self, t: Table, block: dict) -> None:
        p50 = block.get("cpu_cores_p50", 0.0)
        p95 = block.get("cpu_cores_p95", 0.0)
        cores = block.get("cpu_logical_core_count", 0.0)

        t.add_row(
            f"CPU (p50 / p95)",
            "[magenta]|[/magenta]",
            f"{p50:.2f} / {p95:.2f} cores (of {cores:.0f})",
        )

    def _proc_ram_summary(self, t: Table, block: dict) -> None:
        p95 = block.get("ram_used_p95", 0.0)
        peak = block.get("ram_used_peak", 0.0)
        total = block.get("ram_total", 0.0)

        pct_p95 = (p95 / total * 100.0) if total else 0.0
        pct_peak = (peak / total * 100.0) if total else 0.0

        t.add_row(
            "RAM (p95 / peak)",
            "[magenta]|[/magenta]",
            f"{fmt_mem_new(p95)} ({pct_p95:.0f}%) / {fmt_mem_new(peak)} ({pct_peak:.0f}%) "
            f"(total {fmt_mem_new(total)})",
        )

    def _proc_gpu_memory(self, t: Table, block: dict) -> None:
        if not block.get("is_GPU_available", False):
            t.add_row("GPU", "[magenta]|[/magenta]", "[red]Not available[/red]")
            return

        total = block.get("gpu_mem_total_capacity", 0.0)

        used_p95_total = block.get("gpu_mem_used_p95_total", 0.0)
        used_peak_total = block.get("gpu_mem_used_peak_total", 0.0)
        used_peak_single = block.get("gpu_mem_used_peak_single", 0.0)
        reserved_peak_total = block.get("gpu_mem_reserved_peak_total", 0.0)

        # One compact row: totals + single-device risk + reserved
        t.add_row(
            "GPU MEM (p95 / peak)",
            "[magenta]|[/magenta]",
            f"total {fmt_mem_new(used_p95_total)} / {fmt_mem_new(used_peak_total)} "
            f"(cap {fmt_mem_new(total)}) | max device {fmt_mem_new(used_peak_single)} "
            f"| reserved peak {fmt_mem_new(reserved_peak_total)}",
        )

    def log_summary(self, path) -> None:
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
