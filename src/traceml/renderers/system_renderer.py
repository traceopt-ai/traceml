from typing import Dict, Any
import shutil
import numpy as np
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.database.database import Database
from traceml.renderers.display.cli_display_manager import SYSTEM_LAYOUT
from traceml.utils.formatting import fmt_percent, fmt_mem_new
from .utils import append_text, CARD_STYLE


class SystemRenderer(BaseRenderer):
    """
    Renderer that reads from the 'system' table (list of samples)
    and computes live + summary statistics for CPU, RAM, and GPU.
    """

    def __init__(self, database: Database):
        super().__init__(name="System", layout_section_name=SYSTEM_LAYOUT)
        self.db = database
        self._table = database.create_or_get_table("system")

    def _compute_snapshot(self) -> Dict[str, Any]:
        """Return latest system info for live display."""
        latest = self.db.get_last_record("system")
        if not latest:
            return {
                "cpu": 0.0,
                "ram_used": 0.0,
                "ram_total": 0.0,
                "gpu_available": False,
                "gpu_util_total": None,
                "gpu_mem_used": None,
                "gpu_mem_total": None,
                "gpu_temp_max": None,
                "gpu_power_usage": None,
                "gpu_power_limit": None,
                "gpu_count": 0,
            }

        gpu_raw = latest.get("gpu_raw", {}) or {}

        # total util, mem & mem_total across all GPUs
        util_total = sum(v["util"] for v in gpu_raw.values()) if gpu_raw else None
        mem_used_total = (
            sum(v["mem_used"] for v in gpu_raw.values()) if gpu_raw else None
        )
        mem_total_total = (
            sum(v["mem_total"] for v in gpu_raw.values()) if gpu_raw else None
        )

        temp_max = max(v["temperature"] for v in gpu_raw.values()) if gpu_raw else None
        power_total = (
            sum(v["power_usage"] for v in gpu_raw.values()) if gpu_raw else None
        )
        power_limit_total = (
            sum(v["power_limit"] for v in gpu_raw.values()) if gpu_raw else None
        )

        return {
            "cpu": latest.get("cpu_percent", 0.0),
            "ram_used": latest.get("ram_used", 0.0),
            "ram_total": latest.get("ram_total", 0.0),
            "gpu_available": latest.get("gpu_available", False),
            "gpu_util_total": util_total,
            "gpu_mem_used": mem_used_total,
            "gpu_mem_total": mem_total_total,
            "gpu_temp_max": temp_max,
            "gpu_power_usage": power_total,
            "gpu_power_limit": power_limit_total,
            "gpu_count": latest.get("gpu_count", 0),
        }


    def _get_panel_cpu_row(self, table, data):
        ram_pct_str = ""
        if data["ram_total"]:
            ram_pct = data["ram_used"] * 100.0 / data["ram_total"]
            ram_pct_str = f" ({ram_pct:.1f}%)"
        table.add_row(
            f"[bold green]CPU[/bold green] {fmt_percent(data['cpu'])}",
            f"[bold green]RAM[/bold green] {fmt_mem_new(data['ram_used'])}/{fmt_mem_new(data['ram_total'])}{ram_pct_str}",
        )

    def _get_panel_gpu_row(self, table, data):
        if data["gpu_available"]:
            # first GPU row: util + memory
            table.add_row(
                f"[bold green]GPU UTIL[/bold green] {fmt_percent(data['gpu_util_total'])}",
                f"[bold green]GPU MEM[/bold green] {fmt_mem_new(data['gpu_mem_used'])}/{fmt_mem_new(data['gpu_mem_total'])}",
            )
            # second GPU row: temperature + power
            temp = data.get("gpu_temp_max")
            pu = data.get("gpu_power_usage")
            pl = data.get("gpu_power_limit")
            temp_str = (
                f"[bold green]GPU TEMP[/bold green] {temp:.1f}°C"
                if temp is not None
                else "[bold green]GPU TEMP[/bold green] N/A"
            )
            power_str = (
                f"[bold green]GPU POWER[/bold green] {pu:.1f}W / {pl:.1f}W"
                if pu is not None and pl is not None
                else "[bold green]GPU POWER[/bold green] N/A"
            )
            table.add_row(temp_str, power_str)
        else:
            table.add_row("[bold green]GPU[/bold green]", "[red]Not available[/red]")

    def get_panel_renderable(self) -> Panel:
        data = self._compute_snapshot()

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="white")
        table.add_column(justify="left", style="white")

        self._get_panel_cpu_row(table, data)
        self._get_panel_gpu_row(table, data)

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            table,
            title="[bold cyan]System Metrics[/bold cyan]",
            title_align="center",
            border_style="cyan",
            width=panel_width,
        )

    def _get_notebook_cpu_row(self, data):
        ram_pct = ""
        if data["ram_total"]:
            try:
                rp = data["ram_used"] * 100.0 / data["ram_total"]
                ram_pct = f" ({rp:.1f}%)"
            except Exception:
                pass

        cpu_ram_html = f"""
                    <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                        margin-bottom:12px;
                    ">
                        <div>
                            <b>CPU:</b> {fmt_percent(data['cpu'])}
                        </div>
                        <div>
                            <b>RAM:</b>
                            {fmt_mem_new(data['ram_used'])} / {fmt_mem_new(data['ram_total'])}{ram_pct}
                        </div>
                    </div>
                """
        return cpu_ram_html

    def get_notebook_renderable(self) -> HTML:
        data = self._compute_snapshot()

        # --- CPU + RAM ---
        cpu_ram_html = self._get_notebook_cpu_row(data)

        # --- GPU ---
        if data["gpu_available"]:
            gpu_util_html = f"{fmt_percent(data['gpu_util_total'])}"
            gpu_mem_html = f"{fmt_mem_new(data['gpu_mem_used'])} / {fmt_mem_new(data['gpu_mem_total'])}"

            temp = data.get("gpu_temp_max")
            pu = data.get("gpu_power_usage")
            pl = data.get("gpu_power_limit")

            temp_html = f"{temp:.1f}°C" if temp is not None else "N/A"
            power_html = (
                f"{pu:.1f}W / {pl:.1f}W" if pu is not None and pl is not None else "N/A"
            )

            gpu_section = f"""
                <div style="
                    display:flex; 
                    flex-direction:column; 
                    gap:6px;
                ">
                    <div>
                        <b>GPU Util:</b> {gpu_util_html}<br>
                        <b>GPU Mem:</b> {gpu_mem_html}
                    </div>
                    <div>
                        <b>GPU Temp:</b> {temp_html}<br>
                        <b>GPU Power:</b> {power_html}
                    </div>
                </div>
            """
        else:
            gpu_section = """
                <div><b>GPU:</b> <span style='color:red;'>Not available</span></div>
            """

        # --- Final card ---
        html = f"""
        <div style="{CARD_STYLE}">
            <h4 style="color:#d47a00;"; margin-top:0;">System Metrics</h4>

            {cpu_ram_html}

            {gpu_section}
        </div>
        """

        return HTML(html)

    def get_dashboard_renderable(self):
        """
        Return an object suitable for Streamlit's st.write():
        - str / Markdown
        - pandas.DataFrame
        - matplotlib/plotly figure
        - etc.
        """
        data = self._compute_snapshot()
        data["table"] = self._table
        return data


    def _compute_summary(self) -> Dict[str, Any]:
        """Compute averages and peaks from the entire table."""
        if not self._table:
            return {"error": "no data", "total_samples": 0}

        gpu_available = self._table[-1].get("gpu_available", False)
        gpu_count = self._table[-1].get("gpu_count", 0)

        cpu_vals = [x.get("cpu_percent", 0.0) for x in self._table]
        ram_vals = [x.get("ram_used", 0.0) for x in self._table]
        ram_total = self._table[-1].get("ram_total", 0.0)

        summary = {
            "total_samples": len(self._table),

            "cpu_avg_percent": round(float(np.mean(cpu_vals)), 2),
            "cpu_median_percent": round(float(np.median(cpu_vals)), 2),
            "cpu_p95_percent": round(float(np.percentile(cpu_vals, 95)), 2),

            "ram_avg_used": round(float(np.mean(ram_vals)), 2),
            "ram_median_used": round(float(np.median(ram_vals)), 2),
            "ram_p95_used": round(float(np.percentile(ram_vals, 95)), 2),
            "ram_peak_used": round(float(np.max(ram_vals)), 2),

            "ram_total": ram_total,
            "gpu_available": gpu_available,
            "gpu_total_count": gpu_count,
        }

        util_totals = []
        mem_used_totals = []
        mem_total_totals = []
        max_single_gpu_mem = []
        temp_max_vals = []
        power_usage_totals = []
        power_limit_totals = []

        for x in self._table:
            gpu_raw = x.get("gpu_raw", {}) or {}
            if gpu_raw:
                # aggregate over GPUs (job-level)
                util_totals.append(sum(v["util"] for v in gpu_raw.values()))
                mem_used_totals.append(sum(v["mem_used"] for v in gpu_raw.values()))
                mem_total_totals.append(sum(v["mem_total"] for v in gpu_raw.values()))
                power_usage_totals.append(sum(v["power_usage"] for v in gpu_raw.values()))
                power_limit_totals.append(sum(v["power_limit"] for v in gpu_raw.values()))

                # failure-critical per-device signals
                max_single_gpu_mem.append(
                    max(v["mem_used"] for v in gpu_raw.values())
                )
                temp_max_vals.append(max(v["temperature"] for v in gpu_raw.values()))

        if gpu_available and util_totals:
            summary.update(
                {
                    # GPU utilization (aggregate)
                    "gpu_util_total_avg": round(float(np.mean(util_totals)), 2),
                    "gpu_util_total_median": round(float(np.median(util_totals)), 2),
                    "gpu_util_total_peak": round(float(np.max(util_totals)), 2),

                    # GPU memory (aggregate footprint over time)
                    # Near-worst sustained + absolute peak
                    "gpu_mem_total_p95": round(
                        float(np.percentile(mem_used_totals, 95)), 2
                    ),
                    "gpu_mem_total_peak": round(float(np.max(mem_used_totals)), 2),
                    "gpu_mem_total_capacity": round(float(np.mean(mem_total_totals)), 2),

                    # GPU memory (single-device OOM risk)
                    # Absolute risk only
                    "gpu_mem_single_peak": round(float(np.max(max_single_gpu_mem)), 2),

                    # GPU temperature (single-device thermal risk)
                    # Typical + absolute
                    "gpu_temp_median": round(float(np.median(temp_max_vals)), 2),
                    "gpu_temp_peak": round(float(np.max(temp_max_vals)), 2),

                    # GPU power (node-level, informational)
                    # Typical + absolute
                    "gpu_power_avg": round(float(np.mean(power_usage_totals)), 2),
                    "gpu_power_peak": round(float(np.max(power_usage_totals)), 2),
                    "gpu_power_limit": round(float(np.mean(power_limit_totals)), 2),
                }
            )
        return summary


    def _cpu_summary(self, t, s):
        t.add_row(
            "CPU (avg / p50 / p95)",
            "[cyan]|[/cyan]",
            f"{s['cpu_avg_percent']:.1f}% / {s['cpu_median_percent']:.1f}% / {s['cpu_p95_percent']:.1f}%",
        )

    def _ram_summary(self, t, s):
        t.add_row(
            "RAM (avg / p95 / peak / total)",
            "[cyan]|[/cyan]",
            f"{fmt_mem_new(s['ram_avg_used'])} / "
            f"{fmt_mem_new(s['ram_p95_used'])} / "
            f"{fmt_mem_new(s['ram_peak_used'])} / "
            f"{fmt_mem_new(s['ram_total'])}",
        )

    def _gpu_summary(self, t, s):
        if not s["gpu_available"]:
            t.add_row("GPU", "[cyan]|[/cyan]", "[red]Not available[/red]")
            return

        t.add_row("GPU COUNT", "[cyan]|[/cyan]", str(s["gpu_total_count"]))
        t.add_row(
            "GPU UTIL (avg / p50 / peak)",
            "|",
            f"{s['gpu_util_total_avg']:.1f}% / "
            f"{s['gpu_util_total_median']:.1f}% / "
            f"{s['gpu_util_total_peak']:.1f}%",
        )

        t.add_row(
            "GPU MEMORY (p95 / peak / capacity)",
            "|",
            f"{fmt_mem_new(s['gpu_mem_total_p95'])} / "
            f"{fmt_mem_new(s['gpu_mem_total_peak'])}  / "
            f"{fmt_mem_new(s['gpu_mem_total_capacity'])}",
        )
        t.add_row(
            "GPU MEMORY SINGLE DEVICE (peak)",
            "[cyan]|[/cyan]",
            f"{fmt_mem_new(s['gpu_mem_single_peak'])}",
        )

        t.add_row(
            "GPU TEMP (p50 / peak)",
            "[cyan]|[/cyan]",
            f"{s['gpu_temp_median']:.1f}°C / {s['gpu_temp_peak']:.1f}°C",
        )

        t.add_row(
            "GPU POWER (avg / peak / limit)",
            "[cyan]|[/cyan]",
            f"{s['gpu_power_avg']:.0f}W / "
            f"{s['gpu_power_peak']:.0f}W / "
            f"{s['gpu_power_limit']:.0f}W",
        )

    def log_summary(self, path) -> None:
        s = self._compute_summary()
        console = Console(record=True)

        t = Table.grid(padding=(0, 1))
        t.add_column(style="cyan")
        t.add_column(style="dim", no_wrap=True)
        t.add_column(style="white")

        t.add_row("TOTAL SYSTEM SAMPLES", "[cyan]|[/cyan]", str(s["total_samples"]))

        if s["total_samples"]:
            self._cpu_summary(t, s)
            self._ram_summary(t, s)
            self._gpu_summary(t, s)

        console.print(
            Panel(
                t, title="[bold cyan]System - Summary[/bold cyan]", border_style="cyan"
            )
        )
        if path:
            text = console.export_text(clear=False)
            append_text(path, "\n" + text + "\n")
