import shutil
from collections import defaultdict
from typing import Dict, List
import numpy as np

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.database.database import Database
from traceml.renderers.display.cli_display_manager import STEPTIMER_LAYOUT
from traceml.renderers.utils import fmt_time_run


class StepTimerRenderer(BaseRenderer):
    """
    Renderer for timing statistics collected into the step-timer DB tables.

    For each event_name it computes:
      - cpu_avg_s, cpu_max_s  (from step_timer_cpu)
      - gpu_avg_s, gpu_max_s  (from all step_timer_cuda_* tables)

    Displayed "Avg (s)" and "Peak (s)" prefer GPU values if available,
    otherwise fall back to CPU values.
    """

    def __init__(self, database: Database, top_n: int = 5):
        super().__init__(
            name="Step Timers",
            layout_section_name=STEPTIMER_LAYOUT,
        )
        self.db = database
        self.top_n = top_n

    def _collect_cpu_times(self) -> Dict[str, List[float]]:
        """
        Read step_timer_cpu table and group durations (ms) by event_name.
        """
        cpu_table = self.db.create_or_get_table("step_timer_cpu")
        cpu_times: Dict[str, List[float]] = defaultdict(list)

        for row in cpu_table:
            name = row.get("event_name")
            if not name:
                continue
            dur_ms = float(row.get("duration_ms", 0.0))
            cpu_times[name].append(dur_ms)

        return cpu_times

    def _collect_gpu_times(self) -> Dict[str, List[float]]:
        """
        Read all step_timer_cuda_* tables and group durations (ms) by event_name,
        collapsing across all GPUs.
        """
        gpu_times: Dict[str, List[float]] = defaultdict(list)

        for table_name, rows in self.db.all_tables().items():
            if not table_name.startswith("step_timer_cuda"):
                continue

            for row in rows:
                event_name = row.get("event_name")
                if not event_name:
                    continue
                dur_ms = float(row.get("duration_ms", 0.0))
                gpu_times[event_name].append(dur_ms)

        return gpu_times

    def _compute_event_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute cpu_avg_s, cpu_max_s, gpu_avg_s, gpu_max_s for each event_name.
        """
        cpu_times = self._collect_cpu_times()
        gpu_times = self._collect_gpu_times()

        all_names = set(cpu_times.keys()) | set(gpu_times.keys())
        stats: Dict[str, Dict[str, float]] = {}

        for name in all_names:
            cpu_vals = cpu_times.get(name, [])
            gpu_vals = gpu_times.get(name, [])

            if cpu_vals:
                cpu_arr = np.array(cpu_vals, dtype=float)
                cpu_avg_s = float(cpu_arr.mean())
                cpu_max_s = float(cpu_arr.max())
            else:
                cpu_avg_s = 0.0
                cpu_max_s = 0.0

            if gpu_vals:
                gpu_arr = np.array(gpu_vals, dtype=float)
                gpu_avg_s = float(gpu_arr.mean())
                gpu_max_s = float(gpu_arr.max())
            else:
                gpu_avg_s = 0.0
                gpu_max_s = 0.0

            stats[name] = {
                "cpu_avg_s": cpu_avg_s,
                "cpu_max_s": cpu_max_s,
                "gpu_avg_s": gpu_avg_s,
                "gpu_max_s": gpu_max_s,
            }

        return stats

    def _pick_display_values(self, vals: Dict[str, float]) -> Dict[str, float]:
        """
        For rendering, prefer GPU metrics if any GPU data exists,
        otherwise fall back to CPU metrics.
        """
        gpu_avg = vals.get("gpu_avg_s", 0.0)
        gpu_max = vals.get("gpu_max_s", 0.0)
        cpu_avg = vals.get("cpu_avg_s", 0.0)
        cpu_max = vals.get("cpu_max_s", 0.0)

        if gpu_max > 0.0 or gpu_avg > 0.0:
            avg = gpu_avg
            peak = gpu_max
        else:
            avg = cpu_avg
            peak = cpu_max

        return {"avg_s": avg, "peak_s": peak}

    def _aggregate_top(
        self, stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Take full stats dict and return only top-N (N-1 + 'Other') for display.
        Sorting uses the display avg (GPU if present, else CPU).
        """
        if not stats:
            return {}

        # Compute a display avg for sorting
        def sort_key(item):
            name, vals = item
            disp = self._pick_display_values(vals)
            return disp["avg_s"]

        sorted_items = sorted(stats.items(), key=sort_key, reverse=True)

        if self.top_n is None or self.top_n <= 0 or len(sorted_items) <= self.top_n:
            # No 'Other' row needed
            return dict(sorted_items)

        # Keep top_n-1 explicit, rest aggregated as "Other"
        main_n = max(1, self.top_n - 1)
        top_items = sorted_items[:main_n]
        rest_items = sorted_items[main_n:]

        top_dict: Dict[str, Dict[str, float]] = dict(top_items)

        # Aggregate "Other" over remaining events
        if rest_items:
            cpu_avg_vals = [v["cpu_avg_s"] for _, v in rest_items]
            cpu_max_vals = [v["cpu_max_s"] for _, v in rest_items]
            gpu_avg_vals = [v["gpu_avg_s"] for _, v in rest_items]
            gpu_max_vals = [v["gpu_max_s"] for _, v in rest_items]

            other_stats = {
                "cpu_avg_s": float(np.mean(cpu_avg_vals)) if cpu_avg_vals else 0.0,
                "cpu_max_s": float(np.max(cpu_max_vals)) if cpu_max_vals else 0.0,
                "gpu_avg_s": float(np.mean(gpu_avg_vals)) if gpu_avg_vals else 0.0,
                "gpu_max_s": float(np.max(gpu_max_vals)) if gpu_max_vals else 0.0,
            }
            top_dict["Other"] = other_stats

        return top_dict

    def get_data(self) -> Dict[str, Dict[str, float]]:
        """
        Full pipeline:
          - read DB tables
          - compute per-event CPU/GPU averages + peaks
          - keep only top-N (plus 'Other')
        """
        stats = self._compute_event_stats()
        return self._aggregate_top(stats)

    # CLI rendering
    def get_panel_renderable(self) -> Panel:
        data = self.get_data()

        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("Event", justify="left", style="cyan")
        table.add_column("Avg (s)", justify="right", style="white")
        table.add_column("Peak (s)", justify="right", style="magenta")

        if data:
            for name, vals in data.items():
                disp = self._pick_display_values(vals)
                table.add_row(
                    f"[bold]{name}[/bold]",
                    fmt_time_run(disp['avg_s']),
                    fmt_time_run(disp['peak_s']),
                )
        else:
            table.add_row(
                "[dim]No step timings recorded[/dim]",
                "—",
                "—",
            )

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(80, int(cols * 0.6)), 120)

        return Panel(
            Group(table),
            title="[bold blue]Step Timers[/bold blue]",
            border_style="blue",
            width=panel_width,
        )

    # ---------- Notebook rendering ----------
    def get_notebook_renderable(self) -> HTML:
        data = self.get_data()

        rows = ""
        if data:
            # sort alphabetically for stable display
            for name in sorted(data.keys()):
                vals = data[name]
                disp = self._pick_display_values(vals)
                rows += f"""
                <tr style="border-bottom:1px solid #2c2c2c;">
                    <td style="text-align:left; color:#e0e0e0;">{name}</td>
                    <td style="text-align:right; color:#e0e0e0;">{fmt_time_run(disp['avg_s'])}</td>
                    <td style="text-align:right; color:#e0e0e0;">{fmt_time_run(disp['peak_s'])}</td>
                </tr>
                """
        else:
            rows = """
            <tr>
                <td colspan="3" style="text-align:center; color:gray;">
                    No step timings recorded
                </td>
            </tr>
            """

        html = f"""
        <div style="
            border:2px solid #00bcd4;
            border-radius:8px;
            padding:10px;
            margin-top:10px;
            background-color:#111111;
        ">
            <h4 style="color:#00bcd4; margin:0 0 8px 0;">Step Timers</h4>
            <table style="width:100%; border-collapse:collapse;">
                <thead style="background-color:#1e1e1e;">
                    <tr style="border-bottom:2px solid #00bcd4;">
                        <th style="text-align:left; color:#00bcd4;">Event</th>
                        <th style="text-align:right; color:#00bcd4;">Avg (s)</th>
                        <th style="text-align:right; color:#00bcd4;">Peak (s)</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """
        return HTML(html)

    def get_dashboard_renderable(self):
        data = self.get_data()
        return data


    def log_summary(self) -> None:
        """
        Print a one-shot summary of current DB stats to the console.
        """
        console = Console()
        data = self.get_data()

        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("Event", justify="left", style="cyan")
        table.add_column("Avg (s)", justify="right", style="white")
        table.add_column("Peak (s)", justify="right", style="magenta")

        if data:
            for name, vals in data.items():
                disp = self._pick_display_values(vals)
                table.add_row(
                    f"[bold]{name}[/bold]",
                    fmt_time_run(disp['avg_s']),
                    fmt_time_run(disp['peak_s']),
                )
        else:
            table.add_row(
                "[dim]No step timings recorded[/dim]",
                "—",
                "—",
            )

        panel = Panel(
            table,
            title="[bold blue]Step Timer Summary[/bold blue]",
            border_style="blue",
        )
        console.print(panel)
