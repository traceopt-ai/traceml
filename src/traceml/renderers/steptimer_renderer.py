import shutil
from collections import defaultdict
from typing import Dict, List, Optional
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
    Renderer for user-defined step timers.

    Keeps top-N events (plus aggregated 'Other') and shows rolling statistics:
      - Last
      - p50(100)
      - p95(100)
      - Avg(100)
      - Trend (sign of Avg(100) vs Avg(200))
      - Device (GPU preferred if available, else CPU)
    """

    def __init__(self, database: Database, top_n: int = 5):
        super().__init__(
            name="Step Timers",
            layout_section_name=STEPTIMER_LAYOUT,
        )
        self.db = database
        self.top_n = top_n

    def _is_internal(self, name: str) -> bool:
        return name.startswith("_traceml_internal:")

    def _collect_cpu_series(self) -> Dict[str, List[float]]:
        """step_timer_cpu: event_name -> [duration_ms,...]"""
        table = self.db.create_or_get_table("step_timer_cpu")
        out: Dict[str, List[float]] = defaultdict(list)

        for row in table:
            name = row.get("event_name")
            if not name:
                continue
            out[name].append(float(row.get("duration_ms", 0.0)))

        return out

    def _collect_gpu_series(self) -> Dict[str, List[float]]:
        """All step_timer_cuda_* tables collapsed: event_name -> [duration_ms,...]"""
        out: Dict[str, List[float]] = defaultdict(list)

        for table_name, rows in self.db.all_tables().items():
            if not table_name.startswith("step_timer_cuda"):
                continue

            for row in rows:
                name = row.get("event_name")
                if not name:
                    continue
                out[name].append(float(row.get("duration_ms", 0.0)))

        return out

    def _collect_series(self) -> Dict[str, Dict[str, List[float]]]:
        """
        event_name -> {"cpu": [...], "gpu": [...]}
        (Does not filter internal names; add filtering here if needed.)
        """
        cpu = self._collect_cpu_series()
        gpu = self._collect_gpu_series()

        series: Dict[str, Dict[str, List[float]]] = {}
        for name in set(cpu) | set(gpu):
            if self._is_internal(name):
                continue
            series[name] = {"cpu": cpu.get(name, []), "gpu": gpu.get(name, [])}
        return series

    def _aggregate_top_series(
        self, series: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Sort events by display mean (GPU if present else CPU), keep top_n-1,
        aggregate remainder into "Other" by summing per-step durations.
        """
        if not series:
            return {}

        def sort_key(item):
            _, vals = item
            arr = vals["gpu"] if vals["gpu"] else vals["cpu"]
            return float(np.mean(arr)) if arr else 0.0

        items = sorted(series.items(), key=sort_key, reverse=True)

        if self.top_n is None or self.top_n <= 0 or len(items) <= self.top_n:
            return dict(items)

        main_n = max(1, self.top_n - 1)
        top_items = items[:main_n]
        rest_items = items[main_n:]

        out: Dict[str, Dict[str, List[float]]] = dict(top_items)

        other_cpu: List[float] = []
        other_gpu: List[float] = []

        max_len = max(
            max((len(v["cpu"]) for _, v in rest_items), default=0),
            max((len(v["gpu"]) for _, v in rest_items), default=0),
        )

        for i in range(max_len):
            cpu_sum = 0.0
            gpu_sum = 0.0
            for _, v in rest_items:
                if i < len(v["cpu"]):
                    cpu_sum += v["cpu"][i]
                if i < len(v["gpu"]):
                    gpu_sum += v["gpu"][i]
            if cpu_sum > 0:
                other_cpu.append(cpu_sum)
            if gpu_sum > 0:
                other_gpu.append(gpu_sum)

        out["Other"] = {"cpu": other_cpu, "gpu": other_gpu}
        return out

    @staticmethod
    def _safe_percentile(x: np.ndarray, q: float) -> float:
        if x.size == 0:
            return 0.0
        return float(np.percentile(x, q))

    def _compute_row_stats(
        self, cpu_vals: List[float], gpu_vals: List[float]
    ) -> Dict[str, object]:
        """
        Compute display stats for one event:
          - choose GPU series if present else CPU
          - rolling window 100 for p50/p95/avg
          - rolling window 200 for trend
          - trend: '+' if Avg(100) > Avg(200), '-' otherwise (only if >=200)
        Returns a dict suitable for all renderers.
        """
        if gpu_vals:
            arr = np.asarray(gpu_vals, dtype=np.float64)
            device = "GPU"
        else:
            arr = np.asarray(cpu_vals, dtype=np.float64)
            device = "CPU"

        if arr.size == 0:
            return {
                "last": 0.0,
                "p50_100": 0.0,
                "p95_100": 0.0,
                "avg_100": 0.0,
                "trend": "",
                "device": device,
            }

        last = float(arr[-1])

        win100 = arr[-min(100, arr.size) :]
        p50 = self._safe_percentile(win100, 50)
        p95 = self._safe_percentile(win100, 95)
        avg100 = float(win100.mean())

        trend = ""
        if arr.size >= 200:
            win200 = arr[-200:]
            avg200 = float(win200.mean())
            # Sign only, as requested
            trend = "+" if avg100 > avg200 else "-"

        return {
            "last": last,
            "p50_100": p50,
            "p95_100": p95,
            "avg_100": avg100,
            "trend": trend,
            "device": device,
        }

    def get_data(self) -> Dict[str, Dict[str, object]]:
        """
        Returns:
          event_name -> {
            "last", "p50_100", "p95_100", "avg_100", "trend", "device"
          }
        (Top-N + Other already applied.)
        """
        series = self._aggregate_top_series(self._collect_series())
        out: Dict[str, Dict[str, object]] = {}

        for name, vals in series.items():
            out[name] = self._compute_row_stats(
                vals.get("cpu", []), vals.get("gpu", [])
            )

        return out

    def get_panel_renderable(self) -> Panel:
        data = self.get_data()

        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("Event", justify="left", style="cyan")
        table.add_column("Last", justify="right")
        table.add_column("p50(100)", justify="right")
        table.add_column("p95(100)", justify="right")
        table.add_column("Avg(100)", justify="right")
        table.add_column("Trend", justify="center")
        table.add_column("Device", justify="center", style="magenta")

        if not data:
            table.add_row(
                "[dim]No step timers recorded[/dim]", "—", "—", "—", "—", "", "—"
            )
        else:
            # Stable display order: by Avg(100) descending, "Other" last if present
            items = list(data.items())
            items.sort(key=lambda kv: float(kv[1].get("avg_100", 0.0)), reverse=True)
            if any(name == "Other" for name, _ in items):
                items = [kv for kv in items if kv[0] != "Other"] + [
                    kv for kv in items if kv[0] == "Other"
                ]

            for name, s in items:
                table.add_row(
                    f"[bold]{name}[/bold]",
                    fmt_time_run(float(s["last"])),
                    fmt_time_run(float(s["p50_100"])),
                    fmt_time_run(float(s["p95_100"])),
                    fmt_time_run(float(s["avg_100"])),
                    str(s["trend"]),
                    str(s["device"]),
                )

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 110)

        return Panel(
            Group(table),
            title="[bold blue]Trace Timers[/bold blue]",
            border_style="blue",
            width=panel_width,
        )

    def get_notebook_renderable(self) -> HTML:
        data = self.get_data()

        if not data:
            rows = """
            <tr>
                <td colspan="7" style="text-align:center; color:gray;">
                    No step timers recorded
                </td>
            </tr>
            """
        else:
            items = list(data.items())
            items.sort(key=lambda kv: float(kv[1].get("avg_100", 0.0)), reverse=True)
            if any(name == "Other" for name, _ in items):
                items = [kv for kv in items if kv[0] != "Other"] + [
                    kv for kv in items if kv[0] == "Other"
                ]

            rows = ""
            for name, s in items:
                rows += f"""
                <tr style="border-bottom:1px solid #2c2c2c;">
                    <td style="text-align:left; color:#e0e0e0;">{name}</td>
                    <td style="text-align:right; color:#e0e0e0;">{fmt_time_run(float(s["last"]))}</td>
                    <td style="text-align:right; color:#e0e0e0;">{fmt_time_run(float(s["p50_100"]))}</td>
                    <td style="text-align:right; color:#e0e0e0;">{fmt_time_run(float(s["p95_100"]))}</td>
                    <td style="text-align:right; color:#e0e0e0;">{fmt_time_run(float(s["avg_100"]))}</td>
                    <td style="text-align:center; color:#e0e0e0;">{s["trend"]}</td>
                    <td style="text-align:center; color:#e0e0e0;">{s["device"]}</td>
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
                        <th style="text-align:right; color:#00bcd4;">Last</th>
                        <th style="text-align:right; color:#00bcd4;">p50(100)</th>
                        <th style="text-align:right; color:#00bcd4;">p95(100)</th>
                        <th style="text-align:right; color:#00bcd4;">Avg(100)</th>
                        <th style="text-align:center; color:#00bcd4;">Trend</th>
                        <th style="text-align:center; color:#00bcd4;">Device</th>
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
        """
        For dashboards / JSON output. Returns the same structure as get_data(),
        already top-N filtered and with 'Other' included if applicable.
        """
        return self.get_data()

    def log_summary(self, path: Optional[str] = None) -> None:
        """
        Print a one-shot summary of current stats to the console.
        """
        console = Console()
        panel = self.get_panel_renderable()
        console.print(panel)
