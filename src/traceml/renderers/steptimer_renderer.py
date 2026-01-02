import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from IPython.display import HTML

from traceml.database.database import Database
from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import STEPTIMER_LAYOUT
from traceml.renderers.utils import fmt_time_run
from .utils import CARD_STYLE


@dataclass
class StepTimerRow:
    name: str
    last: float
    p50_100: float
    p95_100: float
    avg_100: float
    trend: str
    device: str


class StepTimerRenderer(BaseRenderer):
    """
    Renderer for user-defined step timers.

    Pipeline:
        DB → raw series → top-N aggregation → stats rows → render
    """

    def __init__(self, database: Database, top_n: int = 5):
        super().__init__(name="Step Timers", layout_section_name=STEPTIMER_LAYOUT)
        self.db = database
        self.top_n = top_n

    def _is_internal(self, name: str) -> bool:
        return name.startswith("_traceml_internal:")

    def _collect_cpu_series(self) -> Dict[str, List[float]]:
        table = self.db.create_or_get_table("step_timer_cpu")
        out: Dict[str, List[float]] = defaultdict(list)

        for row in table:
            name = row.get("event_name")
            if name:
                out[name].append(float(row.get("duration_ms", 0.0)))

        return out

    def _collect_gpu_series(self) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = defaultdict(list)

        for table_name, rows in self.db.all_tables().items():
            if not table_name.startswith("step_timer_cuda"):
                continue
            for row in rows:
                name = row.get("event_name")
                if name:
                    out[name].append(float(row.get("duration_ms", 0.0)))

        return out

    def _collect_series(self) -> Dict[str, Dict[str, List[float]]]:
        cpu = self._collect_cpu_series()
        gpu = self._collect_gpu_series()

        return {
            name: {"cpu": cpu.get(name, []), "gpu": gpu.get(name, [])}
            for name in set(cpu) | set(gpu)
            if not self._is_internal(name)
        }

    def _mean_for_display(self, vals: Dict[str, List[float]]) -> float:
        arr = vals["gpu"] if vals["gpu"] else vals["cpu"]
        return float(np.mean(arr)) if arr else 0.0

    def _aggregate_top_n(
        self, series: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, List[float]]]:
        if not series or self.top_n <= 0:
            return series

        items = sorted(
            series.items(),
            key=lambda kv: self._mean_for_display(kv[1]),
            reverse=True,
        )

        if len(items) <= self.top_n:
            return dict(items)

        keep = items[: self.top_n - 1]
        rest = items[self.top_n - 1 :]

        def sum_series(key: str) -> List[float]:
            max_len = max(len(v[key]) for _, v in rest)
            return [
                sum(v[key][i] for _, v in rest if i < len(v[key]))
                for i in range(max_len)
            ]

        out = dict(keep)
        out["Other"] = {
            "cpu": sum_series("cpu"),
            "gpu": sum_series("gpu"),
        }
        return out

    def _safe_percentile(self, arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q)) if arr.size else 0.0

    def _compute_row(
        self, name: str, cpu_vals: List[float], gpu_vals: List[float]
    ) -> StepTimerRow:
        arr = np.asarray(gpu_vals or cpu_vals, dtype=np.float64)
        device = "GPU" if gpu_vals else "CPU"

        if arr.size == 0:
            return StepTimerRow(name, 0, 0, 0, 0, "", device)

        last = float(arr[-1])
        win100 = arr[-min(100, arr.size) :]
        avg100 = float(win100.mean())

        trend = ""
        n = arr.size
        if n >= 50:
            early_p95 = np.percentile(arr[:50], 95)
            later_avg = arr[50:].mean() if n > 50 else arr.mean()

            if early_p95 > 2.0 * later_avg:
                trend = "!"

        if trend == "" and n >= 200:
            recent_avg = arr[-100:].mean()
            prev_avg = arr[-200:-100].mean()
            trend = "+" if recent_avg > prev_avg else "-"

        return StepTimerRow(
            name=name,
            last=last,
            p50_100=self._safe_percentile(win100, 50),
            p95_100=self._safe_percentile(win100, 95),
            avg_100=avg100,
            trend=trend,
            device=device,
        )

    # ------------------------------------------------------------------
    # Public data pipeline
    # ------------------------------------------------------------------

    def _build_rows(self) -> List[StepTimerRow]:
        series = self._aggregate_top_n(self._collect_series())
        rows = [
            self._compute_row(name, vals["cpu"], vals["gpu"])
            for name, vals in series.items()
        ]

        # Stable display order
        rows.sort(key=lambda r: r.avg_100, reverse=True)
        rows.sort(key=lambda r: r.name == "Other")
        return rows

    # ------------------------------------------------------------------
    # Renderers
    # ------------------------------------------------------------------

    def get_panel_renderable(self) -> Panel:
        rows = self._build_rows()

        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("Event", justify="left", style="cyan")
        table.add_column("Last", justify="right")
        table.add_column("p50(100)", justify="right")
        table.add_column("p95(100)", justify="right")
        table.add_column("Avg(100)", justify="right")
        table.add_column("Trend", justify="center")
        table.add_column("Device", justify="center", style="magenta")

        if not rows:
            table.add_row(
                "[dim]No step timers recorded[/dim]", "—", "—", "—", "—", "", "—"
            )
        else:
            for r in rows:
                table.add_row(
                    f"[bold]{r.name}[/bold]",
                    fmt_time_run(r.last),
                    fmt_time_run(r.p50_100),
                    fmt_time_run(r.p95_100),
                    fmt_time_run(r.avg_100),
                    r.trend,
                    r.device,
                )

        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 110)

        return Panel(
            Group(table),
            title="[bold blue]Trace Timers[/bold blue]",
            border_style="blue",
            width=width,
        )

    def get_notebook_renderable(self) -> HTML:
        rows = self._build_rows()

        if not rows:
            body = """
            <tr>
                <td colspan="7" style="text-align:center;color:gray;">
                    No step timers recorded
                </td>
            </tr>
            """
        else:
            body = ""
            for r in rows
                if r.trend == "+":
                    trend_symbol = "↑"
                    trend_color = "#d32f2f"  # red
                elif r.trend == "-":
                    trend_symbol = "↓"
                    trend_color = "#2e7d32"  # green
                body += f"""
                    <tr>
                        <td>{r.name}</td>
                        <td>{fmt_time_run(r.last)}</td>
                        <td>{fmt_time_run(r.p50_100)}</td>
                        <td>{fmt_time_run(r.p95_100)}</td>
                        <td>{fmt_time_run(r.avg_100)}</td>
                        <td> color:{trend_color};">{trend_symbol}</td>
                        <td>{r.device}</td>
                    </tr>
                    """

        table_html = f"""
        <table style="
            width:100%;
            border-collapse:collapse;
            font-size:13px;
        ">
            <thead>
                <tr style="border-bottom:1px solid #e0e0e0;">
                    <th align="left">Step</th>
                    <th align="right">Last</th>
                    <th align="right">p50(100)</th>
                    <th align="right">p95(100)</th>
                    <th align="right">Avg(100)</th>
                    <th align="center">Trend</th>
                    <th align="center">Device</th>
                </tr>
            </thead>
            <tbody>
                {body}
            </tbody>
        </table>
        """

        return HTML(f"""
        <div style="{CARD_STYLE}">
            <h4 style="
                color:#d47a00;
                margin:0 0 10px 0;
            ">
                Step Timings
            </h4>

            {table_html}
        </div>
        """)

    def get_dashboard_renderable(self):
        return [r.__dict__ for r in self._build_rows()]

    def log_summary(self, path: Optional[str] = None) -> None:
        Console().print(self.get_panel_renderable())
