"""
Model Step Summary Renderer

Consumes precomputed payloads produced by ModelSummaryComputer
and renders them to CLI, dashboard, or notebook formats.

This module contains *no aggregation logic*.
"""

import shutil
from typing import Any, Dict, Optional, Tuple, List

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from IPython.display import HTML

from traceml.database.remote_database_store import RemoteDBStore
from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.managers.cli_display_manager import MODEL_COMBINED_LAYOUT
from traceml.renderers.utils import fmt_time_run, fmt_mem_new
from traceml.renderers.utils import CARD_STYLE
from .compute import StepCombinedComputer


class StepCombinedRenderer(BaseRenderer):
    """
    Renders model summary metrics (worst / median per-step).
    """

    def __init__(self, remote_store: RemoteDBStore):
        super().__init__(
            name="Model Summary",
            layout_section_name=MODEL_COMBINED_LAYOUT,
        )
        self.remote_store = remote_store
        self.computer = StepCombinedComputer(remote_store)
        self._cached_payload: Optional[Dict[str, Any]] = None

    def _payload(self) -> Dict[str, Any]:
        payload = self.computer.compute()
        if payload:
            self._cached_payload = payload
        return self._cached_payload or {}

    def get_dashboard_renderable(self) -> Dict[str, Any]:
        return self._payload()

    def get_panel_renderable(self) -> Panel:
        payload = self._payload()

        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("Stat")
        table.add_column("Dataloader Time")
        table.add_column("Step Time")
        table.add_column("Step Memory")

        def cell(key, kind):
            entry = payload.get(key)
            if not entry:
                return "—"

            ws = entry["worst"]["stats"]
            ms = entry["median"]["stats"]
            fmt = fmt_mem_new if "memory" in key else fmt_time_run

            if kind == "last":
                return f"{fmt(ws['last'])}/{fmt(ms['last'])}"
            if kind == "p95":
                return f"{fmt(ws['p95'])}/{fmt(ms['p95'])}"
            if kind == "trend":
                return ws["trend"] or "—"
            if kind == "rank":
                r = entry.get("slowest_rank")
                return f"r{r}" if r is not None else "—"

            return "—"

        rows = [
            ("Worst rank", "rank"),
            ("Trend", "trend"),
            ("p95", "p95"),
            ("Last", "last"),
        ]

        for label, kind in rows:
            table.add_row(
                label,
                cell("dataloading_time", kind),
                cell("step_time", kind),
                cell("step_gpu_memory", kind),
            )

        table.add_row("", "", "", "")
        table.add_row(
            "[bold yellow]Cluster Cost[/bold yellow]",
            "",
            "",
            "",
        )

        def sum_cell(key, stat):
            entry = payload.get(key)
            if not entry or "sum" not in entry:
                return "—"
            stats = entry["sum"]["stats"]
            fmt = fmt_mem_new if "memory" in key else fmt_time_run
            return fmt(stats.get(stat, 0.0))

        table.add_row(
            "Sum (last)",
            sum_cell("dataloading_time", "last"),
            sum_cell("step_time", "last"),
            "—",  # memory sum intentionally hidden
        )

        table.add_row(
            "Sum p95",
            sum_cell("dataloading_time", "p95"),
            sum_cell("step_time", "p95"),
            "—",
        )

        # Recent DL cost line
        dl_entry = payload.get("dataloading_time")
        st_entry = payload.get("step_time")

        dl_cost_line = Text("Recent Dataloading Cost: —", style="dim")

        if dl_entry and st_entry:
            dl_avg = dl_entry.get("sum", {}).get("stats", {}).get("avg100", 0.0)
            st_avg = st_entry.get("sum", {}).get("stats", {}).get("avg100", 0.0)

            if st_avg > 0:
                pct = dl_avg / st_avg * 100.0
                dl_cost_line = Text(
                    f"Recent Dataloading Cost (avg100): "
                    f"{fmt_time_run(dl_avg)} / {fmt_time_run(st_avg)} = {pct:.1f}%",
                    style="bold",
                )

        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 120)

        return Panel(
            Group(table, Text(""), dl_cost_line),
            title="Model Summary (tail latency + cluster cost)",
            width=width,
        )

    def get_notebook_renderable(self) -> HTML:
        """
        Notebook HTML renderable.

        Shows WORST vs MEDIAN summary for each metric,
        plus rank skew and worst rank for the latest completed step of that metric.
        """
        payload = self._payload()

        if not payload and self._cached_notebook:
            return self._cached_notebook

        def _trend_badge(trend: str) -> Tuple[str, str]:
            if not isinstance(trend, str) or not trend:
                return "—", "#666"
            if trend.startswith("+"):
                return f"↑ {trend}", "#d32f2f"
            if trend.startswith("-"):
                return f"↓ {trend}", "#2e7d32"
            if "≈" in trend:
                return trend, "#666"
            return trend, "#666"

        def metric_block(title: str, entry: Dict[str, Any], fmt) -> str:
            ws = entry["worst"]["stats"]
            ms = entry["median"]["stats"]

            trend_text, trend_color = _trend_badge(ws.get("trend", ""))

            skew_abs = float(entry.get("rank_skew_abs", 0.0))
            skew_pct = float(entry.get("rank_skew_pct", 0.0))
            slowest = entry.get("slowest_rank")

            skew_txt = f"{skew_abs:.2f} ({skew_pct * 100:.1f}%)"
            slowest_txt = str(slowest) if slowest is not None else "—"

            return f"""
            <div style="
                flex:1;
                border:1px solid #eee;
                border-radius:8px;
                padding:10px;
                font-size:13px;
            ">
                <div style="font-weight:700; margin-bottom:6px;">
                    {title}
                </div>

                <table style="width:100%; border-collapse:collapse;">
                    <thead>
                        <tr style="border-bottom:1px solid #ddd;">
                            <th align="left">Series</th>
                            <th align="right">Last</th>
                            <th align="right">p50</th>
                            <th align="right">p95</th>
                            <th align="right">Avg(100)</th>
                            <th align="center">Trend</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid #f0f0f0;">
                            <td><b>Worst</b></td>
                            <td align="right">{fmt(ws["last"])}</td>
                            <td align="right">{fmt(ws["p50"])}</td>
                            <td align="right">{fmt(ws["p95"])}</td>
                            <td align="right">{fmt(ws["avg100"])}</td>
                            <td align="center" style="color:{trend_color}; font-weight:700;">
                                {trend_text}
                            </td>
                        </tr>
                        <tr>
                            <td><b>Median</b></td>
                            <td align="right">{fmt(ms["last"])}</td>
                            <td align="right">{fmt(ms["p50"])}</td>
                            <td align="right">{fmt(ms["p95"])}</td>
                            <td align="right">{fmt(ms["avg100"])}</td>
                            <td align="center" style="color:#666; font-weight:700;">—</td>
                        </tr>
                    </tbody>
                </table>

                <div style="margin-top:8px; font-size:12px; color:#444;">
                    <span><b>Rank skew:</b> {skew_txt}</span>
                    <span style="margin-left:10px;"><b>Worst rank:</b> {slowest_txt}</span>
                </div>
            </div>
            """

        blocks: List[str] = []
        if "dataLoader_fetch" in payload:
            blocks.append(
                metric_block(
                    "Dataloader Fetch Time", payload["dataLoader_fetch"], fmt_time_run
                )
            )
        if "step_time" in payload:
            blocks.append(
                metric_block("Training Step Time", payload["step_time"], fmt_time_run)
            )
        if "step_gpu_memory" in payload:
            blocks.append(
                metric_block("GPU Step Memory", payload["step_gpu_memory"], fmt_mem_new)
            )

        html = HTML(
            f"""
        <div style="{CARD_STYLE}; width:100%;">
            <h4 style="color:#d47a00; margin:0 0 12px 0;">
                Model Summary
            </h4>

            <div style="
                display:flex;
                gap:12px;
                align-items:stretch;
            ">
                {''.join(blocks)}
            </div>
        </div>
        """
        )

        self._cached_notebook = html
        return html


    def log_summary(self, path: Optional[str] = None) -> None:
        """Print the CLI panel."""
        Console().print(self.get_panel_renderable())
