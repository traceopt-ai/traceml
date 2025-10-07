import shutil
from collections import defaultdict
from typing import Dict, Any
import numpy as np
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import STEPTIMER_SUMMARY_LAYOUT_NAME


class StepTimerRenderer(BaseRenderer):
    """
    Logger for timing statistics collected by StepTimeSampler.

    Shows top-N timers (default 4) + combines others into "Other".
    Maintains running averages and global peaks for each timer.
    """

    def __init__(self, top_n: int = 5):
        super().__init__(
            name="Step Timers",
            layout_section_name=STEPTIMER_SUMMARY_LAYOUT_NAME,
        )
        self._latest_snapshot: Dict[str, Any] = {}
        self.top_n = top_n

        # Per-event caches
        self._avg_cache: Dict[str, float] = defaultdict(float)
        self._peak_cache: Dict[str, float] = defaultdict(float)

        # Running stats
        self._avg_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "sum": 0.0, "avg": 0.0}
        )

        self._last_snapshot_cache: Dict[str, Dict[str, float]] = defaultdict(dict)

    def _update_cache(self, new_data: Dict[str, Dict[str, float]]):
        """
        Update per-timer caches with latest averages and peaks.
        new_data: {"forward": {"cpu_avg_s": .., "gpu_avg_s": .., "gpu_max_s": ..}, ...}
        """
        for name, vals in new_data.items():
            # Prefer GPU avg if available, otherwise CPU
            avg_val = vals.get("gpu_avg_s") or vals.get("cpu_avg_s") or 0.0
            peak_val = vals.get("gpu_max_s") or vals.get("cpu_max_s") or 0.0

            # Update running average
            stats = self._avg_stats[name]
            stats["count"] += 1
            stats["sum"] += avg_val
            stats["avg"] = stats["sum"] / stats["count"]
            self._avg_cache[name] = stats["avg"]
            self._peak_cache[name] = max(self._peak_cache[name], peak_val)

    def _merge_with_last_snapshot(
        self, snapshot: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Merge current snapshot with last known values, keeping missing timers visible."""
        merged = dict(self._last_snapshot_cache)
        for name, vals in snapshot.items():
            merged[name] = vals
            self._last_snapshot_cache[name] = vals
        return merged

    def _aggregate_top(
        self, snapshot: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        if not snapshot:
            return {}

        avg_source = {k: self._avg_cache.get(k, 0.0) for k in snapshot.keys()}

        # Sort by historical average (stable ordering)
        sorted_items = sorted(
            snapshot.items(), key=lambda kv: avg_source.get(kv[0], 0.0), reverse=True
        )

        top_n = self.top_n - 1
        top_items = sorted_items[:top_n]
        rest_items = sorted_items[top_n:]

        top_dict = dict(top_items)

        if rest_items:
            cpu_avg = (
                np.mean([v.get("cpu_avg_s", 0.0) for _, v in rest_items])
                if rest_items
                else 0.0
            )
            gpu_avg = (
                np.mean([v.get("gpu_avg_s", 0.0) for _, v in rest_items])
                if rest_items
                else 0.0
            )
            cpu_max = (
                np.max([v.get("cpu_max_s", 0.0) for _, v in rest_items])
                if rest_items
                else 0.0
            )
            gpu_max = (
                np.max([v.get("gpu_max_s", 0.0) for _, v in rest_items])
                if rest_items
                else 0.0
            )
            top_dict["Other"] = {
                "cpu_avg_s": cpu_avg,
                "gpu_avg_s": gpu_avg,
                "cpu_max_s": cpu_max,
                "gpu_max_s": gpu_max,
            }

        return top_dict

    def get_data(self) -> Dict[str, Any]:
        raw_snapshot = (self._latest_snapshot or {}).get("StepTimerSampler", {}).get(
            "data"
        ) or {}
        self._update_cache(raw_snapshot)
        # Merge to keep missing timers visible
        merged_snapshot = self._merge_with_last_snapshot(raw_snapshot)
        # Aggregate top based on merged (persistent) snapshot
        filtered = self._aggregate_top(merged_snapshot)
        return filtered

    def get_panel_renderable(self) -> Panel:
        data = self.get_data()
        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("Event", justify="left", style="cyan")
        table.add_column("Avg (s)", justify="right", style="white")
        table.add_column("Peak (s)", justify="right", style="magenta")

        for name, vals in data.items():
            avg_cached = self._avg_cache.get(name, 0.0)
            peak_cached = self._peak_cache.get(name, 0.0)

            table.add_row(
                f"[bold]{name}[/bold]",
                f"{avg_cached:.4f}",
                f"{peak_cached:.4f}",
            )

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(80, int(cols * 0.6)), 120)

        return Panel(
            Group(table),
            title="[bold blue]Step Timers[/bold blue]",
            border_style="blue",
            width=panel_width,
        )

    def get_notebook_renderable(self) -> HTML:
        data = self.get_data()

        rows = ""
        for name, vals in data.items():
            avg_cached = self._avg_cache.get(name, 0.0)
            peak_cached = self._peak_cache.get(name, 0.0)
            rows += f"""
                <tr>
                    <td><b>{name}</b></td>
                    <td style='text-align:right'>{avg_cached:.4f}</td>
                    <td style='text-align:right'>{peak_cached:.4f}</td>
                </tr>
            """

        html = f"""
        <div style="margin-top:10px;">
            <h4 style="color:#00bcd4;">Step Timers</h4>
            <table style="width:100%; border-collapse:collapse;">
                <thead>
                    <tr style="border-bottom:2px solid #00bcd4;">
                        <th align="left">Event</th>
                        <th align="right">Avg (s)</th>
                        <th align="right">Peak (s)</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """
        return HTML(html)

    def log_summary(self, summary: Dict[str, Any]):
        console = Console()
        table = Table(show_header=True, header_style="bold blue", box=None)
        table.add_column("Event", justify="left", style="cyan")
        table.add_column("Avg (s)", justify="right", style="white")
        table.add_column("Peak (s)", justify="right", style="magenta")

        for name, avg in self._avg_cache.items():
            peak = self._peak_cache.get(name, 0.0)
            table.add_row(f"[bold]{name}[/bold]", f"{avg:.4f}", f"{peak:.4f}")

        panel = Panel(
            table,
            title="[bold blue]Step Timer Summary[/bold blue]",
            border_style="blue",
        )
        console.print(panel)
