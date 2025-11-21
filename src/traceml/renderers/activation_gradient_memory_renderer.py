import shutil
from collections import defaultdict
from typing import Dict, Any, List
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import (
    ACTIVATION_GRADIENT_SUMMARY_LAYOUT_NAME,
)
from traceml.utils.formatting import fmt_mem_new


class ActivationGradientRenderer(BaseRenderer):
    """
    Combined logger for:
      - Per-layer memory
      - Activation totals (avg + max across all layers)
      - Gradient totals (avg + max across all layers)
    Keeps a cache of last-seen per-layer values so totals are consistent
    even if some layers don’t update every snapshot.
    """

    def __init__(self, layer_table: List[Dict[str, Any]], activation_db):
        super().__init__(
            name="Activation & Gradient Stats",
            layout_section_name=ACTIVATION_GRADIENT_SUMMARY_LAYOUT_NAME,
        )
        self._layer_table = layer_table
        self.activation_db = activation_db

        self._latest_snapshot: Dict[str, Any] = {}

        # Per-layer caches (last seen current values)
        self._gradient_cache: Dict[str, float] = defaultdict(float)

        # Global stats
        self._activation_stats = {"count": 0, "sum": 0.0, "avg": 0.0}
        self._activation_global_max = 0.0

        self._gradient_stats = {"count": 0, "sum": 0.0, "avg": 0.0}
        self._gradient_global_max = 0.0

    def _update_layer_cache(
        self, cache: Dict[str, float], new_data: Dict[str, Dict[str, float]]
    ):
        """Update per-layer cache with latest current_peak values."""
        for layer, entry in (new_data or {}).items():
            cur = float(entry.get("current_peak", 0.0))
            cache[layer] = cur

    def _update_totals(
        self, cache: Dict[str, float], stats: Dict[str, float], global_max: float
    ) -> float:
        """Compute total across all cached layers and update stats."""
        total_now = sum(cache.values())
        stats["count"] += 1
        stats["sum"] += total_now
        stats["avg"] = stats["sum"] / stats["count"]
        return max(global_max, total_now)

    def _process_gradient_snapshot(self, snapshot: Dict[str, Any]):
        layers = snapshot.get("layers", {}) or {}
        self._update_layer_cache(self._gradient_cache, layers)
        self._gradient_global_max = self._update_totals(
            self._gradient_cache, self._gradient_stats, self._gradient_global_max
        )

    def _get_layer_memory_info(self) -> Dict[str, Any]:
        """
        Extract the latest layer memory info from the layer table.
        Returns: {"total_memory": float, "model_index": str}
        """
        if not self._layer_table:
            return {"total_memory": 0.0, "model_index": "—"}

        latest = self._layer_table[-1]
        return {
            "total_memory": float(latest.get("total_memory", 0.0) or 0.0),
            "model_index": latest.get("model_index", "—"),
        }

    def _compute_current_activation_total(self) -> float:
        """
        Compute the current total activation memory across all layers.
        """
        total = 0.0

        for layer_name, table in self.activation_db.all_tables().items():
            if not table:
                continue

            last_row = table[-1]  # most recent activation snapshot
            mem_dict = last_row.get("memory", {}) or {}
            layer_total = max(float(v) for v in mem_dict.values())
            total += layer_total

        return total

    def _update_activation_stats(self, current_total: float) -> None:
        """
        Update rolling statistics for activation memory.
        """
        stats = self._activation_stats

        stats["count"] += 1
        stats["sum"] += current_total
        stats["avg"] = stats["sum"] / stats["count"]

        # Update global peak
        self._activation_global_max = max(self._activation_global_max, current_total)

    def get_data(self) -> Dict[str, Any]:
        snaps = self._latest_snapshot or {}

        # Extract samplers
        layer_info = self._get_layer_memory_info()
        current_activation = self._compute_current_activation_total()
        self._update_activation_stats(current_activation)

        gradient_sampler = (snaps.get("GradientMemorySampler") or {}).get("data") or {}

        # Update caches + stats
        self._process_gradient_snapshot(gradient_sampler)

        return {
            "total_memory": float(layer_info.get("total_memory", 0.0) or 0.0),
            "model_index": layer_info.get("model_index", "—"),
            "activation": {
                "avg": self._activation_stats["avg"],
                "max": self._activation_global_max,
            },
            "gradient": {
                "avg": self._gradient_stats["avg"],
                "max": self._gradient_global_max,
            },
        }

    # CLI rendering
    def get_panel_renderable(self) -> Panel:
        data = self.get_data()
        act = data["activation"]
        grad = data["gradient"]

        table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
        table.add_column("Metric", justify="left", style="bold")
        table.add_column("Value", justify="right", style="white")

        # Activation stats
        table.add_row("[cyan]Approx Activation Avg[/cyan]", fmt_mem_new(act["avg"]))
        table.add_row("[cyan]Approx Activation Max[/cyan]", fmt_mem_new(act["max"]))

        # Gradient stats
        table.add_row("[green]Approx Gradient Avg[/green]", fmt_mem_new(grad["avg"]))
        table.add_row("[green]Approx Gradient Max[/green]", fmt_mem_new(grad["max"]))

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(80, int(cols * 0.6)), 120)

        return Panel(
            Group(table),
            title=f"[bold blue]Model #{data['model_index']}[/bold blue]  •  "
            f"Total Mem: [white]{fmt_mem_new(data['total_memory'])}[/white]",
            border_style="blue",
            width=panel_width,
        )

    # Notebook rendering
    def get_notebook_renderable(self) -> HTML:
        data = self.get_data()
        act = data["activation"]
        grad = data["gradient"]

        # Activation panel
        act_html = f"""
        <div style="flex:1; border:2px solid #00bcd4; border-radius:8px; padding:10px;">
            <h4 style="color:#00bcd4; margin:0;">
                Activation (Model #{data['model_index']})
            </h4>
            <p><b>Avg:</b> {fmt_mem_new(act['avg'])}</p>
            <p><b>Max:</b> {fmt_mem_new(act['max'])}</p>
        </div>
        """

        # Gradient panel
        grad_html = f"""
        <div style="flex:1; border:2px solid #4caf50; border-radius:8px; padding:10px;">
            <h4 style="color:#4caf50; margin:0;">
                Gradient (Model #{data['model_index']})
            </h4>
            <p><b>Avg:</b> {fmt_mem_new(grad['avg'])}</p>
            <p><b>Max:</b> {fmt_mem_new(grad['max'])}</p>
        </div>
        """

        # Combine in a row
        combined = f"""
        <div style="display:flex; gap:20px; margin-top:10px;">
            {act_html}
            {grad_html}
        </div>
        """
        return HTML(combined)

    def log_summary(self, summary: Dict[str, Any]):
        console = Console()

        act_avg = self._activation_stats["avg"]
        grad_avg = self._gradient_stats["avg"]

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="white")

        table.add_row(
            "[cyan]Approx ACTIVATION AVG[/cyan]", "[dim]|[/dim]", fmt_mem_new(act_avg)
        )
        table.add_row(
            "[cyan]Approx ACTIVATION MAX[/cyan]",
            "[dim]|[/dim]",
            fmt_mem_new(self._activation_global_max),
        )
        table.add_row(
            "[green]Approx GRADIENT AVG[/green]", "[dim]|[/dim]", fmt_mem_new(grad_avg)
        )
        table.add_row(
            "[green]Approx GRADIENT MAX[/green]",
            "[dim]|[/dim]",
            fmt_mem_new(self._gradient_global_max),
        )

        panel = Panel(
            table,
            title="[bold blue]Activation & Gradient - Summary[/bold blue]",
            border_style="blue",
        )
        console.print(panel)
