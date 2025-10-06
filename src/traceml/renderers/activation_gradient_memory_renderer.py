import shutil
from collections import defaultdict
from typing import Dict, Any
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

    def __init__(self, top_n: int = 10):
        super().__init__(
            name="Activation & Gradient Stats",
            layout_section_name=ACTIVATION_GRADIENT_SUMMARY_LAYOUT_NAME,
        )
        self._latest_snapshot: Dict[str, Any] = {}

        # Per-layer caches (last seen current values)
        self._activation_cache: Dict[str, float] = defaultdict(float)
        self._gradient_cache: Dict[str, float] = defaultdict(float)

        # Global stats
        self._activation_stats = {"count": 0, "sum": 0.0, "avg": 0.0}
        self._activation_global_max = 0.0

        self._gradient_stats = {"count": 0, "sum": 0.0, "avg": 0.0}
        self._gradient_global_max = 0.0

        self.top_n = top_n

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

    def _process_activation_snapshot(self, snapshot: Dict[str, Any]):
        layers = snapshot.get("layers", {}) or {}
        self._update_layer_cache(self._activation_cache, layers)
        self._activation_global_max = self._update_totals(
            self._activation_cache, self._activation_stats, self._activation_global_max
        )

    def _process_gradient_snapshot(self, snapshot: Dict[str, Any]):
        layers = snapshot.get("layers", {}) or {}
        self._update_layer_cache(self._gradient_cache, layers)
        self._gradient_global_max = self._update_totals(
            self._gradient_cache, self._gradient_stats, self._gradient_global_max
        )

    def get_data(self) -> Dict[str, Any]:
        snaps = self._latest_snapshot or {}

        # Extract samplers
        layer_sampler = (snaps.get("LayerMemorySampler") or {}).get("data") or {}
        activation_sampler = (snaps.get("ActivationMemorySampler") or {}).get(
            "data"
        ) or {}
        gradient_sampler = (snaps.get("GradientMemorySampler") or {}).get("data") or {}

        # Update caches + stats
        self._process_activation_snapshot(activation_sampler)
        self._process_gradient_snapshot(gradient_sampler)

        return {
            "total_memory": float(layer_sampler.get("total_memory", 0.0) or 0.0),
            "model_index": layer_sampler.get("model_index", "—"),
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
