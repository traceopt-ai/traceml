import shutil
from typing import Dict, Any
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from IPython.display import HTML

from traceml.database.database import Database
from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import (
    ACTIVATION_GRADIENT_LAYOUT,
)
from traceml.utils.formatting import fmt_mem_new


class LayerForwardBackwardRenderer(BaseRenderer):
    """
    Combined logger for:
      - Per-layer memory
      - Forward totals (avg + max across all layers)
      - Backward totals (avg + max across all layers)
    Keeps a cache of last-seen per-layer values so totals are consistent
    even if some layers don’t update every snapshot.
    """

    def __init__(
        self,
        layer_db: Database,
        layer_forward_db: Database,
        layer_backward_db: Database,
    ):
        super().__init__(
            name="Forward & Backward Stats",
            layout_section_name=ACTIVATION_GRADIENT_LAYOUT,
        )
        self._layer_table = layer_db.create_or_get_table("layer_memory")
        self.layer_forward_db = layer_forward_db
        self.layer_backward_db = layer_backward_db

        # Global stats
        self._forward_stats = {"count": 0, "sum": 0.0, "avg": 0.0}
        self._forward_global_max = 0.0

        self._backward_stats = {"count": 0, "sum": 0.0, "avg": 0.0}
        self._backward_global_max = 0.0

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

    def _compute_current_total(self, is_forward=True) -> float:
        """
        Compute the current total activation memory across all layers.
        """
        total = 0.0
        db = self.layer_forward_db if is_forward else self.layer_backward_db

        for layer_name, table in db.all_tables().items():
            if not table:
                continue

            last_row = table[-1]  # most recent activation snapshot
            mem_dict = last_row.get("memory", {}) or {}
            layer_total = max(float(v) for v in mem_dict.values())
            total += layer_total

        return total

    def _update_stats(self, current_total: float, is_forward=True) -> None:
        """
        Update rolling statistics for activation memory.
        """
        stats = self._forward_stats if is_forward else self._backward_stats

        stats["count"] += 1
        stats["sum"] += current_total
        stats["avg"] = stats["sum"] / stats["count"]

        # Update global peak
        if is_forward:
            self._forward_global_max = max(
                self._forward_global_max, current_total
            )
        else:
            self._backward_global_max = max(self._backward_global_max, current_total)

    def get_data(self) -> Dict[str, Any]:

        layer_info = self._get_layer_memory_info()

        current_forward = self._compute_current_total(is_forward=True)
        self._update_stats(current_forward, is_forward=True)

        current_backward = self._compute_current_total(is_forward=False)
        self._update_stats(current_backward, is_forward=False)

        return {
            "total_memory": float(layer_info.get("total_memory", 0.0) or 0.0),
            "model_index": layer_info.get("model_index", "—"),
            "forward": {
                "avg": self._forward_stats["avg"],
                "max": self._forward_global_max,
            },
            "backward": {
                "avg": self._backward_stats["avg"],
                "max": self._backward_global_max,
            },
        }

    # CLI rendering
    def get_panel_renderable(self) -> Panel:
        data = self.get_data()
        fwd = data["forward"]
        bwd = data["backward"]

        table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
        table.add_column("Metric", justify="left", style="bold")
        table.add_column("Value", justify="right", style="white")

        # Activation stats
        table.add_row("[cyan]~ Forward Footprint Avg[/cyan]", fmt_mem_new(fwd["avg"]))
        table.add_row("[cyan]~ Forward Footprint Max[/cyan]", fmt_mem_new(fwd["max"]))

        # Gradient stats
        table.add_row("[green]~ Backward Footprint Avg[/green]", fmt_mem_new(bwd["avg"]))
        table.add_row("[green]~ Backward Footprint Max[/green]", fmt_mem_new(bwd["max"]))

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
        fwd = data["forward"]
        bwd = data["backward"]

        # Activation panel
        act_html = f"""
        <div style="flex:1; border:2px solid #00bcd4; border-radius:8px; padding:10px;">
            <h4 style="color:#00bcd4; margin:0;">
                Activation Footprint (Model #{data['model_index']})
            </h4>
            <p><b>Avg:</b> {fmt_mem_new(fwd['avg'])}</p>
            <p><b>Max:</b> {fmt_mem_new(fwd['max'])}</p>
        </div>
        """

        # Gradient panel
        grad_html = f"""
        <div style="flex:1; border:2px solid #4caf50; border-radius:8px; padding:10px;">
            <h4 style="color:#4caf50; margin:0;">
                Gradient Footprint (Model #{data['model_index']})
            </h4>
            <p><b>Avg:</b> {fmt_mem_new(bwd['avg'])}</p>
            <p><b>Max:</b> {fmt_mem_new(bwd['max'])}</p>
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

    def get_dashboard_renderable(self):
        data = self.get_data()
        return data

    def log_summary(self, path) -> None:
        console = Console()

        fwd_avg = self._forward_stats["avg"]
        bwd_avg = self._backward_stats["avg"]

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="white")

        table.add_row(
            "[cyan]~ FORWARD FOOTPRINT AVG[/cyan]", "[dim]|[/dim]", fmt_mem_new(fwd_avg)
        )
        table.add_row(
            "[cyan]~ FORWARD FOOTPRINT MAX[/cyan]",
            "[dim]|[/dim]",
            fmt_mem_new(self._forward_global_max),
        )
        table.add_row(
            "[green]~ BACKWARD FOOTPRINT AVG[/green]", "[dim]|[/dim]", fmt_mem_new(bwd_avg)
        )
        table.add_row(
            "[green]~ BACKWARD FOOTPRINT MAX[/green]",
            "[dim]|[/dim]",
            fmt_mem_new(self._backward_global_max),
        )

        panel = Panel(
            table,
            title="[bold blue]Activation & Gradient Footprint - Summary[/bold blue]",
            border_style="blue",
        )
        console.print(panel)
