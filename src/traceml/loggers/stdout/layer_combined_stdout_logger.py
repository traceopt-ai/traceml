from rich.panel import Panel
from rich.table import Table
from rich.console import Group, Console
from typing import Dict, Any, Optional
import shutil

from .base_logger import BaseStdoutLogger
from .display_manager import LAYER_SUMMARY_LAYOUT_NAME
from traceml.utils.formatting import fmt_mem_new


class LayerCombinedStdoutLogger(BaseStdoutLogger):
    """
    Combined logger for layer memory and activation memory:
      - Shows both per-layer allocated memory and activation memory (current / global).
      - Activation memory persists across refreshes until updated.
      - If no activation value is known for a layer, shows "—".
    """

    def __init__(self, top_n: Optional[int] = 10):
        super().__init__(
            name="Layer Combined Memory", layout_section_name=LAYER_SUMMARY_LAYOUT_NAME
        )
        self._latest_snapshot: Dict[str, Any] = {}
        # persist activation values: {layer: {"current": float, "global": float}}
        self._activation_cache: Dict[str, Dict[str, float]] = {}
        self.top_n = top_n

    def _truncate(self, s: str, max_len: int = 42) -> str:
        if not isinstance(s, str):
            return str(s)
        return s if len(s) <= max_len else s[: max_len - 1] + "…"

    def _update_activation_cache(self, activation_snapshot: Dict[str, Any]) -> None:
        """
        Update the activation cache with new values from ActivationMemorySampler.
        Expects activation_snapshot['layers'] = {layer: {"current_peak": float, "global_peak": float}}
        """
        if not activation_snapshot:
            return

        layers_info = activation_snapshot.get("layers", {}) or {}
        for layer, val in layers_info.items():
            current = float(val.get("current_peak", 0.0))
            global_ = float(val.get("global_peak", current))
            self._activation_cache[layer] = {"current": current, "global": global_}

    def _get_activation_value(self, layer: str) -> str:
        """
        Return cached activation memory for a layer if known, else "—".
        Format: current / global
        """
        if layer in self._activation_cache:
            vals = self._activation_cache[layer]
            return f"{fmt_mem_new(vals['current'])} / {fmt_mem_new(vals['global'])}"
        return "—"

    def get_panel_renderable(self) -> Panel:
        snaps = self._latest_snapshot or {}

        # Extract layer memory sampler
        layer_sampler = (snaps.get("LayerMemorySampler") or {}).get("data") or {}
        layer_data: Dict[str, float] = layer_sampler.get("layer_memory", {}) or {}
        total_memory = float(layer_sampler.get("total_memory", 0.0) or 0.0)
        model_index = layer_sampler.get("model_index", "—")

        # Extract activation memory sampler
        activation_sampler = (snaps.get("ActivationMemorySampler") or {}).get("data") or {}
        self._update_activation_cache(activation_sampler)

        # Sort layers by memory (desc), slice top-N if required
        items = sorted(layer_data.items(), key=lambda kv: float(kv[1]), reverse=True)
        if self.top_n > 0:
            items = items[: self.top_n]

        # Build table
        table = Table(
            show_header=True,
            header_style="bold blue",
            box=None,
            pad_edge=False,
            padding=(0, 1),
        )
        table.add_column("Layer", justify="left", style="magenta")
        table.add_column("Memory", justify="right", style="white", no_wrap=True)
        table.add_column("% of total", justify="right", style="white", no_wrap=True)
        table.add_column("Activation (curr/peak)", justify="right", style="cyan", no_wrap=True)

        if items:
            for name, memory in items:
                pct = (float(memory) / total_memory * 100.0) if total_memory > 0 else 0.0
                table.add_row(
                    self._truncate(str(name)),
                    fmt_mem_new(memory),
                    f"{pct:.1f}%",
                    self._get_activation_value(name),
                )
        else:
            table.add_row("[dim]No layers detected[/dim]", "—", "—", "—")

        title_total = fmt_mem_new(total_memory)
        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            Group(table),
            title=f"[bold blue]Model #{model_index}[/bold blue]  •  Total: [white]{title_total}[/white]",
            border_style="blue",
            width=panel_width,
        )

    def log_summary(self, summary: Dict[str, Any]):
        """
        Pretty-print final cumulative summary.
        Adds top-3 global activation memory layers.
        """
        console = Console()
        summary = (summary or {}).get("LayerMemorySampler") or {}

        # Layer memory summary
        total_samples = summary.get("total_samples", "—")
        total_models = summary.get("total_models_seen", "—")
        avg_model_mem = fmt_mem_new(summary.get("average_model_memory", 0.0))
        peak_model_mem = fmt_mem_new(summary.get("peak_model_memory", 0.0))

        # Top-3 global activation peaks
        top_items = []
        if self._activation_cache:
            top_items = sorted(
                self._activation_cache.items(),
                key=lambda kv: kv[1]["global"],
                reverse=True,
            )[:3]

        # Build summary table
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="blue")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="white")

        table.add_row("TOTAL SAMPLES TAKEN", "[blue]|[/blue]", str(total_samples))
        table.add_row("TOTAL MODELS SEEN", "[blue]|[/blue]", str(total_models))
        table.add_row("AVERAGE MODEL MEMORY", "[blue]|[/blue]", avg_model_mem)
        table.add_row("PEAK MODEL MEMORY", "[blue]|[/blue]", peak_model_mem)

        # Add activations
        table.add_row("TOP-3 ACTIVATIONS", "[blue]|[/blue]", "")
        if top_items:
            for layer, vals in top_items:
                table.add_row(f"  • {layer}", "", fmt_mem_new(vals["global"]))
        else:
            table.add_row("  • None", "", "—")


        panel = Panel(
            table,
            title="[bold blue]Model Layer - Summary[/bold blue]",
            border_style="blue",
        )
        console.print(panel)
