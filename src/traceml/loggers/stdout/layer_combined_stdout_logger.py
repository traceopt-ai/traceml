from rich.panel import Panel
from rich.table import Table
from rich.console import Group, Console
from typing import Dict, Any, Optional
import shutil

from .base_logger import BaseStdoutLogger
from .display_manager import LAYER_COMBINED_SUMMARY_LAYOUT_NAME
from traceml.utils.formatting import fmt_mem_new


class LayerCombinedStdoutLogger(BaseStdoutLogger):
    """
    Combined logger for per-layer:
      - Memory (allocated params + buffers)
      - Activation (curr/global)
      - Gradient  (curr/global)
    Values persist across refreshes and update only when new data arrives.
    Missing values show as "—".
    """

    def __init__(self, top_n: Optional[int] = 20):
        super().__init__(
            name="Layer Combined Memory",
            layout_section_name=LAYER_COMBINED_SUMMARY_LAYOUT_NAME,
        )
        self._latest_snapshot: Dict[str, Any] = {}
        self._activation_cache: Dict[str, Dict[str, float]] = (
            {}
        )  # {layer: {current, global}}
        self._gradient_cache: Dict[str, Dict[str, float]] = (
            {}
        )  # {layer: {current, global}}
        self.top_n = top_n

    def _truncate(self, s: str, max_len: int = 30) -> str:
        """Truncate long layer names keeping start and end."""
        if not isinstance(s, str):
            s = str(s)
        if len(s) <= max_len:
            return s
        half = (max_len - 1) // 2
        return s[:half] + "…" + s[-half:]


    def _merge_cache(
        self, cache: Dict[str, Dict[str, float]], new_data: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Merge latest snapshot data into a persistent cache.
        Expected new_data: {layer: {"current_peak": float, "global_peak": float}}
        Only provided fields update; missing fields keep prior values.
        """
        if not new_data:
            return
        for layer, entry in new_data.items():
            if not isinstance(entry, dict):
                continue
            cur = entry.get("current_peak")
            gbl = entry.get("global_peak")
            rec = cache.get(layer)
            if rec is None:
                if cur is not None or gbl is not None:
                    cache[layer] = {
                        "current": float(cur if cur is not None else 0.0),
                        "global": float(
                            gbl
                            if gbl is not None
                            else (cur if cur is not None else 0.0)
                        ),
                    }
                continue
            if cur is not None:
                rec["current"] = float(cur)
            if gbl is not None:
                rec["global"] = max(float(gbl), float(rec.get("global", 0.0)))

    def _format_cache_value(
        self, cache: Dict[str, Dict[str, float]], layer: str
    ) -> str:
        """Return 'curr / global' string for a given layer from a cache; '—' if unknown."""
        rec = cache.get(layer)
        if not rec:
            return "—"
        return f"{fmt_mem_new(rec.get('current', 0.0))} / {fmt_mem_new(rec.get('global', 0.0))}"

    def _top_n_from_dict(self, d: Dict[str, float], n: int = 3):
        """Top-n items from a {key: value} dict, sorted by value desc."""
        if not d:
            return []
        return sorted(d.items(), key=lambda kv: float(kv[1]), reverse=True)[:n]

    def _build_layer_table(
        self,
        layer_data: Dict[str, float],
        total_memory: float,
        activation_layers: Dict[str, Dict[str, float]],
        gradient_layers: Dict[str, Dict[str, float]],
    ) -> Table:
        """Build the live table and update caches from latest snapshots."""
        # Update caches first
        self._merge_cache(self._activation_cache, activation_layers)
        self._merge_cache(self._gradient_cache, gradient_layers)

        # Sort by allocated memory
        items = sorted(layer_data.items(), key=lambda kv: float(kv[1]), reverse=True)
        if self.top_n and self.top_n > 0:
            items = items[: self.top_n]

        table = Table(
            show_header=True,
            header_style="bold blue",
            box=None,
            pad_edge=False,
            padding=(0, 1),
        )
        table.add_column("Layer", justify="left", style="magenta")
        table.add_column("Memory", justify="right", style="white", no_wrap=False, overflow="fold")
        table.add_column("% of total", justify="right", style="white", no_wrap=True)
        table.add_column(
            "Activation (curr/peak)", justify="right", style="cyan", no_wrap=True
        )
        table.add_column(
            "Gradient (curr/peak)", justify="right", style="green", no_wrap=True
        )

        if items:
            for name, memory in items:
                lname = str(name)
                pct = (
                    (float(memory) / total_memory * 100.0) if total_memory > 0 else 0.0
                )
                table.add_row(
                    self._truncate(lname),
                    fmt_mem_new(memory),
                    f"{pct:.1f}%",
                    self._format_cache_value(self._activation_cache, lname),
                    self._format_cache_value(self._gradient_cache, lname),
                )
        else:
            table.add_row("[dim]No layers detected[/dim]", "—", "—", "—", "—")

        return table

    def get_panel_renderable(self) -> Panel:
        snaps = self._latest_snapshot or {}

        # LayerMemorySampler data (allocated per layer)
        layer_sampler = (snaps.get("LayerMemorySampler") or {}).get("data") or {}
        layer_data: Dict[str, float] = layer_sampler.get("layer_memory", {}) or {}
        total_memory = float(layer_sampler.get("total_memory", 0.0) or 0.0)
        model_index = layer_sampler.get("model_index", "—")

        # Activation + Gradient snapshots (curr/global per layer)
        activation_sampler = (snaps.get("ActivationMemorySampler") or {}).get(
            "data"
        ) or {}
        activation_layers = activation_sampler.get("layers", {}) or {}

        gradient_sampler = (snaps.get("GradientMemorySampler") or {}).get("data") or {}
        gradient_layers = gradient_sampler.get("layers", {}) or {}

        table = self._build_layer_table(
            layer_data, total_memory, activation_layers, gradient_layers
        )

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
        Shows top-3 activation and gradient global peaks; falls back to caches if needed.
        """
        console = Console()
        summary = summary or {}

        # Layer memory stats
        layer_mem = summary.get("LayerMemorySampler") or {}
        total_samples = layer_mem.get("total_samples", "—")
        total_models = layer_mem.get("total_models_seen", "—")
        avg_model_mem = fmt_mem_new(layer_mem.get("average_model_memory", 0.0))
        peak_model_mem = fmt_mem_new(layer_mem.get("peak_model_memory", 0.0))

        # Activation peaks
        activation_sum = (summary.get("ActivationMemorySampler") or {}).get(
            "data"
        ) or {}
        act_global = activation_sum.get("layer_global_peaks") or {}
        if not act_global and self._activation_cache:
            act_global = {
                k: v.get("global", 0.0) for k, v in self._activation_cache.items()
            }
        top_acts = self._top_n_from_dict(act_global, n=3)

        # Gradient peaks
        gradient_sum = (summary.get("GradientMemorySampler") or {}).get("data") or {}
        grad_global = gradient_sum.get("layer_global_peaks") or {}
        if not grad_global and self._gradient_cache:
            grad_global = {
                k: v.get("global", 0.0) for k, v in self._gradient_cache.items()
            }
        top_grads = self._top_n_from_dict(grad_global, n=3)

        # Build summary table
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="white")

        # Section: Stats
        table.add_row(
            "[blue]TOTAL SAMPLES TAKEN[/blue]", "[dim]|[/dim]", str(total_samples)
        )
        table.add_row(
            "[blue]TOTAL MODELS SEEN[/blue]", "[dim]|[/dim]", str(total_models)
        )
        table.add_row(
            "[blue]AVERAGE MODEL MEMORY[/blue]", "[dim]|[/dim]", avg_model_mem
        )
        table.add_row("[blue]PEAK MODEL MEMORY[/blue]", "[dim]|[/dim]", peak_model_mem)

        # Section: Activations
        table.add_row("[cyan]TOP-3 ACTIVATIONS[/cyan]", "[dim]|[/dim]", "")
        if top_acts:
            for layer, g_peak in top_acts:
                table.add_row(
                    f"  [cyan]• {layer}[/cyan]",
                    "",
                    f"[cyan]{fmt_mem_new(g_peak)}[/cyan]",
                )
        else:
            table.add_row("  • None", "", "—")

        # Section: Gradients
        table.add_row("[green]TOP-3 GRADIENTS[/green]", "[dim]|[/dim]", "")
        if top_grads:
            for layer, g_peak in top_grads:
                table.add_row(
                    f"  [green]• {layer}[/green]",
                    "",
                    f"[green]{fmt_mem_new(g_peak)}[/green]",
                )
        else:
            table.add_row("  • None", "", "—")

        panel = Panel(
            table,
            title="[bold blue]Model Layer - Summary[/bold blue]",
            border_style="blue",
        )
        console.print(panel)
