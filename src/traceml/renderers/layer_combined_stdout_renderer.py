from rich.panel import Panel
from rich.table import Table
from rich.console import Group, Console
from IPython.display import HTML
from typing import Dict, Any, Optional, List
import shutil

from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import (
    LAYER_COMBINED_SUMMARY_LAYOUT_NAME,
)
from traceml.utils.formatting import fmt_mem_new


class LayerCombinedRenderer(BaseRenderer):
    """
    Combined logger for per-layer:
      - Memory (allocated params + buffers)
      - Activation (curr/global)
      - Gradient  (curr/global)
    Values persist across refreshes and update only when new data arrives.
    Missing values show as "—".
    """

    def __init__(
        self,
        layer_table: List[Dict[str, Any]],
        activation_db,
        gradient_db,
        top_n_layers: Optional[int] = 20,
    ):
        super().__init__(
            name="Layer Combined Memory",
            layout_section_name=LAYER_COMBINED_SUMMARY_LAYOUT_NAME,
        )
        self._layer_table = layer_table
        self.activation_db = activation_db
        self.gradient_db = gradient_db
        self.top_n = top_n_layers

        self._activation_cache: Dict[str, Dict[str, float]] = {}
        self._gradient_cache: Dict[str, Dict[str, float]] = {}

    def _truncate(self, s: str, max_len: int = 40) -> str:
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

    def _compute_layer_snapshot(self) -> Dict[str, Any]:
        """
        Return last entry from layer_table safely.
        """
        if not self._layer_table:
            return {
                "layer_memory": {},
                "total_memory": 0.0,
                "model_index": "—",
            }
        return self._layer_table[-1]

    def _compute_snapshot(self, is_activation=True) -> Dict[str, Dict[str, float]]:
        """
        Return CURRENT and PEAK memory per layer.
        """

        layer_peaks = {}
        layer_current = {}
        if is_activation:
            db = self.activation_db
        else:
            db = self.gradient_db

        for table_name, rows in db.all_tables().items():
            layer = table_name
            if not rows:
                continue

            # Last row = CURRENT activation (max across devices)
            last_row = rows[-1]
            mem_dict = last_row.get("memory", {}) or {}
            current_peak = max(float(v) for v in mem_dict.values())
            layer_current[layer] = current_peak

            # PEAK = max across all rows
            peak = 0.0
            for r in rows:
                m = r.get("memory", {}) or {}
                peak = max(peak, max(float(v) for v in m.values()))

            layer_peaks[layer] = peak

        merged = {
            layer: {
                "current_peak": layer_current.get(layer, 0.0),
                "global_peak": layer_peaks.get(layer, 0.0),
            }
            for layer in set(layer_peaks) | set(layer_current)
        }

        return merged

    def get_data(self) -> Dict[str, Any]:
        layer_snapshot = self._compute_layer_snapshot()
        activation_snapshot = self._compute_snapshot(is_activation=True)
        gradient_snapshot = self._compute_snapshot(is_activation=False)

        # update caches
        self._merge_cache(self._activation_cache, activation_snapshot)
        self._merge_cache(self._gradient_cache, gradient_snapshot)

        return {
            "layers": layer_snapshot.get("layer_memory", {}) or {},
            "total_memory": layer_snapshot.get("total_memory", 0.0) or 0.0,
            "model_index": layer_snapshot.get("model_index", "—"),
            "activation_cache": self._activation_cache,
            "gradient_cache": self._gradient_cache,
        }

    def _compute_display_data(self) -> Dict[str, Any]:
        """
        Compute the aggregated data used by both Rich (CLI) and HTML (Notebook) renderers.
        """
        data = self.get_data()
        layers = data["layers"]
        total_memory = data["total_memory"]
        model_index = data["model_index"]

        # sort + top_n split
        sorted_items = sorted(layers.items(), key=lambda kv: float(kv[1]), reverse=True)
        top_items = sorted_items[: self.top_n] if self.top_n else sorted_items
        other_items = sorted_items[self.top_n :] if self.top_n else []

        other_total = sum(float(v) for _, v in other_items)
        pct_other = (other_total / total_memory * 100.0) if total_memory > 0 else 0.0

        act_cache = data["activation_cache"]
        grad_cache = data["gradient_cache"]

        def _agg_peaks(cache: Dict[str, Dict[str, float]], names: list[str]):
            if not names:
                return {"current": 0.0, "global": 0.0}
            cur = sum(cache.get(n, {}).get("current", 0.0) for n in names)
            gbl = sum(cache.get(n, {}).get("global", 0.0) for n in names)
            return {"current": cur, "global": gbl}

        other_act = _agg_peaks(act_cache, [n for n, _ in other_items])
        other_grad = _agg_peaks(grad_cache, [n for n, _ in other_items])

        return {
            "top_items": top_items,
            "other": {
                "total": other_total,
                "pct": pct_other,
                "activation": other_act,
                "gradient": other_grad,
            },
            "total_memory": total_memory,
            "model_index": model_index,
            "activation_cache": act_cache,
            "gradient_cache": grad_cache,
        }

    ## CLI rendering in Terminal
    def get_panel_renderable(self) -> Panel:
        d = self._compute_display_data()

        table = Table(
            show_header=True,
            header_style="bold blue",
            box=None,
            pad_edge=False,
            padding=(0, 1),
        )
        table.add_column("Layer", justify="left", style="magenta")
        table.add_column("Memory", justify="right", style="white")
        table.add_column("% of total", justify="right", style="white")
        table.add_column("Activation (curr/peak)", justify="right", style="cyan")
        table.add_column("Gradient (curr/peak)", justify="right", style="green")

        if d["top_items"]:
            for name, memory in d["top_items"]:
                pct = (
                    (float(memory) / d["total_memory"] * 100.0)
                    if d["total_memory"] > 0
                    else 0.0
                )
                table.add_row(
                    self._truncate(name),
                    fmt_mem_new(memory),
                    f"{pct:.1f}%",
                    self._format_cache_value(d["activation_cache"], name),
                    self._format_cache_value(d["gradient_cache"], name),
                )

        # “Other” row
        if d["other"]["total"] > 0:
            o = d["other"]
            table.add_row(
                "Other Layers",
                f"{fmt_mem_new(o['total'])}",
                f"{o['pct']:.1f}%",
                f"{fmt_mem_new(o['activation']['current'])} / {fmt_mem_new(o['activation']['global'])}",
                f"{fmt_mem_new(o['gradient']['current'])} / {fmt_mem_new(o['gradient']['global'])}",
            )

        if not d["top_items"]:
            table.add_row("[dim]No layers detected[/dim]", "—", "—", "—", "—")

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            Group(table),
            title=f"[bold blue]Model #{d['model_index']}[/bold blue] • Total: [white]{fmt_mem_new(d['total_memory'])}[/white]",
            border_style="blue",
            width=panel_width,
        )

    # Notebook rendering
    def get_notebook_renderable(self) -> HTML:
        d = self._compute_display_data()

        rows = ""
        if d["top_items"]:
            for name, memory in d["top_items"]:
                pct = (
                    (float(memory) / d["total_memory"] * 100.0)
                    if d["total_memory"] > 0
                    else 0.0
                )
                rows += f"""
                   <tr>
                       <td style="text-align:left;">{self._truncate(name)}</td>
                       <td style="text-align:right;">{fmt_mem_new(memory)}</td>
                       <td style="text-align:right;">{pct:.1f}%</td>
                       <td style="text-align:right;">{self._format_cache_value(d['activation_cache'], name)}</td>
                       <td style="text-align:right;">{self._format_cache_value(d['gradient_cache'], name)}</td>
                   </tr>
                   """

        if d["other"]["total"] > 0:
            o = d["other"]
            rows += f"""
               <tr style="color:gray;">
                   <td>Other Layers</td>
                   <td style="text-align:right;">{fmt_mem_new(o['total'])}</td>
                   <td style="text-align:right;">{o['pct']:.1f}%</td>
                   <td style="text-align:right;">{fmt_mem_new(o['activation']['current'])} / {fmt_mem_new(o['activation']['global'])}</td>
                   <td style="text-align:right;">{fmt_mem_new(o['gradient']['current'])} / {fmt_mem_new(o['gradient']['global'])}</td>
               </tr>
               """

        if not rows.strip():
            rows = """<tr><td colspan="5" style="text-align:center; color:gray;">No layers detected</td></tr>"""

        html = f"""
           <div style="border:2px solid #2196f3; border-radius:8px; padding:10px; margin-top:10px;">
               <h4 style="color:#2196f3; margin:0;">
                   Model #{d['model_index']} • Total: {fmt_mem_new(d['total_memory'])}
               </h4>
               <table style="width:100%; border-collapse:collapse; margin-top:8px;">
                   <thead style="background:#f0f8ff;">
                       <tr>
                           <th style="text-align:left;">Layer</th>
                           <th style="text-align:right;">Memory</th>
                           <th style="text-align:right;">% of total</th>
                           <th style="text-align:right;">Activation (curr/peak)</th>
                           <th style="text-align:right;">Gradient (curr/peak)</th>
                       </tr>
                   </thead>
                   <tbody>
                       {rows}
                   </tbody>
               </table>
           </div>
           """
        return HTML(html)

    def _top_n_from_dict(self, d: Dict[str, float], n: int = 3):
        """Top-n items from a {key: value} dict, sorted by value desc."""
        if not d:
            return []
        return sorted(d.items(), key=lambda kv: float(kv[1]), reverse=True)[:n]

    def _compute_layer_memory_summary(self) -> Dict[str, Any]:
        """
        Compute statistics directly from the layer memory table.
        """
        if not self._layer_table:
            return {
                "total_samples": 0,
                "total_models_seen": 0,
                "average_model_memory": 0.0,
                "peak_model_memory": 0.0,
            }

        total_samples = len(self._layer_table)
        model_signatures = {entry.get("model_signature") for entry in self._layer_table}

        totals = [float(entry.get("total_memory", 0.0)) for entry in self._layer_table]
        avg_memory = sum(totals) / len(totals) if totals else 0.0
        peak_memory = max(totals) if totals else 0.0

        return {
            "total_samples": total_samples,
            "total_models_seen": len(model_signatures),
            "average_model_memory": avg_memory,
            "peak_model_memory": peak_memory,
        }

    def _compute_peaks(self, is_activation=True) -> Dict[str, float]:
        """Compute globalpeak per layer from activation/gradient_db."""
        if is_activation:
            db = self.activation_db
        else:
            db = self.gradient_db

        act_global = {}
        for layer_name, rows in db.all_tables().items():
            peak = 0.0
            for r in rows:
                mem = r.get("memory", {}) or {}
                peak = max(peak, max(float(v) for v in mem.values()))
            act_global[layer_name] = peak
        return act_global

    def _render_section_layer_stats(self, table, stats):
        table.add_row(
            "[blue]TOTAL SAMPLES TAKEN[/blue]",
            "[dim]|[/dim]",
            str(stats["total_samples"]),
        )
        table.add_row(
            "[blue]TOTAL MODELS SEEN[/blue]",
            "[dim]|[/dim]",
            str(stats["total_models_seen"]),
        )
        table.add_row(
            "[blue]AVERAGE MODEL MEMORY[/blue]",
            "[dim]|[/dim]",
            fmt_mem_new(stats["average_model_memory"]),
        )
        table.add_row(
            "[blue]PEAK MODEL MEMORY[/blue]",
            "[dim]|[/dim]",
            fmt_mem_new(stats["peak_model_memory"]),
        )

    def _render_section_topk(self, table, title: str, items: List, color: str):
        table.add_row(f"[{color}]{title}[/{color}]", "[dim]|[/dim]", "")

        if items:
            for layer, value in items:
                table.add_row(
                    f"  [{color}]• {layer}[/{color}]",
                    "",
                    f"[{color}]{fmt_mem_new(value)}[/{color}]",
                )
        else:
            table.add_row("  • None", "", "—")

    def log_summary(self, summary: Dict[str, Any]):
        console = Console()

        # Compute sections
        layer_stats = self._compute_layer_memory_summary()
        act_peaks = self._compute_peaks()
        grad_peaks = self._compute_peaks()

        # Top-k
        top_acts = self._top_n_from_dict(act_peaks, n=3)
        top_grads = self._top_n_from_dict(grad_peaks, n=3)

        # Render
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="white")

        # Sections
        self._render_section_layer_stats(table, layer_stats)
        self._render_section_topk(table, "TOP-3 ACTIVATIONS", top_acts, "cyan")
        self._render_section_topk(table, "TOP-3 GRADIENTS", top_grads, "green")

        # Output
        panel = Panel(
            table,
            title="[bold blue]Model Layer - Summary[/bold blue]",
            border_style="blue",
        )
        console.print(panel)
