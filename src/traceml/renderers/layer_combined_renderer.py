from typing import Dict, Any, List, Optional
import shutil

from rich.panel import Panel
from rich.table import Table
from rich.console import Group, Console
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.database.database import Database
from traceml.renderers.display.cli_display_manager import (
    LAYER_COMBINED_SUMMARY_LAYOUT_NAME,
)
from traceml.utils.formatting import fmt_mem_new

from traceml.renderers.layer_combined.services import (
    LayerCombinedData,
    LayerMemorySummary,
)
from traceml.renderers.layer_combined.utils import (
    truncate_layer_name,
    format_cache_value,
)


class LayerCombinedRenderer(BaseRenderer):
    """
    Combined logger for per-layer:
      - Memory (allocated params + buffers)
      - Activation (curr/global)
      - Gradient  (curr/global)

    Responsibilities:
      - orchestrate data services
      - build Rich / HTML renderables
      - print CLI summary

    All heavy computation and caching lives in `LayerCombinedDataService`
    and `LayerMemorySummaryService`.
    """

    def __init__(
        self,
        layer_db: Database,
        activation_db: Database,
        gradient_db: Database,
        top_n_layers: Optional[int] = 20,
    ):
        super().__init__(
            name="Layer Combined Memory",
            layout_section_name=LAYER_COMBINED_SUMMARY_LAYOUT_NAME,
        )

        layer_table = layer_db.create_or_get_table("layer_memory")
        self._data_service = LayerCombinedData(
            layer_table=layer_table,
            activation_db=activation_db,
            gradient_db=gradient_db,
            top_n_layers=top_n_layers,
        )
        self._summary_service = LayerMemorySummary(
            layer_table=layer_table,
            activation_db=activation_db,
            gradient_db=gradient_db,
        )

    # -------------------------------------------------------------------------
    # CLI + Notebook renderables
    # -------------------------------------------------------------------------

    def get_panel_renderable(self) -> Panel:
        """
        Rich (terminal) representation for top-N layers + "Other Layers".
        """
        d = self._data_service.compute_display_data()

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

        # Top-N rows
        if d["top_items"]:
            for name, memory in d["top_items"]:
                pct = (
                    (float(memory) / d["total_memory"] * 100.0)
                    if d["total_memory"] > 0.0
                    else 0.0
                )
                table.add_row(
                    truncate_layer_name(name),
                    fmt_mem_new(memory),
                    f"{pct:.1f}%",
                    format_cache_value(d["activation_cache"], name),
                    format_cache_value(d["gradient_cache"], name),
                )

        # “Other Layers” aggregated row
        if d["other"]["total"] > 0:
            o = d["other"]
            table.add_row(
                "Other Layers",
                fmt_mem_new(o["total"]),
                f"{o['pct']:.1f}%",
                f"{fmt_mem_new(o['activation']['current'])} / "
                f"{fmt_mem_new(o['activation']['global'])}",
                f"{fmt_mem_new(o['gradient']['current'])} / "
                f"{fmt_mem_new(o['gradient']['global'])}",
            )

        # Empty case
        if not d["top_items"] and d["other"]["total"] <= 0:
            table.add_row("[dim]No layers detected[/dim]", "—", "—", "—", "—")

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 100)

        title = (
            f"[bold blue]Model #{d['model_index']}[/bold blue] • "
            f"Total: [white]{fmt_mem_new(d['total_memory'])}[/white]"
        )
        return Panel(Group(table), title=title, border_style="blue", width=panel_width)

    def get_notebook_renderable(self) -> HTML:
        """
        HTML (Jupyter) representation for top-N layers + "Other Layers".
        """
        d = self._data_service.compute_display_data()

        rows_html = ""
        if d["top_items"]:
            for name, memory in d["top_items"]:
                pct = (
                    (float(memory) / d["total_memory"] * 100.0)
                    if d["total_memory"] > 0.0
                    else 0.0
                )
                rows_html += f"""
                    <tr>
                        <td style="text-align:left;">{truncate_layer_name(name)}</td>
                        <td style="text-align:right;">{fmt_mem_new(memory)}</td>
                        <td style="text-align:right;">{pct:.1f}%</td>
                        <td style="text-align:right;">
                            {format_cache_value(d['activation_cache'], name)}
                        </td>
                        <td style="text-align:right;">
                            {format_cache_value(d['gradient_cache'], name)}
                        </td>
                    </tr>
                """

        o = d["other"]
        if o["total"] > 0:
            rows_html += f"""
                <tr style="color:gray;">
                    <td>Other Layers</td>
                    <td style="text-align:right;">{fmt_mem_new(o['total'])}</td>
                    <td style="text-align:right;">{o['pct']:.1f}%</td>
                    <td style="text-align:right;">
                        {fmt_mem_new(o['activation']['current'])} /
                        {fmt_mem_new(o['activation']['global'])}
                    </td>
                    <td style="text-align:right;">
                        {fmt_mem_new(o['gradient']['current'])} /
                        {fmt_mem_new(o['gradient']['global'])}
                    </td>
                </tr>
            """

        if not rows_html.strip():
            rows_html = """
                <tr>
                    <td colspan="5"
                        style="text-align:center; color:gray;">
                        No layers detected
                    </td>
                </tr>
            """

        html = f"""
        <div style="border:2px solid #2196f3; border-radius:8px;
                    padding:10px; margin-top:10px;">
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
                    {rows_html}
                </tbody>
            </table>
        </div>
        """
        return HTML(html)

    def get_dashboard_renderable(self) -> Dict[str, Any]:
        """
        Dashboard output: top layers sorted by total memory cost.
        """
        return self._data_service.compute_dashboard_data()


    def log_summary(self) -> None:
        """
        Print a global summary to the terminal:
          - total samples, models, avg & peak memory
          - top-3 activation/global peaks
          - top-3 gradient/global peaks
        """
        console = Console()

        layer_stats = self._summary_service.compute_layer_memory_summary()
        act_peaks = self._summary_service.compute_global_peaks(is_activation=True)
        grad_peaks = self._summary_service.compute_global_peaks(is_activation=False)

        top_acts = self._summary_service.top_n_from_dict(act_peaks, n=3)
        top_grads = self._summary_service.top_n_from_dict(grad_peaks, n=3)

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="white")

        self._render_section_layer_stats(table, layer_stats)
        self._render_section_topk(table, "TOP-3 ACTIVATIONS", top_acts, "cyan")
        self._render_section_topk(table, "TOP-3 GRADIENTS", top_grads, "green")

        panel = Panel(
            table,
            title="[bold blue]Model Layer - Summary[/bold blue]",
            border_style="blue",
        )
        console.print(panel)

    # -------------------------------------------------------------------------
    # Internal helpers for CLI summary formatting
    # -------------------------------------------------------------------------

    def _render_section_layer_stats(self, table: Table, stats: Dict[str, Any]) -> None:
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

    def _render_section_topk(
        self,
        table: Table,
        title: str,
        items: List,
        color: str,
    ) -> None:
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

