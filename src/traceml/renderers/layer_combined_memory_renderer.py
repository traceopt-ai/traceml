from typing import Dict, Any, List, Optional
import shutil

from rich.panel import Panel
from rich.table import Table
from rich.console import Group, Console
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.database.database import Database
from traceml.renderers.display.cli_display_manager import (
    LAYER_COMBINED_MEMORY_LAYOUT,
)
from traceml.utils.formatting import fmt_mem_new

from traceml.renderers.combined_memory.services import (
    LayerCombinedMemoryData,
    LayerCombinedMemorySummary,
)
from traceml.renderers.utils import truncate_layer_name


class LayerCombinedMemoryRenderer(BaseRenderer):
    """
    Combined renderer using NEW total_peak_memory logic:

       total_peak = param + activation_peak + gradient_peak

    Sorting, % calculations, and display use this unified metric.
    """

    def __init__(
        self,
        layer_db: Database,
        activation_db: Database,
        gradient_db: Database,
        top_n_layers: Optional[int] = 20,
    ):
        super().__init__(
            name="Layer-wise Combined Memory",
            layout_section_name=LAYER_COMBINED_MEMORY_LAYOUT,
        )

        layer_table = layer_db.create_or_get_table("layer_memory")
        self._data_service = LayerCombinedMemoryData(
            layer_table=layer_table,
            activation_db=activation_db,
            gradient_db=gradient_db,
            top_n_layers=top_n_layers,
        )
        self._summary_service = LayerCombinedMemorySummary(
            layer_table=layer_table,
            activation_db=activation_db,
            gradient_db=gradient_db,
        )

    def get_panel_renderable(self) -> Panel:
        d = self._data_service.compute_display_data()

        table = Table(
            show_header=True,
            header_style="bold blue",
            box=None,
            pad_edge=False,
            padding=(0, 1),
        )

        table.add_column("Layer", justify="left", style="magenta")
        table.add_column("Params", justify="right", style="white")
        table.add_column("Act (current/peak)", justify="right", style="cyan")
        table.add_column("Grad (current/peak)", justify="right", style="green")
        table.add_column("% curr", justify="right", style="white")

        for row in d["top_items"]:
            table.add_row(
                truncate_layer_name(row["layer"]),
                fmt_mem_new(row["param_memory"]),
                f"{fmt_mem_new(row['activation_current'])} / "
                f"{fmt_mem_new(row['activation_peak'])}",
                f"{fmt_mem_new(row['gradient_current'])} / "
                f"{fmt_mem_new(row['gradient_peak'])}",
                f"{row['pct']:.1f}%",
            )

        o = d["other"]
        if o["total_current_memory"] > 0:
            table.add_row(
                "Other Layers",
                fmt_mem_new(o["param_memory"]),
                f"{fmt_mem_new(o['activation_current'])} / {fmt_mem_new(o['activation_peak'])}",
                f"{fmt_mem_new(o['gradient_current'])} / {fmt_mem_new(o['gradient_peak'])}",
                f"{o['pct']:.1f}%",
            )

        if not d["top_items"] and o["total_current_memory"] <= 0:
            table.add_row("[dim]No layers detected[/dim]", "—", "—", "—",  "—")

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 120)  # allow wider output

        title = (
            f"[bold blue]Model #{d['model_index']}[/bold blue] • "
            f"Total Current: [white]{fmt_mem_new(d['total_current_sum'])}[/white]"
        )
        return Panel(Group(table), title=title, border_style="blue", width=panel_width)

    def get_notebook_renderable(self) -> HTML:
        d = self._data_service.compute_display_data()

        rows_html = ""

        for row in d["top_items"]:
            rows_html += f"""
                <tr>
                    <td>{truncate_layer_name(row['layer'])}</td>

                    <td style="text-align:right;">{fmt_mem_new(row['param_memory'])}</td>

                    <td style="text-align:right;">
                        {fmt_mem_new(row['activation_current'])} /
                        {fmt_mem_new(row['activation_peak'])}
                    </td>

                    <td style="text-align:right;">
                        {fmt_mem_new(row['gradient_current'])} /
                        {fmt_mem_new(row['gradient_peak'])}
                    </td>

                    <td style="text-align:right;">{fmt_mem_new(row['total_current_memory'])}</td>
                    <td style="text-align:right;">{row['pct']:.1f}%</td>
                </tr>
            """

        # -------------------------------------------------------
        # OTHER LAYERS
        # -------------------------------------------------------
        o = d["other"]
        if o["total_current_memory"] > 0:
            rows_html += f"""
                <tr style="color:gray;">
                    <td>Other Layers</td>

                    <td style="text-align:right;">{fmt_mem_new(o['param_memory'])}</td>

                    <td style="text-align:right;">
                        {fmt_mem_new(o['activation_current'])} /
                        {fmt_mem_new(o['activation_peak'])}
                    </td>

                    <td style="text-align:right;">
                        {fmt_mem_new(o['gradient_current'])} /
                        {fmt_mem_new(o['gradient_peak'])}
                    </td>

                    <td style="text-align:right;">{fmt_mem_new(o['total_current_memory'])}</td>
                    <td style="text-align:right;">{o['pct']:.1f}%</td>
                </tr>
            """

        if not rows_html.strip():
            rows_html = """
                <tr>
                    <td colspan="6" style="text-align:center; color:gray;">
                        No layers detected
                    </td>
                </tr>
            """

        html = f"""
        <div style="border:2px solid #2196f3; border-radius:8px;
                    padding:10px; margin-top:10px;">
            <h4 style="color:#2196f3; margin:0;">
                Model #{d['model_index']}
                • Total Current: {fmt_mem_new(d['total_current_sum'])}
            </h4>

            <table style="width:100%; border-collapse:collapse; margin-top:8px;">
                <thead style="background:#f0f8ff;">
                    <tr>
                        <th style="text-align:left;">Layer</th>
                        <th style="text-align:right;">Params</th>
                        <th style="text-align:right;">Activation (cur/peak)</th>
                        <th style="text-align:right;">Gradient (cur/peak)</th>
                        <th style="text-align:right;">Total Curr</th>
                        <th style="text-align:right;">% curr</th>
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
        return self._data_service.compute_display_data()


    def log_summary(self) -> None:
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

    def _render_section_layer_stats(self, table: Table, stats: Dict[str, Any]) -> None:
        table.add_row(
            "[blue]MODEL MEMORY[/blue]", "[dim]|[/dim]",
            fmt_mem_new(stats["model_memory"]))

    def _render_section_topk(self, table: Table, title: str, items: List, color: str) -> None:
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
