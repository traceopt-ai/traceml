from typing import Dict, Any, List, Optional
import shutil

from rich.panel import Panel
from rich.table import Table
from rich.console import Group, Console
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.managers.cli_display_manager import (
    LAYER_COMBINED_MEMORY_LAYOUT,
)
from traceml.utils.formatting import fmt_mem_new
from traceml.renderers.utils import truncate_layer_name

from traceml.renderers.layer_combined_memory.compute1 import (
    LayerCombinedMemoryData,
    LayerCombinedMemorySummary,
)

from traceml.renderers.layer_combined_memory.schema import (
    LayerCombinedMemoryResult,
    LayerCombinedMemoryRow,
    LayerCombinedOther,
)


class LayerCombinedMemoryRenderer(BaseRenderer):
    """
    Renderer for combined per-layer memory usage.
    This renderer displays a *capacity-oriented* view of memory:
        total_current = param + forward_current + backward_current
        total_peak    = param + forward_peak    + backward_peak

    Notes
    -----
    - Uses the typed `LayerCombinedMemoryResult` contract
    """

    def __init__(
        self,
        remote_store: Optional[Any] = None,
        top_n_layers: Optional[int] = 5,
    ):
        super().__init__(
            name="Layer-wise Combined Memory",
            layout_section_name=LAYER_COMBINED_MEMORY_LAYOUT,
        )

        self._compute_service = LayerCombinedMemoryData(
            top_n_layers=top_n_layers,
            remote_store=remote_store,
        )
        self._summary_service = LayerCombinedMemorySummary(
            remote_store=remote_store,
        )

    def get_panel_renderable(self) -> Panel:
        result: LayerCombinedMemoryResult = (
            self._compute_service.compute_display_data()
        )

        table = Table(
            show_header=True,
            header_style="bold blue",
            box=None,
            pad_edge=False,
            padding=(0, 1),
        )

        table.add_column("Layer", justify="left", style="magenta")
        table.add_column("Params", justify="right", style="white")
        table.add_column("Forward (curr/peak)", justify="right", style="cyan")
        table.add_column("Backward (curr/peak)", justify="right", style="green")
        table.add_column("% curr", justify="right", style="white")

        for row in result.top_items:
            table.add_row(
                truncate_layer_name(row.layer),
                fmt_mem_new(row.param_memory),
                f"{fmt_mem_new(row.forward_current)}/{fmt_mem_new(row.forward_peak)}",
                f"{fmt_mem_new(row.backward_current)}/{fmt_mem_new(row.backward_peak)}",
                f"{row.pct:.1f}%",
            )

        other: LayerCombinedOther = result.other
        if other.total_current_memory > 0:
            table.add_row(
                "Other Layers",
                fmt_mem_new(other.param_memory),
                f"{fmt_mem_new(other.forward_current)}/{fmt_mem_new(other.forward_peak)}",
                f"{fmt_mem_new(other.backward_current)}/{fmt_mem_new(other.backward_peak)}",
                f"{other.pct:.1f}%",
            )

        if not result.top_items and other.total_current_memory <= 0:
            table.add_row("[dim]No layers detected[/dim]", "—", "—", "—", "—")

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 120)  # allow wider output

        status_suffix = (
            f" • [dim]{result.status_message}[/dim]"
            if result.status_message
            else ""
        )

        title = (
            f"[bold blue]Model #{result.model_index}[/bold blue] • "
            f"[white]curr=max across ranks, peak=max curr across steps[/white]"
            f"{status_suffix}"
        )

        return Panel(
            Group(table), title=title, border_style="blue", width=panel_width,
        )


    def get_notebook_renderable(self) -> HTML:
        result: LayerCombinedMemoryResult = (
            self._compute_service.compute_display_data()
        )

        rows_html = ""

        for row in result.top_items:
            rows_html += f"""
                <tr>
                    <td>{truncate_layer_name(row.layer)}</td>
                    <td style="text-align:right;">{fmt_mem_new(row.param_memory)}</td>
                    <td style="text-align:right;">
                        {fmt_mem_new(row.forward_current)}/{fmt_mem_new(row.forward_peak)}
                    </td>
                    <td style="text-align:right;">
                        {fmt_mem_new(row.backward_current)}/{fmt_mem_new(row.backward_peak)}
                    </td>
                    <td style="text-align:right;">
                        {fmt_mem_new(row.total_current_memory)}
                    </td>
                    <td style="text-align:right;">{row.pct:.1f}%</td>
                </tr>
            """

        other = result.other
        if other.total_current_memory > 0:
            rows_html += f"""
                <tr style="color:gray;">
                    <td>Other Layers</td>
                    <td style="text-align:right;">{fmt_mem_new(other.param_memory)}</td>
                    <td style="text-align:right;">
                        {fmt_mem_new(other.forward_current)}/{fmt_mem_new(other.forward_peak)}
                    </td>
                    <td style="text-align:right;">
                        {fmt_mem_new(other.backward_current)}/{fmt_mem_new(other.backward_peak)}
                    </td>
                    <td style="text-align:right;">
                        {fmt_mem_new(other.total_current_memory)}
                    </td>
                    <td style="text-align:right;">{other.pct:.1f}%</td>
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
                Model #{result.model_index}
                • Total Current: {fmt_mem_new(result.total_current_sum)}
            </h4>

            <table style="width:100%; border-collapse:collapse; margin-top:8px;">
                <thead style="background:#f0f8ff;">
                    <tr>
                        <th style="text-align:left;">Layer</th>
                        <th style="text-align:right;">Params</th>
                        <th style="text-align:right;">Forward(curr/peak)</th>
                        <th style="text-align:right;">Backward(curr/peak)</th>
                        <th style="text-align:right;">Current Total</th>
                        <th style="text-align:right;">Total Share (%)</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """
        return HTML(html)

    def get_dashboard_renderable(self) -> LayerCombinedMemoryResult:
        """
        Return the typed compute result directly to dashboard consumers.
        """
        return self._compute_service.compute_display_data()


    def log_summary(self, path) -> None:
        console = Console()

        layer_stats = self._summary_service.compute_layer_memory_summary()
        fwd_peaks = self._summary_service.compute_global_peaks(is_forward=True)
        bwd_peaks = self._summary_service.compute_global_peaks(is_forward=False)

        top_fwds = self._summary_service.top_n_from_dict(fwd_peaks, n=3)
        top_bwds = self._summary_service.top_n_from_dict(bwd_peaks, n=3)

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="white")

        self._render_section_layer_stats(table, layer_stats)
        self._render_section_topk(table, "TOP-3 FORWARD", top_fwds, "cyan")
        self._render_section_topk(table, "TOP-3 BACKWARD", top_bwds, "green")

        panel = Panel(
            table,
            title="[bold blue]Model Layer - Summary[/bold blue]",
            border_style="blue",
        )
        console.print(panel)

    def _render_section_layer_stats(self, table: Table, stats: Dict[str, Any]) -> None:
        table.add_row(
            "[blue]MODEL MEMORY[/blue]",
            "[dim]|[/dim]",
            fmt_mem_new(stats["model_memory"]),
        )

    def _render_section_topk(
            self, table: Table, title: str, items: List, color: str
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
