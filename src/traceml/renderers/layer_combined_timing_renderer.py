import shutil
from typing import Dict, Any, List, Optional

from rich.panel import Panel
from rich.table import Table
from rich.console import Group, Console
from IPython.display import HTML


from traceml.renderers.base_renderer import BaseRenderer
from traceml.database.database import Database
from traceml.renderers.display.cli_display_manager import (
    LAYER_COMBINED_TIMER_LAYOUT,
)
from traceml.renderers.layer_combined_timing.services import (
    LayerCombinedTimerData,
    LayerCombinedTimerSummary,
)
from traceml.renderers.utils import truncate_layer_name
from traceml.utils.formatting import fmt_time_ms
from traceml.database.remote_database_store import RemoteDBStore


class LayerCombinedTimerRenderer(BaseRenderer):
    """
    Layer-wise activation timing renderer.
    """

    def __init__(
        self,
        forward_db: Database = None,
        backward_db: Database = None,
        top_n_layers: int = 20,
        remote_store: Optional[RemoteDBStore] = None,
    ):
        super().__init__(
            name="Layer-wise Combined Timings",
            layout_section_name=LAYER_COMBINED_TIMER_LAYOUT,
        )
        self._service = LayerCombinedTimerData(
            forward_db=forward_db,
            backward_db=backward_db,
            top_n_layers=top_n_layers,
            remote_store=remote_store,
        )

        self._summary_service = LayerCombinedTimerSummary(
            forward_db=backward_db,
            backward_db=backward_db,
        )

    def get_panel_renderable(self) -> Panel:
        d = self._service.compute_display_data()

        table = Table(
            show_header=True,
            header_style="bold blue",
            box=None,
            pad_edge=False,
            padding=(0, 1),
        )

        table.add_column("Layer", justify="left", style="magenta")
        table.add_column("Forward (curr/avg)", justify="right", style="white")
        table.add_column("Backward (curr/avg)", justify="right", style="cyan")
        table.add_column("% ", justify="right", style="white")

        if d.get("top_items"):
            for r in d["top_items"]:
                table.add_row(
                    truncate_layer_name(r["layer"]),
                    f"{fmt_time_ms(r.get('forward_current', 0.0))}/{fmt_time_ms(r.get('forward_avg', 0.0))}",
                    f"{fmt_time_ms(r.get('backward_current', 0.0))}/{fmt_time_ms(r.get('backward_avg', 0.0))}",
                    f"{float(r.get('pct', 0.0)):.1f}%",
                )
        else:
            table.add_row("[dim]No timing data[/dim]", "—", "—", "—")

        o = d.get("other") or {}
        # show Other only if it contributes something
        if (
            o.get("total_forward_current", 0.0) + o.get("total_backward_current", 0.0)
        ) > 0:
            table.add_row(
                "Other Layers",
                f"{fmt_time_ms(o.get('total_forward_current', 0.0))}/{fmt_time_ms(o.get('total_forward_avg', 0.0))}",
                f"{fmt_time_ms(o.get('total_backward_current', 0.0))}/{fmt_time_ms(o.get('total_backward_avg', 0.0))}",
                f"{float(o.get('pct', 0.0)):.1f}%",
            )

        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 130)  # allow a bit wider

        return Panel(
            Group(table),
            title="[bold blue]Layer Timing (Forward + Backward)[/bold blue]",
            border_style="blue",
            width=width,
        )

    # ---------------- Notebook ----------------

    def get_notebook_renderable(self) -> HTML:
        d = self._service.compute_display_data()

        rows = ""
        for r in d.get("top_items", []):
            rows += f"""
            <tr>
                <td>{truncate_layer_name(r["layer"])}</td>
                <td style="text-align:right;">
                    {fmt_time_ms(r.get('forward_current', 0.0))}/{fmt_time_ms(r.get('forward_peak', 0.0))}
                </td>
                <td style="text-align:right;">
                    {fmt_time_ms(r.get('backward_current', 0.0))}/{fmt_time_ms(r.get('backward_peak', 0.0))}
                </td>
                <td style="text-align:right;">{float(r.get('pct', 0.0)):.1f}%</td>
            </tr>
            """

        if not rows.strip():
            rows = """
            <tr>
                <td colspan="4" style="text-align:center; color:gray;">
                    No timing data
                </td>
            </tr>
            """

        html = f"""
        <div style="border:2px solid #2196f3; border-radius:8px; padding:10px;">
            <h4 style="color:#2196f3; margin:0;">Layer Timing</h4>
            <table style="width:100%; border-collapse:collapse; margin-top:8px;">
                <thead>
                    <tr>
                        <th style="text-align:left;">Layer</th>
                        <th style="text-align:right;">Forward (curr/avg)</th>
                        <th style="text-align:right;">Backward (curr/avg)</th>
                        <th style="text-align:right;">Share(%)</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """
        return HTML(html)

    def get_dashboard_renderable(self) -> Dict[str, Any]:
        return self._service.compute_display_data()

    def log_summary(self, path) -> None:
        console = Console()

        layer_stats = self._summary_service.compute_layer_timing_summary()
        act_peaks = self._summary_service.compute_global_averages(is_forward=True)
        grad_peaks = self._summary_service.compute_global_averages(is_forward=False)

        top_acts = self._summary_service.top_n_from_dict(act_peaks, n=3)
        top_grads = self._summary_service.top_n_from_dict(grad_peaks, n=3)

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="white")

        self._render_section_layer_stats(table, layer_stats)
        self._render_section_topk(table, "Top 3 Forward Layers (Avg)", top_acts, "cyan")
        self._render_section_topk(
            table, "TOP-3 Backward Layers (Avg)", top_grads, "green"
        )

        panel = Panel(
            table,
            title="[bold blue]Layerwise Timing - Summary[/bold blue]",
            border_style="blue",
        )
        console.print(panel)

    def _render_section_layer_stats(self, table: Table, stats: Dict[str, Any]) -> None:
        table.add_row(
            "[blue]TOTAL LAYERS SEEN[/blue]",
            "[dim]|[/dim]",
            str(stats["total_layers_seen"]),
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
                    f"[{color}]{fmt_time_ms(value)}[/{color}]",
                )
        else:
            table.add_row("  • None", "", "—")
