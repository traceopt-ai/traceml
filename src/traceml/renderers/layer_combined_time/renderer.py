import shutil
from typing import Dict, Any, List, Optional

from rich.panel import Panel
from rich.table import Table
from rich.console import Group, Console
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.managers.cli_display_manager import (
    LAYER_COMBINED_TIMER_LAYOUT,
)
from traceml.renderers.layer_combined_time.compute import (
    LayerCombinedTimerData,
    LayerCombinedTimerSummary,
)
from traceml.renderers.layer_combined_time.schema import (
    LayerCombinedTimerResult
)
from traceml.renderers.utils import truncate_layer_name
from traceml.utils.formatting import fmt_time_ms
from traceml.database.remote_database_store import RemoteDBStore


class LayerCombinedTimeRenderer(BaseRenderer):
    """
    Layer-wise activation timing renderer.

    This renderer consumes a typed LayerCombinedTimerResult produced
    by the aggregator-side compute service.
    """

    def __init__(
        self,
        top_n_layers: int = 20,
        remote_store: Optional[RemoteDBStore] = None,
    ):
        super().__init__(
            name="Layer-wise Combined Timings",
            layout_section_name=LAYER_COMBINED_TIMER_LAYOUT,
        )

        self._service = LayerCombinedTimerData(
            top_n_layers=top_n_layers,
            remote_store=remote_store,
        )

        self._summary_service = LayerCombinedTimerSummary(
            remote_store=remote_store,
        )


    def get_panel_renderable(self) -> Panel:
        d: LayerCombinedTimerResult = self._service.compute_display_data()

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
        table.add_column("%", justify="right", style="white")

        if d.top_items:
            for r in d.top_items:
                table.add_row(
                    truncate_layer_name(r.layer),
                    f"{fmt_time_ms(r.forward_current)}/{fmt_time_ms(r.forward_avg)}",
                    f"{fmt_time_ms(r.backward_current)}/{fmt_time_ms(r.backward_avg)}",
                    f"{r.pct:.1f}%",
                )
        else:
            table.add_row("[dim]No timing data[/dim]", "—", "—", "—")

        o = d.other
        if (o.total_forward_current + o.total_backward_current) > 0:
            table.add_row(
                "Other Layers",
                f"{fmt_time_ms(o.total_forward_current)}/{fmt_time_ms(o.total_forward_avg)}",
                f"{fmt_time_ms(o.total_backward_current)}/{fmt_time_ms(o.total_backward_avg)}",
                f"{o.pct:.1f}%",
            )

        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 130)

        title = "[bold blue]Layer Timing (Forward + Backward)[/bold blue]"
        if d.incomplete:
            title += f" • [dim]missing ranks: {d.missing_ranks}[/dim]"

        return Panel(
            Group(table),
            title=title,
            border_style="blue",
            width=width,
        )


    def get_notebook_renderable(self) -> HTML:
        d: LayerCombinedTimerResult = self._service.compute_display_data()

        rows_html = ""
        for r in d.top_items:
            rows_html += f"""
            <tr>
                <td>{truncate_layer_name(r.layer)}</td>
                <td style="text-align:right;">
                    {fmt_time_ms(r.forward_current)}/{fmt_time_ms(r.forward_avg)}
                </td>
                <td style="text-align:right;">
                    {fmt_time_ms(r.backward_current)}/{fmt_time_ms(r.backward_avg)}
                </td>
                <td style="text-align:right;">{r.pct:.1f}%</td>
            </tr>
            """

        if not rows_html.strip():
            rows_html = """
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
                        <th style="text-align:right;">Share (%)</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """
        return HTML(html)


    def get_dashboard_renderable(self) -> LayerCombinedTimerResult:
        """
        Dashboard consumes the typed dataclass directly.
        """
        return self._service.compute_display_data()


    def log_summary(self, path) -> None:
        pass
