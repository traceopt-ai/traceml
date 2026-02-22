import shutil
from typing import Any, Optional

from IPython.display import HTML
from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from traceml.renderers.base_renderer import BaseRenderer
from traceml.aggregator.display_drivers.layout import (
    LAYER_COMBINED_MEMORY_LAYOUT,
)
from traceml.renderers.layer_combined_memory.compute import (
    LayerCombinedMemoryData,
    LayerCombinedMemorySummary,
)
from traceml.renderers.layer_combined_memory.schema import (
    LayerCombinedMemoryResult,
    LayerCombinedOther,
)
from traceml.renderers.utils import truncate_layer_name
from traceml.utils.formatting import fmt_mem_new


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
        table.add_column(
            "Backward (curr/peak)", justify="right", style="green"
        )
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
        panel_width = min(
            max(100, int(cols * 0.75)), 120
        )  # allow wider output

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
            Group(table),
            title=title,
            border_style="blue",
            width=panel_width,
        )

    def get_dashboard_renderable(self) -> LayerCombinedMemoryResult:
        """
        Return the typed compute result directly to dashboard consumers.
        """
        return self._compute_service.compute_display_data()

    def get_notebook_renderable(self) -> HTML:
        pass

    def log_summary(self, path) -> None:
        pass
