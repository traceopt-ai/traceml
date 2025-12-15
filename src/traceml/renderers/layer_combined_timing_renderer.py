import shutil
from typing import Dict, Any

from rich.panel import Panel
from rich.table import Table
from rich.console import Group
from IPython.display import HTML

from traceml.renderers.base_renderer import BaseRenderer
from traceml.renderers.display.cli_display_manager import (
    COMBINED_TIMING_LAYOUT,
)
from traceml.renderers.layer_combined_timing.services import LayerTimingData
from traceml.renderers.utils import fmt_time_ms, truncate_layer_name


class LayerTimingRenderer(BaseRenderer):
    """
    Layer-wise activation timing renderer.
    """

    def __init__(self, timing_db, top_n_layers: int = 20):
        super().__init__(
            name="Layer Timing",
            layout_section_name=COMBINED_TIMING_LAYOUT,
        )
        self._service = LayerTimingData(
            timing_db=timing_db,
            top_n_layers=top_n_layers,
        )

    # ---------------- CLI ----------------

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
        table.add_column("Current", justify="right", style="white")
        table.add_column("Peak", justify="right", style="cyan")
        table.add_column("Device", justify="center", style="green")
        table.add_column("%", justify="right", style="white")

        for r in d["top_items"]:
            table.add_row(
                truncate_layer_name(r["layer"]),
                fmt_time_ms(r["current"]),
                fmt_time_ms(r["global"]),
                "GPU" if r["on_gpu"] else "CPU",
                f"{r['pct']:.1f}%",
            )

        o = d["other"]
        if o["current"] > 0:
            table.add_row(
                "Other Layers",
                fmt_time_ms(o["current"]),
                fmt_time_ms(o["global"]),
                "—",
                f"{o['pct']:.1f}%",
            )

        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 120)

        return Panel(
            Group(table),
            title="[bold blue]Layer Timing (Activation)[/bold blue]",
            border_style="blue",
            width=width,
        )

    # ---------------- Notebook ----------------

    def get_notebook_renderable(self) -> HTML:
        d = self._service.compute_display_data()

        rows = ""
        for r in d["top_items"]:
            rows += f"""
            <tr>
                <td>{truncate_layer_name(r["layer"])}</td>
                <td style="text-align:right;">{fmt_time_ms(r["current"])}</td>
                <td style="text-align:right;">{fmt_time_ms(r["global"])}</td>
                <td style="text-align:center;">{"GPU" if r["on_gpu"] else "CPU"}</td>
                <td style="text-align:right;">{r["pct"]:.1f}%</td>
            </tr>
            """

        if not rows:
            rows = """
            <tr>
                <td colspan="5" style="text-align:center; color:gray;">
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
                        <th>Layer</th>
                        <th style="text-align:right;">Current</th>
                        <th style="text-align:right;">Peak</th>
                        <th style="text-align:center;">Device</th>
                        <th style="text-align:right;">%</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """
        return HTML(html)

    # ---------------- Dashboard ----------------

    def get_dashboard_renderable(self) -> Dict[str, Any]:
        return self._service.compute_display_data()
