
from typing import Dict, Any, Optional
import shutil
from rich.table import Table
from rich.panel import Panel
from rich.console import Group

from .base_logger import BaseStdoutLogger
from .display_manager import ACTIVATION_GRADIENT_SUMMARY_LAYOUT_NAME
from traceml.utils.formatting import fmt_mem, fmt_percent


class ActivationGradientMemoryStdoutLogger(BaseStdoutLogger):
    """
    Combined activation + gradient memory panel logger.

    Expects BaseStdoutLogger.log() to receive a dict shaped like:
      {
        "ActivationSampler": { "data": { "devices": {...}, "overall_avg_mb": ... }, ... },
        "GradientSampler":   { "data": { "devices": {...}, "overall_avg_mb": ... }, ... }
      }
    """

    def __init__(self):
        super().__init__(
            name="Activation+Gradient Memory",
            layout_section_name=ACTIVATION_GRADIENT_SUMMARY_LAYOUT_NAME,
        )
        self._latest_snapshot: Dict[str, Any] = {}

    def _get_panel_renderable(self) -> Panel:
        snaps = self._latest_snapshot or {}
        act_data = (snaps.get("ActivationSampler") or {}).get("data") or {}
        grad_data = (snaps.get("GradientSampler") or {}).get("data") or {}

        act_devices = act_data.get("devices", {})
        grad_devices = grad_data.get("devices", {})

        # --- Build tables ---
        table = Table.grid(padding=(0, 2))
        table.add_column("Device", justify="left", style="cyan")
        table.add_column("Activations", justify="right", style="magenta", no_wrap=True)
        table.add_column("Gradients", justify="right", style="yellow", no_wrap=True)

        if act_devices or grad_devices:
            # Union of device keys from both
            all_devices = sorted(set(act_devices.keys()) | set(grad_devices.keys()))
            for dev in all_devices:
                act_row = act_devices.get(dev, {})
                grad_row = grad_devices.get(dev, {})

                act_mem = act_row.get("avg_mb") or act_row.get("sum_mb") or 0.0
                grad_mem = grad_row.get("avg_mb") or grad_row.get("sum_mb") or 0.0

                table.add_row(
                    f"[cyan]{dev}[/cyan]",
                    fmt_mem(act_mem),
                    fmt_mem(grad_mem),
                )
        else:
            table.add_row("[dim]No data[/dim]", "—", "—")

        # --- Overall row ---
        act_overall = act_data.get("overall_avg_mb", 0.0)
        grad_overall = grad_data.get("overall_avg_mb", 0.0)
        table.add_row(
            "[bold]Overall[/bold]",
            fmt_mem(act_overall),
            fmt_mem(grad_overall),
        )

        # --- Panel width ---
        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(80, int(cols * 0.75)), 120)

        return Panel(
            Group(table),
            title="[bold cyan]Activation + Gradient Memory[/bold cyan]",
            border_style="cyan",
            width=panel_width,
        )

    def log_summary(self, summary: Dict[str, Any]):
        """
        Render a compact summary (averages/peaks for both activation and gradient).
        Expects summary shaped like:
          {
            "ActivationSampler": {...},
            "GradientSampler": {...}
          }
        """
        from rich.console import Console

        console = Console()

        def render_block(name: str, block: Dict[str, Any], color: str):
            t = Table.grid(padding=(0, 1))
            t.add_column(justify="left", style=color)
            t.add_column(justify="center", style="dim")
            t.add_column(justify="right", style="white")
            for k, v in block.items():
                key = str(k).replace("_", " ").upper()
                if isinstance(v, (int, float)):
                    val = fmt_mem(v)
                else:
                    val = str(v)
                t.add_row(key, "|", val)
            console.print(
                Panel(t, title=f"[bold {color}]{name} Summary[/bold {color}]", border_style=color)
            )

        act_summary = (summary or {}).get("ActivationSampler") or {}
        grad_summary = (summary or {}).get("GradientSampler") or {}
        if act_summary:
            render_block("Activations", act_summary, "magenta")
        if grad_summary:
            render_block("Gradients", grad_summary, "yellow")
