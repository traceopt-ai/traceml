# traceml/loggers/stdout/gradient_time_logger.py

from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from typing import Dict, Any

from .base_logger import BaseStdoutLogger
from .display_manager import GRADIENT_TIME_LAYOUT_NAME


class GradientTimeStdoutLogger(BaseStdoutLogger):
    """
    Logger that visualizes gradient/backward and optimizer step times.
    """

    def __init__(self):
        super().__init__(name="Gradient Time", layout_section_name=GRADIENT_TIME_LAYOUT_NAME)
        self._latest_snapshot: Dict[str, Any] = {}

    def _get_panel_renderable(self) -> Panel:
        """
        Render live gradient timing snapshot as a Rich table.
        """
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Phase", justify="left")
        table.add_column("Time (s)", justify="right")

        backward_time = self._latest_snapshot.get("backward_time", 0.0)
        optimizer_time = self._latest_snapshot.get("optimizer_time", 0.0)
        total_time = self._latest_snapshot.get("total_time", 0.0)

        table.add_row("Backward", f"{backward_time:.6f}")
        table.add_row("Optimizer", f"{optimizer_time:.6f}")
        table.add_row("Total Step", f"{total_time:.6f}")

        label = self._latest_snapshot.get("label", "train")
        drained = self._latest_snapshot.get("drained_events", 0)
        stale = self._latest_snapshot.get("stale", False)

        title = f"[Gradient Time] {label} (events={drained})"
        if stale:
            title += " [stale]"

        return Panel(
            table,
            title=title,
            border_style="cyan",
            width=60,
        )

    def log_summary(self, summary: Dict[str, Any]):
        """
        Final summary panel after display stops.
        """
        console = Console()

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold cyan3")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="bold white")

        keys = [
            ("total_steps", "TOTAL STEPS"),
            ("avg_backward_time", "AVG BACKWARD (s)"),
            ("avg_optimizer_time", "AVG OPTIMIZER (s)"),
            ("avg_total_time", "AVG STEP (s)"),
        ]

        for key, display in keys:
            val = summary.get(key, 0.0)
            if isinstance(val, float):
                display_val = f"{val:.6f}"
            else:
                display_val = str(val)
            table.add_row(display, "[cyan3]|[/cyan3]", display_val)

        panel = Panel(table, title=f"[bold cyan3]{self.name} - Final Summary", border_style="cyan3")
        console.print(panel)
