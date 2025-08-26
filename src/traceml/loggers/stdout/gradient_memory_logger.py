from typing import Dict, Any, Optional
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
import shutil

from .base_logger import BaseStdoutLogger
from .display_manager import GRADIENT_SUMMARY_LAYOUT_NAME
from traceml.utils.formatting import fmt_mem  # shared formatter


class GradientMemoryStdoutLogger(BaseStdoutLogger):
    """
    Single-panel logger for gradient memory (no scrollable history).
    Expects GradientMemorySampler.sample() envelope with `data`:
      {
        "timestamp": float,
        "devices": {
           "cuda:0": {"count": int, "sum_memory": float, "avg_memory": float,
                      "max_memory": float, "min_nonzero_memory": Optional[float],
                      "pressure_90pct": Optional[bool]},
           ...
        },
        "overall_avg_memory": float,
        "drained_events": int,
        "stale": bool,
        "note": Optional[str],
        "error": Optional[str]
      }
    """

    def __init__(self, layout_section_name: str = GRADIENT_SUMMARY_LAYOUT_NAME):
        super().__init__(
            name="Gradient Memory", layout_section_name=layout_section_name
        )
        self._latest_snapshot: Dict[str, Any] = {}

    def _pressure_badge(self, flag: Optional[bool]) -> str:
        if flag is True:
            return "[bold red]HIGH[/bold red]"
        if flag is False:
            return "[green]OK[/green]"
        return "[dim]n/a[/dim]"

    def log_summary(self, summary: Dict[str, Any]):
        """Pretty-print final cumulative summary from GradientMemorySampler.get_summary()."""
        console = Console()

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="yellow")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="white")

        ever_seen = bool(summary.get("ever_seen", False))
        table.add_row(
            "EVER SEEN EVENTS", "[yellow]|[/yellow]", "Yes" if ever_seen else "No"
        )

        raw_kept = int(summary.get("raw_events_kept", 0) or 0)
        table.add_row("RAW EVENTS KEPT", "[yellow]|[/yellow]", str(raw_kept))

        # Per-device cumulative
        per_dev = summary.get("per_device_cumulative", {}) or {}
        if per_dev:
            table.add_row("", "", "")
            table.add_row(
                "[bold underline]PER-DEVICE CUMULATIVE[/bold underline]", "", ""
            )
            for dev, stats in per_dev.items():
                c_count = int(stats.get("cumulative_count", 0) or 0)
                c_sum = float(stats.get("cumulative_sum_memory", 0.0) or 0.0)
                c_avg = float(stats.get("cumulative_avg_memory", 0.0) or 0.0)
                c_max = float(stats.get("cumulative_max_memory", 0.0) or 0.0)
                row = (
                    f"{dev} | count={c_count}  "
                    f"sum={fmt_mem(c_sum)}  avg={fmt_mem(c_avg)}  max={fmt_mem(c_max)}"
                )
                table.add_row(row, "", "")

        panel = Panel(
            table,
            title=f"[bold yellow]{self.name} - Summary[/bold yellow]",
            border_style="yellow",
        )
        console.print(panel)

    def _get_panel_renderable(self) -> Panel:
        snap = self._latest_snapshot or {}
        devices = snap.get("devices", {}) or {}
        overall_avg = float(snap.get("overall_avg_memory", 0.0) or 0.0)
        drained = int(snap.get("drained_events", 0) or 0)
        stale = bool(snap.get("stale", False))
        note = snap.get("note")
        error = snap.get("error")

        # Header (Overall Avg | Events | Status)
        header = Table.grid(padding=(0, 2))
        header.add_column(justify="left")
        header.add_column(justify="right")

        status = "[green]LIVE[/green]"
        if stale:
            status = "[yellow]STALE[/yellow]"
        if error:
            status = "[bold red]ERROR[/bold red]"

        header.add_row(
            f"[bold]Overall Avg:[/bold] {fmt_mem(overall_avg)}",
            f"[bold]Events:[/bold] {drained}   [bold]Status:[/bold] {status}",
        )

        header.add_row("")
        # Per-device table
        dev_table = Table(
            show_header=True,
            header_style="bold yellow",
            box=None,
            expand=True,
            padding=(0, 1),
        )
        dev_table.add_column("Device", justify="left", style="magenta")
        dev_table.add_column("Avg", justify="right", style="white", no_wrap=True)
        dev_table.add_column("Max", justify="right", style="white", no_wrap=True)
        dev_table.add_column("Min>0", justify="right", style="white", no_wrap=True)
        dev_table.add_column("Count", justify="right", style="white", no_wrap=True)
        dev_table.add_column("Pressure", justify="center", style="white", no_wrap=True)

        if devices:
            # Stable ordering by device key
            for dev in sorted(devices.keys()):
                stats = devices[dev] or {}
                min_nz = stats.get("min_nonzero_memory")
                dev_table.add_row(
                    str(dev),
                    fmt_mem(stats.get("avg_memory")),
                    fmt_mem(stats.get("max_memory")),
                    fmt_mem(min_nz) if min_nz is not None else "—",
                    str(int(stats.get("count", 0) or 0)),
                    self._pressure_badge(stats.get("pressure_90pct")),
                )
        else:
            dev_table.add_row(
                "[dim]no devices[/dim]", "—", "—", "—", "0", "[dim]n/a[/dim]"
            )

        # Optional note / error
        body = Table.grid()
        body.add_row(header)
        body.add_row(dev_table)
        if error:
            body.add_row(
                Panel(str(error), border_style="red", title="Error", padding=(0, 1))
            )
        elif note:
            body.add_row(
                Panel(str(note), border_style="dim", title="Note", padding=(0, 1))
            )

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(50, int(cols * 0.5)), 90)

        return Panel(
            body,
            title="[bold yellow]Live Gradient Memory[/bold yellow]",
            border_style="yellow",
            width=panel_width,
        )
