from typing import Dict, Any, Optional

from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from .base_logger import BaseStdoutLogger
from .display_manager import ACTIVATION_SUMMARY_LAYOUT_NAME


class ActivationMemoryStdoutLogger(BaseStdoutLogger):
    """
    Single-panel logger for activation memory (no scrollable history).
    Expects snapshots shaped like ActivationQueueSampler.sample() output:
      {
        "timestamp": ...,
        "devices": {
           "cuda:0": {"count": N, "sum_mb": ..., "avg_mb": ..., "max_mb": ..., "min_nonzero_mb": ..., "pressure_90pct": bool},
           ...
        },
        "overall_avg_mb": ...,
        "drained_events": K,
        "stale": False/True,
        # optional: "note": "...helpful guidance..."
      }
    """

    def __init__(self, layout_section_name: str = ACTIVATION_SUMMARY_LAYOUT_NAME):
        super().__init__(
            name="Activation Memory", layout_section_name=layout_section_name
        )
        self._latest_snapshot: Dict[str, Any] = {}

    def log_summary(self, summary: Dict[str, Any]):
        """Pretty-print final cumulative summary from sampler.get_summary()."""
        console = Console()
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold magenta3")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="bold white")
        ever_seen = summary.get("ever_seen", False)
        table.add_row(
            "EVER SEEN EVENTS", "[magenta3]|[/magenta3]", "Yes" if ever_seen else "No"
        )
        raw_kept = summary.get("raw_events_kept", 0)
        table.add_row("RAW EVENTS KEPT", "[magenta3]|[/magenta3]", str(raw_kept))
        per_dev = summary.get("per_device_cumulative", {}) or {}
        if per_dev:
            table.add_row("", "", "")
            table.add_row(
                "[bold underline]PER-DEVICE CUMULATIVE[/bold underline]", "", ""
            )
            for dev, stats in per_dev.items():
                c_count = stats.get("cumulative_count", 0)
                c_sum = stats.get("cumulative_sum_memory", 0.0)
                c_avg = stats.get("cumulative_avg_memory", 0.0)
                c_max = stats.get("cumulative_max_memory", 0.0)
                row = (
                    f"{dev} | count={c_count}  "
                    f"sum={c_sum:.2f} MB  avg={c_avg:.2f} MB  max={c_max:.2f} MB"
                )
                table.add_row(row, "", "")

        panel = Panel(
            table,
            title=f"[bold magenta3]{self.name} - Final Summary",
            border_style="magenta3",
        )
        console.print(panel)

    def _format_mb(self, value: Optional[float]) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{v / 1024.0:.2f} GB" if v >= 1024.0 else f"{v:.2f} MB"

    def _pressure_badge(self, flag: Optional[bool]) -> str:
        if flag is True:
            return "[bold red]HIGH[/bold red]"
        if flag is False:
            return "[green]OK[/green]"
        return "[dim]n/a[/dim]"

    def _get_panel_renderable(self) -> Panel:
        snap = self._latest_snapshot or {}
        devices = snap.get("devices", {}) or {}
        overall_avg = snap.get("overall_avg_memory", 0.0) or 0.0
        drained = snap.get("drained_events", 0) or 0
        stale = bool(snap.get("stale", False))
        note = snap.get("note")

        # Top row
        header = Table.grid(padding=(0, 2))
        header.add_column(justify="left")
        header.add_column(justify="right")
        status = "[yellow]STALE[/yellow]" if stale else "[green]LIVE[/green]"
        header.add_row(
            f"[bold]Overall Avg:[/bold] {self._format_mb(overall_avg)}",
            f"[bold]Events:[/bold] {drained}   [bold]Status:[/bold] {status}",
        )

        # Per-device table
        dev_table = Table(
            show_header=True,
            header_style="bold magenta",
            box=None,
            expand=True,
            padding=(0, 1),
        )
        dev_table.add_column("Device", justify="left")
        dev_table.add_column("Avg", justify="right")
        dev_table.add_column("Max", justify="right")
        dev_table.add_column("Min>0", justify="right")
        dev_table.add_column("Count", justify="right")
        dev_table.add_column("Pressure", justify="center")

        if devices:
            for dev, stats in devices.items():
                dev_table.add_row(
                    str(dev),
                    self._format_mb(stats.get("avg_memory")),
                    self._format_mb(stats.get("max_memory")),
                    (
                        self._format_mb(stats.get("min_nonzero_memory"))
                        if stats.get("min_nonzero_memory") is not None
                        else "—"
                    ),
                    str(stats.get("count", 0)),
                    self._pressure_badge(stats.get("pressure_90pct")),
                )
        else:
            dev_table.add_row(
                "[dim]no devices[/dim]", "—", "—", "—", "0", "[dim]n/a[/dim]"
            )

        if note:
            note_panel = Panel(
                str(note), border_style="dim", title="Note", padding=(0, 1)
            )
            content = Table.grid()
            content.add_row(header)
            content.add_row(dev_table)
            content.add_row(note_panel)
        else:
            content = Table.grid()
            content.add_row(header)
            content.add_row(dev_table)

        return Panel(
            content,
            title="Live Activation Memory",
            border_style="cyan",
            width=100,
        )
