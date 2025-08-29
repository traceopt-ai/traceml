from typing import Dict, Any, Optional
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
import shutil

from .base_logger import BaseStdoutLogger
from .display_manager import ACTIVATION_SUMMARY_LAYOUT_NAME
from traceml.utils.formatting import fmt_mem  # shared MB/GB formatter


class ActivationMemoryStdoutLogger(BaseStdoutLogger):
    """
    Single-panel logger for activation memory (no scrollable history).
    Expects Activation sampler envelope with `data`:
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
        "note": Optional[str]
      }
    """

    def __init__(self, layout_section_name: str = ACTIVATION_SUMMARY_LAYOUT_NAME):
        super().__init__(
            name="Activation Memory", layout_section_name=layout_section_name
        )
        self._latest_snapshot: Dict[str, Any] = {}

    def _pressure_badge(self, flag: Optional[bool]) -> str:
        if flag is True:
            return "[bold red]HIGH[/bold red]"
        if flag is False:
            return "[green]OK[/green]"
        return "[dim]n/a[/dim]"

    def log_summary(self, summary: Dict[str, Any]):
        """Pretty-print final cumulative summary from sampler.get_summary()."""
        console = Console()

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="green")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="white")

        ever_seen = bool(summary.get("ever_seen", False))
        table.add_row(
            "EVER SEEN EVENTS", "[green]|[/green]", "Yes" if ever_seen else "No"
        )

        raw_kept = int(summary.get("raw_events_kept", 0) or 0)
        table.add_row("RAW EVENTS KEPT", "[green]|[/green]", str(raw_kept))

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
            title=f"[bold green]{self.name} - Summary[/bold green]",
            border_style="green",
        )
        console.print(panel)

    def _get_panel_renderable(self) -> Panel:
        snap = self._latest_snapshot or {}
        devices = snap.get("devices", {}) or {}
        overall_avg = float(
            snap.get("overall_avg_memory", snap.get("overall_avg_mb", 0.0)) or 0.0
        )
        drained = int(snap.get("drained_events", 0) or 0)
        stale = bool(snap.get("stale", False))
        note = snap.get("note")

        # Header (Overall Avg | Events | Status)
        header = Table.grid(padding=(0, 2))
        header.add_column(justify="left")
        header.add_column(justify="right")

        status = "[green]LIVE[/green]"
        if stale:
            status = "[yellow]STALE[/yellow]"

        header.add_row(
            f"[bold]Overall Avg:[/bold] {fmt_mem(overall_avg)}",
            f"[bold]Events:[/bold] {drained}   [bold]Status:[/bold] {status}",
        )

        header.add_row("")
        dev_table = Table(
            show_header=True,
            header_style="bold green",
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
            for dev in sorted(devices.keys()):
                stats = devices[dev] or {}
                min_nz = stats.get("min_nonzero_memory") or stats.get("min_nonzero_mb")
                dev_table.add_row(
                    str(dev),
                    fmt_mem(stats.get("avg_memory") or stats.get("avg_mb")),
                    fmt_mem(stats.get("max_memory") or stats.get("max_mb")),
                    fmt_mem(min_nz) if min_nz is not None else "—",
                    str(int(stats.get("count", 0) or 0)),
                    self._pressure_badge(stats.get("pressure_90pct")),
                )
        else:
            dev_table.add_row(
                "[dim]no devices[/dim]", "—", "—", "—", "0", "[dim]n/a[/dim]"
            )

        body = Table.grid()
        body.add_row(header)
        body.add_row(dev_table)


        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(50, int(cols * 0.5)), 90)

        return Panel(
            body,
            title="[bold green]Live Activation Memory[/bold green]",
            border_style="green",
            width=panel_width,
        )
