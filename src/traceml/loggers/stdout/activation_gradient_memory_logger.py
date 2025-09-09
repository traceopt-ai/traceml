from typing import Dict, Any, Optional, Tuple
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
import shutil

from .base_logger import BaseStdoutLogger
from .display_manager import ACTIVATION_GRADIENT_SUMMARY_LAYOUT_NAME
from traceml.utils.formatting import fmt_mem


class ActivationGradientMemoryStdoutLogger(BaseStdoutLogger):
    """
    Combined activation + gradient memory panel logger.
    """

    def __init__(
        self, layout_section_name: str = ACTIVATION_GRADIENT_SUMMARY_LAYOUT_NAME
    ):
        super().__init__(
            name="Activation & Gradient Memory",
            layout_section_name=layout_section_name,
        )
        self._latest_snapshot: Dict[str, Any] = {}

    @staticmethod
    def _pressure_badge(flag: Optional[bool]) -> str:
        if flag is True:
            return "[bold red]HIGH[/bold red]"
        if flag is False:
            return "[green]OK[/green]"
        return "[dim]n/a[/dim]"

    def _extract(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        snaps = self._latest_snapshot or {}
        act_data = (snaps.get("ActivationMemorySampler") or {}).get("data") or {}
        grad_data = (snaps.get("GradientMemorySampler") or {}).get("data") or {}
        return act_data, grad_data

    # --- Summary --------------------------------------------------------------
    def log_summary(self, summary: Dict[str, Any]):
        """
        Pretty-print final cumulative summary using a single combined object:

        summary = {
          "ever_seen": bool,
          "raw_events_kept": int,
          "activation": {"per_device_cumulative": {...}},
          "gradient":   {"per_device_cumulative": {...}},
        }
        """
        console = Console()

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="white")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="white")

        ever_seen = bool(summary.get("ever_seen", False))
        table.add_row(
            "EVER SEEN EVENTS", "[green]|[/green]", "Yes" if ever_seen else "No"
        )

        raw_kept = int(summary.get("raw_events_kept", 0) or 0)
        table.add_row("RAW EVENTS KEPT", "[green]|[/green]", str(raw_kept))

        # Activation cumulative
        act_per = (summary.get("activation") or {}).get("per_device_cumulative") or {}
        if act_per:
            table.add_row("", "", "")
            table.add_row(
                "[bold underline]ACTIVATION — PER-DEVICE CUMULATIVE[/bold underline]",
                "",
                "",
            )
            for dev, stats in act_per.items():
                c_count = int(stats.get("cumulative_count", 0) or 0)
                c_sum = float(stats.get("cumulative_sum_memory", 0.0) or 0.0)
                c_avg = float(stats.get("cumulative_avg_memory", 0.0) or 0.0)
                c_max = float(stats.get("cumulative_max_memory", 0.0) or 0.0)
                row = (
                    f"{dev} | count={c_count}  "
                    f"sum={fmt_mem(c_sum)}  avg={fmt_mem(c_avg)}  max={fmt_mem(c_max)}"
                )
                table.add_row(row, "", "")

        # Gradient cumulative
        grad_per = (summary.get("gradient") or {}).get("per_device_cumulative") or {}
        if grad_per:
            table.add_row("", "", "")
            table.add_row(
                "[bold underline]GRADIENT — PER-DEVICE CUMULATIVE[/bold underline]",
                "",
                "",
            )
            for dev, stats in grad_per.items():
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
            title=f"[bold cyan]{self.name} - Summary[/bold cyan]",
            border_style="cyan",
        )
        console.print(panel)

    # --- Live panel -----------------------------------------------------------

    def _header_block(self) -> Table:
        act, grad = self._extract()

        a_avg = float(
            act.get("overall_avg_memory", act.get("overall_avg_mb", 0.0)) or 0.0
        )
        a_events = int(act.get("drained_events", 0) or 0)
        a_stale = bool(act.get("stale", False))
        a_error = act.get("error")
        a_status = "[green]LIVE[/green]" if not a_stale else "[yellow]STALE[/yellow]"
        if a_error:
            a_status = "[bold red]ERROR[/bold red]"

        g_avg = float(
            grad.get("overall_avg_memory", grad.get("overall_avg_mb", 0.0)) or 0.0
        )
        g_events = int(grad.get("drained_events", 0) or 0)
        g_stale = bool(grad.get("stale", False))
        g_error = grad.get("error")
        g_status = "[green]LIVE[/green]"
        if g_stale:
            g_status = "[yellow]STALE[/yellow]"
        if g_error:
            g_status = "[bold red]ERROR[/bold red]"

        # Build table
        header = Table.grid(padding=(0, 3))
        header.add_column(justify="left", style="white")
        header.add_column(justify="left", style="white")
        header.add_column(justify="left", style="white")

        # Activation row
        header.add_row(
            f"[bold green]Activation Avg:[/bold green] {fmt_mem(a_avg)}",
            f"[bold green]Events:[/bold green] {a_events}",
            f"[bold green]Status:[/bold green] {a_status}",
        )

        # Gradient row
        header.add_row(
            f"[bold yellow]Gradient Avg:[/bold yellow] {fmt_mem(g_avg)}",
            f"[bold yellow]Events:[/bold yellow] {g_events}",
            f"[bold yellow]Status:[/bold yellow] {g_status}",
        )
        return header

    def _merged_device_table(self) -> Table:
        act, grad = self._extract()
        a_devs = act.get("devices") or {}
        g_devs = grad.get("devices") or {}

        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Section", justify="left", style="cyan", no_wrap=True)
        table.add_column("Device", justify="left", style="magenta", no_wrap=True)
        table.add_column("Avg", justify="right", style="white", no_wrap=True)
        table.add_column("Max", justify="right", style="white", no_wrap=True)
        table.add_column("Min>0", justify="right", style="white", no_wrap=True)
        table.add_column("Count", justify="right", style="white", no_wrap=True)
        table.add_column("Pressure", justify="center", style="white", no_wrap=True)

        def _row(section: str, dev: str, stats: Dict[str, Any], allow_mb_keys: bool):
            avg = stats.get("avg_memory")
            mx = stats.get("max_memory")
            mnz = stats.get("min_nonzero_memory")
            if allow_mb_keys:
                avg = avg if avg is not None else stats.get("avg_mb")
                mx = mx if mx is not None else stats.get("max_mb")
                mnz = mnz if mnz is not None else stats.get("min_nonzero_mb")
            table.add_row(
                section,
                str(dev),
                fmt_mem(avg),
                fmt_mem(mx),
                fmt_mem(mnz) if mnz is not None else "—",
                str(int(stats.get("count", 0) or 0)),
                self._pressure_badge(stats.get("pressure_90pct")),
            )

        # Activation rows
        if a_devs:
            for dev in sorted(a_devs.keys()):
                _row(
                    "[green]Activation[/green]",
                    dev,
                    a_devs[dev] or {},
                    allow_mb_keys=True,
                )
        else:
            table.add_row(
                "[green]Activation[/green]",
                "[dim]no devices[/dim]",
                "—",
                "—",
                "—",
                "0",
                "[dim]n/a[/dim]",
            )

        # Gradient rows
        if g_devs:
            for dev in sorted(g_devs.keys()):
                _row(
                    "[yellow]Gradient[/yellow]",
                    dev,
                    g_devs[dev] or {},
                    allow_mb_keys=False,
                )
        else:
            table.add_row(
                "[yellow]Gradient[/yellow]",
                "[dim]no devices[/dim]",
                "—",
                "—",
                "—",
                "0",
                "[dim]n/a[/dim]",
            )

        return table

    def _get_panel_renderable(self) -> Panel:
        outer_table = Table()
        outer_table.add_row(self._header_block())
        outer_table.add_row(self._merged_device_table())

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            outer_table,
            title="[bold cyan]Live Activation & Gradient Memory[/bold cyan]",
            border_style="cyan",
            width=panel_width,
        )
