from typing import Dict, Any, Optional
from rich.panel import Panel
from rich.table import Table
import shutil

from .base_logger import BaseStdoutLogger
from .display_manager import ACTIVATION_GRADIENT_SUMMARY_LAYOUT_NAME
from traceml.utils.formatting import fmt_mem_new


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

    def _section_header(self, label: str, data: Dict[str, Any], color: str) -> Table:
        avg = float(
            data.get("overall_avg_memory", data.get("overall_avg_mb", 0.0)) or 0.0
        )
        events = int(data.get("drained_events", 0) or 0)
        stale = bool(data.get("stale", False))

        if stale:
            status = "[yellow]STALE[/yellow]"
        else:
            status = "[green]LIVE[/green]"

        t = Table.grid(padding=(0, 3))
        t.add_column(justify="left", style="white")
        t.add_column(justify="left", style="white")
        t.add_column(justify="left", style="white")

        t.add_row(
            f"[bold {color}]{label} Avg:[/bold {color}] {fmt_mem_new(avg)}",
            f"[bold {color}]Events:[/bold {color}] {events}",
            f"[bold {color}]Status:[/bold {color}] {status}",
        )
        return t

    def _device_table(
        self,
        section_label: str,
        devs: Dict[str, Any],
        color: str,
    ) -> Table:
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

        def _row(dev: str, stats: Dict[str, Any]):
            avg = stats.get("avg_memory")
            mx = stats.get("max_memory")
            mnz = stats.get("min_nonzero_memory")
            table.add_row(
                f"[{color}]{section_label}[/{color}]",
                str(dev),
                fmt_mem_new(avg),
                fmt_mem_new(mx),
                fmt_mem_new(mnz) if mnz is not None else "—",
                str(int(stats.get("count", 0) or 0)),
                self._pressure_badge(stats.get("pressure_90pct")),
            )

        if devs:
            for dev in sorted(devs.keys()):
                _row(dev, devs[dev] or {})
        else:
            table.add_row(
                f"[{color}]{section_label}[/{color}]",
                "[dim]no devices[/dim]",
                "—",
                "—",
                "—",
                "0",
                "[dim]n/a[/dim]",
            )
        return table

    def get_panel_renderable(self) -> Panel:
        act, grad = self._extract()
        a_devs = act.get("devices") or {}
        g_devs = grad.get("devices") or {}

        # Outer layout: Activation, space, Gradient
        outer = Table.grid(padding=(0, 0))
        outer.add_row(self._section_header("Activation", act, "green"))
        outer.add_row(self._device_table("Activation", a_devs, "green"))

        spacer = Table.grid()
        spacer.add_row("")
        outer.add_row(spacer)

        outer.add_row(self._section_header("Gradient", grad, "yellow"))
        outer.add_row(self._device_table("Gradient", g_devs, "yellow"))

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            outer,
            title="[bold cyan]Live Activation & Gradient Memory[/bold cyan]",
            border_style="cyan",
            width=panel_width,
        )

    # Summary
    def log_summary(self, summary: Dict[str, Any]):
        """
        Pretty-print summaries for Activation and Gradient samplers.
        """
        from rich.console import Console

        console = Console()

        act_summary = (summary or {}).get("ActivationMemorySampler") or {}
        grad_summary = (summary or {}).get("GradientMemorySampler") or {}

        def render_block(name: str, block: Dict[str, Any], color: str):
            t = Table.grid(padding=(0, 1))
            t.add_column(justify="left", style=color)
            t.add_column(justify="center", style="dim", no_wrap=True)
            t.add_column(justify="right", style="white")

            ever_seen = bool(block.get("ever_seen", False))
            t.add_row("EVER SEEN", "[cyan]|[/cyan]", "Yes" if ever_seen else "No")

            raw_kept = int(block.get("raw_events_kept", 0) or 0)
            t.add_row("RAW EVENTS KEPT", "[cyan]|[/cyan]", str(raw_kept))

            # per-device cumulative stats
            per_dev = block.get("per_device_cumulative") or {}
            if per_dev:
                t.add_row("", "", "")
                t.add_row("PER-DEVICE CUMULATIVE", "[cyan]|[/cyan]", "")
                for dev, stats in per_dev.items():
                    c_count = int(stats.get("cumulative_count", 0) or 0)
                    c_sum = float(stats.get("cumulative_sum_memory", 0.0) or 0.0)
                    c_avg = float(stats.get("cumulative_avg_memory", 0.0) or 0.0)
                    c_max = float(stats.get("cumulative_max_memory", 0.0) or 0.0)
                    row = (
                        f"{dev} | "
                        f"count={c_count}  "
                        f"sum={fmt_mem_new(c_sum)}  "
                        f"avg={fmt_mem_new(c_avg)}  "
                        f"max={fmt_mem_new(c_max)}"
                    )
                    t.add_row(row, "", "")

            console.print(
                Panel(
                    t,
                    title=f"[bold {color}]{name} Summary[/bold {color}]",
                    border_style=color,
                )
            )

        if act_summary:
            render_block("Activation", act_summary, "green")
        if grad_summary:
            render_block("Gradient", grad_summary, "yellow")
