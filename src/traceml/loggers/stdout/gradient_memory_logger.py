from typing import Dict, Any, Optional

from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from .base_logger import BaseStdoutLogger
from .display_manager import GRADIENT_SUMMARY_LAYOUT_NAME


class GradientMemoryStdoutLogger(BaseStdoutLogger):
    """
    Single-panel logger for gradient memory (no scrollable history).
    Expects snapshots shaped like GradientMemorySampler.sample() output (enveloped),
    with `data` carrying:
      {
        "timestamp": ...,
        "devices": {
           "cuda:0": {"count": N, "sum_memory": ..., "avg_memory": ..., "max_memory": ..., "min_nonzero_memory": ..., "pressure_90pct": bool},
           ...
        },
        "overall_avg_memory": ...,
        "drained_events": K,
        "stale": False/True,
        "note": Optional[str],
        "error": Optional[str]
      }
    """

    def __init__(self, layout_section_name: str = GRADIENT_SUMMARY_LAYOUT_NAME):
        super().__init__(
            name="Gradient Memory", layout_section_name=layout_section_name
        )
        self._latest_snapshot: Dict[str, Any] = {}

    def log_summary(self, summary: Dict[str, Any]):
        """Pretty-print final cumulative summary from GradientMemorySampler.get_summary()."""
        console = Console()
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold magenta3")
        table.add_column(justify="center", style="dim", no_wrap=True)
        table.add_column(justify="right", style="bold white")

        ever_seen = bool(summary.get("ever_seen", False))
        table.add_row(
            "EVER SEEN EVENTS", "[magenta3]|[/magenta3]", "Yes" if ever_seen else "No"
        )

        raw_kept = int(summary.get("raw_events_kept", 0) or 0)
        table.add_row("RAW EVENTS KEPT", "[magenta3]|[/magenta3]", str(raw_kept))

        # Per-device cumulative stats
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
                    f"sum={c_sum:.2f} MB  avg={c_avg:.2f} MB  max={c_max:.2f} MB"
                )
                table.add_row(row, "", "")

        panel = Panel(
            table,
            title=f"[bold magenta3]{self.name} - Final Summary",
            border_style="magenta3",
        )
        console.print(panel)
