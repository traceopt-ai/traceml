"""
Process renderer.

This module contains all presentation logic for process-level telemetry.
"""

import shutil
from typing import Any, Dict, Optional

from rich.panel import Panel
from rich.table import Table

from traceml_ai.aggregator.display_drivers.layout import PROCESS_LAYOUT
from traceml_ai.loggers.error_log import get_error_logger
from traceml_ai.renderers.base_renderer import BaseRenderer
from traceml_ai.renderers.shared.freshness import (
    RankReporting,
    RunReporting,
    stale_rank_label,
    stale_run_label,
)
from traceml_ai.utils.formatting import fmt_mem_new, fmt_mem_triple

from .computer import ProcessMetricsComputer
from .dashboard_models import ProcessDashboardPayload


class ProcessRenderer(BaseRenderer):
    """
    Renderer for process-level telemetry.
    """

    NAME = "Process"

    def __init__(
        self,
        db_path: str,
        sampler_interval_s: Optional[float] = None,
    ) -> None:
        super().__init__(name=self.NAME, layout_section_name=PROCESS_LAYOUT)
        self.db_path = db_path
        self._computer = ProcessMetricsComputer(
            db_path=self.db_path,
            sampler_interval_s=sampler_interval_s,
        )
        self._logger = get_error_logger(self.NAME + "Renderer")
        # The run-wide status from the latest panel read, so the
        # terminal's run-wide line costs no read of its own.
        self._run_reporting: Optional[RunReporting] = None

    def get_staleness_text(self, after_s: Optional[float] = None) -> str:
        """``No Process data from any rank for 42s.`` once the run is quiet.

        From the read :meth:`get_panel_renderable` made this tick, so call
        it after that. Empty while any rank still sends, before the first
        read, and when this tick has no status. With ``after_s``, on the
        arrival clock, also empty unless the run's newest arrival is later
        than that.
        """
        run = self._run_reporting
        if run is None or not run.is_stale:
            return ""
        seen = run.last_seen_s
        if after_s is not None and (seen is None or seen <= after_s):
            return ""
        return stale_run_label(run)

    def _read_snapshot(self) -> Dict[str, Any]:
        """This tick's snapshot, keeping its run-wide status.

        A read that raises leaves no run-wide status, then re-raises for
        the driver to show. An earlier tick's status never stands in.
        """
        self._run_reporting = None
        snap = self._computer.compute_cli()
        run = snap.get("run_reporting")
        self._run_reporting = RunReporting(**run) if run else None
        return snap

    def get_panel_renderable(self) -> Panel:
        """
        Build the Rich panel for process telemetry.

        The snapshot is already aggregated by ProcessMetricsComputer:
        - CPU is worst-rank CPU at latest committed seq
        - GPU memory is taken from the least-headroom rank
        - a rank that stopped sending Process data is named, because the
          figures above stay anchored on its last seq
        """
        snap = self._read_snapshot()

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="bright_white", no_wrap=True)
        table.add_column(justify="right", style="bright_white", no_wrap=True)

        cpu_used = float(snap.get("cpu_used") or 0.0)
        cpu_cores = cpu_used / 100.0
        table.add_row(
            "[bold green]CPU (worst rank)[/bold green]",
            f"{cpu_cores:.2f} cores",
        )

        gpu_used = snap.get("gpu_used")
        gpu_reserved = snap.get("gpu_reserved")
        gpu_total = snap.get("gpu_total")
        gpu_rank = snap.get("gpu_rank")

        if (
            gpu_used is not None
            and gpu_reserved is not None
            and gpu_total is not None
        ):
            gpu_str = fmt_mem_triple(gpu_used, gpu_reserved, gpu_total)
            if gpu_rank is not None:
                gpu_str += f" [dim](rank {gpu_rank})[/dim]"
        else:
            gpu_str = "[red]Not available[/red]"

        table.add_row(
            "[bold green]GPU MEM (used/reserved/total)[/bold green]",
            gpu_str,
        )

        gpu_imbalance = snap.get("gpu_used_imbalance")
        if gpu_imbalance is not None and gpu_imbalance > 0.0:
            table.add_row(
                "[bold green]GPU used imbalance[/bold green]",
                fmt_mem_new(gpu_imbalance),
            )

        for rank in snap.get("rank_reporting") or ():
            if rank.get("freshness") == "stale":
                table.add_row(
                    f"[bold yellow]{stale_rank_label(RankReporting(**rank))}"
                    "[/bold yellow]",
                    "",
                )

        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            table,
            title="[bold cyan]Process Metrics[/bold cyan]",
            border_style="cyan",
            width=width,
        )

    def get_dashboard_renderable(self) -> ProcessDashboardPayload:
        """
        Return the dashboard payload for the Process card.
        """
        return self._computer.compute_dashboard()
