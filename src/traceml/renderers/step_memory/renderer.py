"""
Step Memory Combined Renderer

This renderer presents a **window-peak, rank-agnostic** summary of
step-level peak memory metrics computed by `StepMemoryCombinedComputer`.

Table semantics
---------------
Columns:
- One column per memory metric:
    - Peak Allocated
    - Peak Reserved

Rows (over last K fully completed steps, aligned across ranks):
- Multi-rank:
    - Median Peak (max/K) : median rank's *peak* over the window
    - Worst Peak (max/K)  : worst rank's *peak* over the window
    - Worst Rank          : rank responsible for worst peak
    - Skew (%)            : (worst − median) / median
- Single-rank:
    - Peak (max/K)        : only rank's peak over the window

Optional (low-noise):
- Multi-rank: Worst Trend (Δ)
- Single-rank: Trend (Δ)

This table is intentionally stable and low-noise.
Per-step volatility belongs in plots, not summaries.
"""

from typing import Optional

from rich.panel import Panel

from traceml.aggregator.display_drivers.layout import MODEL_MEMORY_LAYOUT
from traceml.renderers.base_renderer import BaseRenderer

from .computer import StepMemoryMetricsComputer
from .formatter import StepMemoryRichFormatter
from .schema import StepMemoryCombinedResult


class StepMemoryRenderer(BaseRenderer):
    """
    CLI renderer for step-level combined memory summary.

    This renderer shows a **window-peak summary table** where:
    - columns = memory metrics (allocated, reserved)
    - rows    = rank-aware peak summary (+ optional trend)

    It is designed to surface **OOM risk and rank imbalance quickly**
    without overwhelming the user with per-step noise.
    """

    def __init__(self, db_path: str):
        super().__init__(
            name="Model Step Memory",
            layout_section_name=MODEL_MEMORY_LAYOUT,
        )
        self._computer = StepMemoryMetricsComputer(db_path=db_path)
        self._formatter = StepMemoryRichFormatter()
        self._cached: Optional[StepMemoryCombinedResult] = None

    def _payload(self) -> Optional[StepMemoryCombinedResult]:
        """
        Fetch latest computed payload.

        Uses a simple cache to avoid flicker when data is temporarily
        incomplete (e.g., ranks slightly out of sync).
        """
        payload = self._computer.compute_cli()
        if payload and payload.metrics:
            self._cached = payload
            return payload

        if self._cached is not None:
            return self._cached

        return payload

    def get_panel_renderable(self) -> Panel:
        """
        Render the CLI panel containing the memory summary table.

        Returns
        -------
        rich.panel.Panel
            Renderer-ready panel for CLI display.
        """
        payload = self._payload()

        return self._formatter.format(
            payload
            if payload is not None
            else StepMemoryCombinedResult(
                metrics=[],
                status_message=(
                    "Waiting for first fully completed step across all ranks…"
                ),
            )
        )

    def get_dashboard_renderable(self) -> StepMemoryCombinedResult:
        """
        Return the typed compute result directly to dashboard consumers.
        """
        return self._computer.compute_dashboard()
