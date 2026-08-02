"""
Step Combined Renderer

CLI renderer for rank-combined step-time averages.

Behavior:
- show selected-clock diagnosis metrics, not public duration metrics
- If world_size == 1 (single-rank run): show only average rows
- Else: show Median/Worst/Worst Rank/Skew rows
"""

import math
import shutil

from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from traceml_ai.aggregator.display_drivers.layout import MODEL_COMBINED_LAYOUT
from traceml_ai.diagnostics.step_time import (
    LIVE_STEP_TIME_POLICY,
    build_step_diagnosis,
    format_cli_diagnosis,
)
from traceml_ai.diagnostics.trends import (
    DEFAULT_TREND_CONFIG,
    compute_trend_pct,
    format_trend_pct,
)
from traceml_ai.renderers.base_renderer import BaseRenderer
from traceml_ai.renderers.utils import fmt_time_run
from traceml_ai.step_time.model import StepTimeMetric, StepTimeResult
from traceml_ai.utils.step_time_window import (
    diagnose_step_time_window,
    median_iteration_component_share,
)

from .compute import StepCombinedComputer

METRIC_LABELS = {
    "input_wait": "IW",
    "h2d": "H2D",
    "forward": "FWD",
    "backward": "BWD",
    "optimizer_step": "OPT",
    "step_time": "STEP",
    "residual_proxy": "RESIDUAL",
}

TABLE_METRIC_ORDER = (
    "input_wait",
    "h2d",
    "forward",
    "backward",
    "optimizer_step",
    "step_time",
    "residual_proxy",
)


class StepCombinedRenderer(BaseRenderer):
    """
    CLI renderer for the selected-clock step-time diagnosis summary.
    """

    def __init__(self, db_path):
        super().__init__(
            name="Model Step Summary",
            layout_section_name=MODEL_COMBINED_LAYOUT,
        )
        self._computer = StepCombinedComputer(db_path=db_path)

    def _payload(self) -> StepTimeResult:
        """
        CLI compute is summary-only (cheap).

        The computer bridges transient empty reads from its own last-ok
        window. Persisted rows are not treated as proof of producer liveness.
        """
        return self._computer.compute_cli()

    def get_panel_renderable(self) -> Panel:
        payload = self._payload()

        window = payload.window
        if (
            window is None or not window.metrics
        ) and not self._computer.had_ok:
            # Never had data: normal warm-up, keep the calm waiting state
            # instead of alarming with a NO DATA diagnosis.
            return Panel(
                "Waiting for first fully completed step across all ranks…",
                title="Model Step Summary",
            )

        diag = (
            diagnose_step_time_window(
                window,
                policy=LIVE_STEP_TIME_POLICY,
                training_strategy=payload.training_strategy,
            ).primary
            if window is not None
            else build_step_diagnosis([])
        )
        diag_text = format_cli_diagnosis(diag)
        metrics = _table_metrics(window.metrics if window is not None else [])

        if not metrics:
            # Had data before, none now: the last-good empty-read bridge
            # expired.
            return Panel(
                Group(
                    diag_text,
                    "",
                    "No fresh step timing; the last window expired.",
                ),
                title="Model Step Summary",
                border_style="cyan",
            )

        step_metric = next(
            (m for m in metrics if m.metric == "step_time"), None
        )
        residual_metric = next(
            (m for m in metrics if m.metric == "residual_proxy"), None
        )

        # All metrics share the same window size by construction
        K = metrics[0].summary.steps_used
        world_size = metrics[0].coverage.world_size
        ranks_present = metrics[0].coverage.ranks_present
        single_rank = (world_size <= 1) or (ranks_present <= 1)

        table = Table(
            show_header=True, header_style="bold blue", box=None, expand=False
        )
        table.add_column("Metric", style="magenta")

        for m in metrics:
            title = METRIC_LABELS.get(
                m.metric,
                m.metric.replace("_", " ").title(),
            )
            table.add_column(title, justify="right")

        subtitle = (
            f"Averaged over last {K} fully completed steps"
            if K > 0
            else "Waiting for first fully completed step"
        )

        if single_rank:
            table.add_row(
                f"Average ({K} steps)",
                *[
                    _format_step_time_value(m.summary.worst_total)
                    for m in metrics
                ],
            )
        else:
            table.add_row(
                f"Median avg ({K} steps)",
                *[
                    _format_step_time_value(m.summary.median_total)
                    for m in metrics
                ],
            )
            table.add_row(
                f"Worst avg ({K} steps)",
                *[
                    _format_step_time_value(m.summary.worst_total)
                    for m in metrics
                ],
            )
            table.add_row(
                "Worst Rank",
                *[
                    (
                        f"r{m.summary.worst_rank}"
                        if m.summary.worst_rank is not None
                        else "—"
                    )
                    for m in metrics
                ],
            )
            table.add_row(
                "Skew (%)",
                *[_format_skew(m.summary.skew_pct) for m in metrics],
            )

        table.add_row("")
        if K >= DEFAULT_TREND_CONFIG.min_points and any(
            m.series is not None for m in metrics
        ):
            table.add_row(
                "Trend",
                *[_metric_trend_label(m, single_rank) for m in metrics],
            )
            table.add_row("")

        # Optional residual share line (still meaningful in both modes)
        if step_metric and residual_metric and window is not None:
            residual_share = median_iteration_component_share(
                window.per_rank_timing,
                "residual_proxy",
            )
            table.add_row(
                "Residual Share (%)",
                *[
                    (
                        (
                            f"[red]{residual_share * 100:.1f}%[/red]"
                            if residual_share is not None
                            else "n/a"
                        )
                        if m.metric == "residual_proxy"
                        else ""
                    )
                    for m in metrics
                ],
            )

        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 120)

        footer = (
            "\n\n[dim]"
            "Clock="
            f"{(window.clock if window is not None else 'cpu').upper()} | "
            "IW=input wait | H2D=host-to-device | FWD=forward | "
            "BWD=backward | OPT=optimizer | STEP=traced step | "
            "RESIDUAL=STEP−H2D−FWD−BWD−OPT"
            "[/dim]"
        )
        return Panel(
            Group(diag_text, "", table, footer),
            title=f"Model Step Summary ({subtitle})",
            border_style="cyan",
            width=width,
        )

    def get_dashboard_renderable(self) -> StepTimeResult:
        """
        Dashboard uses the same selected-clock Step Time payload as the CLI.
        """
        return self._computer.compute_dashboard()


def _table_metrics(
    metrics: list[StepTimeMetric],
) -> list[StepTimeMetric]:
    """
    Return selected-clock metrics in the CLI column order.

    The live table intentionally uses diagnosis metrics.
    """
    by_key = {metric.metric: metric for metric in metrics}
    ordered = [by_key[key] for key in TABLE_METRIC_ORDER if key in by_key]
    extras = [
        metric for metric in metrics if metric.metric not in TABLE_METRIC_ORDER
    ]
    return ordered + extras


def _format_skew(value: float | None) -> str:
    """Format a skew ratio without turning unavailable into zero."""
    return "n/a" if value is None else f"+{value * 100:.1f}%"


def _metric_trend_label(metric, single_rank: bool) -> str:
    if metric is None or metric.series is None:
        return "—"
    series = metric.series.worst if single_rank else metric.series.median
    return format_trend_pct(
        compute_trend_pct(series, config=DEFAULT_TREND_CONFIG),
        deadband_pct=DEFAULT_TREND_CONFIG.deadband_pct,
    )


def _format_step_time_value(ms: float | None) -> str:
    """Format selected Step Time values while keeping real zeros visible."""
    if ms is None:
        return "n/a"
    try:
        value = float(ms)
    except Exception:
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    if value <= 0.0:
        return "0.0 ms"
    return fmt_time_run(value)
