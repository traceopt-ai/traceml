"""CLI presentation for canonical Step Time analyses.

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
from traceml_ai.diagnostics.step_time.api import StepDiagnosis
from traceml_ai.diagnostics.trends import (
    DEFAULT_TREND_CONFIG,
    compute_trend_pct,
    format_trend_pct,
)
from traceml_ai.renderers.base_renderer import BaseRenderer
from traceml_ai.renderers.utils import fmt_time_run
from traceml_ai.step_time.model import StepTimeMetric
from traceml_ai.step_time.pipeline import (
    LiveStepTimeResult,
    LiveStepTimeSession,
)

METRIC_LABELS = {
    "step_time": "Step Time",
    "input_wait": "IW",
    "h2d": "H2D",
    "forward": "FWD",
    "backward": "BWD",
    "optimizer_step": "OPT",
    "residual_proxy": "RESIDUAL",
}

TABLE_METRIC_ORDER = (
    "step_time",
    "input_wait",
    "h2d",
    "forward",
    "backward",
    "optimizer_step",
    "residual_proxy",
)


def format_cli_diagnosis(diagnosis: StepDiagnosis) -> str:
    """Render one concise Rich-friendly diagnosis block."""
    if diagnosis.kind == "BALANCED":
        style = "bold green"
    elif diagnosis.kind in {"NO_DATA", "WARMUP", "INCOMPLETE_DATA"}:
        style = "bold bright_black"
    elif diagnosis.kind == "INPUT_BOUND":
        style = "bold yellow"
    elif diagnosis.kind in {
        "H2D_BOUND",
        "INPUT_STRAGGLER",
        "COMPUTE_STRAGGLER",
        "H2D_STRAGGLER",
        "STRAGGLER",
        "RESIDUAL_HEAVY",
    }:
        style = "bold red" if diagnosis.severity == "crit" else "bold yellow"
    else:
        style = {
            "crit": "bold red",
            "warn": "bold yellow",
            "info": "bold",
        }.get(diagnosis.severity, "bold")

    lines = [
        f"[bold]Issue:[/bold] [{style}]{diagnosis.status}[/{style}]",
        f"[bold]Why:[/bold] {diagnosis.reason}",
        f"[bold]Next:[/bold] {diagnosis.action}",
    ]
    if diagnosis.note:
        lines.append(f"[bold]Note:[/bold] {diagnosis.note}")
    return "\n".join(lines)


class StepTimeRenderer(BaseRenderer):
    """Pure Rich presenter for one canonical live Step Time result."""

    def __init__(self, session: LiveStepTimeSession) -> None:
        super().__init__(
            name="Model Step Summary",
            layout_section_name=MODEL_COMBINED_LAYOUT,
        )
        self._session = session

    def get_panel_renderable(self) -> Panel:
        """Refresh once and render the session's analyzed result."""
        return self.render(self._session.refresh())

    def render(self, payload: LiveStepTimeResult) -> Panel:
        """Format one live result without loading or diagnosing telemetry."""
        window = payload.analysis.window
        if payload.freshness == "cold":
            # Never had data: normal warm-up, keep the calm waiting state
            # instead of alarming with a NO DATA diagnosis.
            return Panel(
                "Waiting for first fully completed step across all ranks…",
                title="Model Step Summary",
            )

        diag_text = format_cli_diagnosis(payload.analysis.diagnosis.primary)
        metrics = _table_metrics(window.metrics)

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

        step_time_metric = next(
            (m for m in metrics if m.metric == "step_time"), None
        )
        residual_metric = next(
            (m for m in metrics if m.metric == "residual_proxy"), None
        )

        # All metrics share the same window size by construction
        K = window.coverage.steps_used
        world_size = window.coverage.world_size
        ranks_present = window.coverage.ranks_present
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
                *[_format_step_time_value(m.worst_total) for m in metrics],
            )
        else:
            table.add_row(
                f"Median avg ({K} steps)",
                *[_format_step_time_value(m.median_total) for m in metrics],
            )
            table.add_row(
                f"Worst avg ({K} steps)",
                *[_format_step_time_value(m.worst_total) for m in metrics],
            )
            table.add_row(
                "Worst Rank",
                *[
                    (f"r{m.worst_rank}" if m.worst_rank is not None else "—")
                    for m in metrics
                ],
            )
            table.add_row(
                "Skew (%)",
                *[_format_skew(m.skew_pct) for m in metrics],
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
        if step_time_metric and residual_metric:
            residual_share = window.residual_share
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
            "Selected clock="
            f"{window.clock.upper()} | "
            "IW=input wait | H2D=host-to-device | FWD=forward | "
            "BWD=backward | OPT=optimizer | "
            "Step Time=IW + inner trace envelope | "
            "RESIDUAL=inner trace envelope−H2D−FWD−BWD−OPT"
            "[/dim]"
        )
        return Panel(
            Group(diag_text, "", table, footer),
            title=f"Model Step Summary ({subtitle})",
            border_style="cyan",
            width=width,
        )


def _table_metrics(
    metrics: list[StepTimeMetric],
) -> list[StepTimeMetric]:
    """
    Return selected-clock metrics in the CLI column order.

    The live table intentionally uses diagnosis metrics.
    """
    by_key = {metric.metric: metric for metric in metrics}
    return [by_key[key] for key in TABLE_METRIC_ORDER if key in by_key]


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
