"""
Rich formatter for the step-memory CLI renderer.
"""

from __future__ import annotations

import shutil

from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from traceml.core import Formatter
from traceml.utils.formatting import fmt_mem_new

from .diagnostics import build_step_memory_diagnosis, format_cli_diagnosis
from .schema import StepMemoryCombinedMetric, StepMemoryCombinedResult


class StepMemoryRichFormatter(Formatter[StepMemoryCombinedResult, Panel]):
    """Format step-memory combined metrics as a Rich panel."""

    name = "step_memory_rich"

    def format(self, payload: StepMemoryCombinedResult) -> Panel:
        """
        Convert a step-memory payload into the CLI Rich panel.
        """
        if not payload.metrics:
            return Panel(
                (
                    payload.status_message
                    if payload.status_message
                    else "Waiting for first fully completed step across all ranks…"
                ),
                title="Model Step Memory",
            )

        metrics = self._sort_metrics(payload.metrics)

        diag = build_step_memory_diagnosis(metrics)
        diag_text = format_cli_diagnosis(diag)

        # All metrics share the same window size by construction.
        steps_used = metrics[0].summary.steps_used
        single_rank = bool(
            metrics[0].coverage.world_size <= 1
            or metrics[0].coverage.ranks_present <= 1
        )

        table = self._build_table(metrics, single_rank=single_rank)
        subtitle = (
            f"Peaks over last {steps_used} fully completed steps"
            if steps_used > 0
            else "Waiting for first fully completed step"
        )
        footer = (
            "\n\n[dim]Peaks = max over last K aligned steps for the only rank.[/dim]"
            if single_rank
            else "\n\n[dim]Peaks = per-rank max over last K; median/worst = across ranks.[/dim]"
        )

        cols, _ = shutil.get_terminal_size()
        width = min(max(100, int(cols * 0.75)), 120)

        return Panel(
            Group(
                diag_text,
                "",
                table,
                footer,
            ),
            title=f"Model Step Memory ({subtitle})",
            border_style="cyan",
            width=width,
        )

    @staticmethod
    def _sort_metrics(
        metrics: list[StepMemoryCombinedMetric],
    ) -> list[StepMemoryCombinedMetric]:
        """
        Return metrics in stable CLI display order.
        """

        def _sort_key(metric: StepMemoryCombinedMetric) -> int:
            if metric.metric == "peak_allocated":
                return 0
            if metric.metric == "peak_reserved":
                return 1
            return 99

        return sorted(metrics, key=_sort_key)

    def _build_table(
        self,
        metrics: list[StepMemoryCombinedMetric],
        *,
        single_rank: bool,
    ) -> Table:
        """
        Build the window-peak summary table.
        """
        steps_used = metrics[0].summary.steps_used

        table = Table(
            show_header=True,
            header_style="bold blue",
            box=None,
            expand=False,
        )

        table.add_column("Metric", style="magenta")
        for metric in metrics:
            table.add_column(
                self._metric_title(metric.metric), justify="right"
            )

        if single_rank:
            table.add_row(
                f"Peak (max/{steps_used})",
                *[
                    fmt_mem_new(metric.summary.worst_peak)
                    for metric in metrics
                ],
            )
        else:
            table.add_row(
                f"Median Peak (max/{steps_used})",
                *[
                    fmt_mem_new(metric.summary.median_peak)
                    for metric in metrics
                ],
            )
            table.add_row(
                f"Worst Peak (max/{steps_used})",
                *[
                    fmt_mem_new(metric.summary.worst_peak)
                    for metric in metrics
                ],
            )
            table.add_row(
                "Worst Rank",
                *[
                    (
                        f"r{metric.summary.worst_rank}"
                        if metric.summary.worst_rank is not None
                        else "—"
                    )
                    for metric in metrics
                ],
            )
            table.add_row(
                "Skew (%)",
                *[
                    f"+{metric.summary.skew_pct * 100:.1f}%"
                    for metric in metrics
                ],
            )

        table.add_row("")
        table.add_row(
            "Head/Tail Delta" if single_rank else "Head/Tail Delta (worst)",
            *[
                self._format_worst_trend_delta(
                    metric,
                    single_rank=single_rank,
                )
                for metric in metrics
            ],
        )

        return table

    @staticmethod
    def _metric_title(metric: str) -> str:
        if metric == "peak_allocated":
            return "Peak Allocated"
        if metric == "peak_reserved":
            return "Peak Reserved"
        return metric.replace("_", " ").title()

    @staticmethod
    def _format_worst_trend_delta(
        metric: StepMemoryCombinedMetric,
        *,
        single_rank: bool = False,
    ) -> str:
        """
        Format a stable head-vs-tail delta for the displayed series.

        Multi-rank mode uses the worst series; single-rank mode uses the only
        rank series, which is stored identically in both median and worst.
        """
        series = metric.series
        values_source = series.median if single_rank else series.worst
        if not series.steps or not values_source:
            return "—"

        values = [float(value) for value in values_source]
        n_values = len(values)
        if n_values < 2:
            return "—"

        segment = max(4, int(round(n_values * 0.20)))
        segment = min(segment, max(1, n_values // 2))

        head_avg = sum(values[:segment]) / segment
        tail_avg = sum(values[-segment:]) / segment
        delta = tail_avg - head_avg

        if delta == 0.0:
            return fmt_mem_new(0.0)

        sign = "+" if delta > 0.0 else "-"
        return f"{sign}{fmt_mem_new(abs(delta))}"


__all__ = [
    "StepMemoryRichFormatter",
]
