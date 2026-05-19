# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""
System renderer.

Presentation logic for system-level telemetry:
- CLI rendering (Rich)
- Dashboard payload (dict)
- Summary logging (optional)

All metric computation is delegated to SystemMetricsComputer.
"""

import shutil
from typing import Any, Dict

from rich.panel import Panel
from rich.table import Table

from traceml.aggregator.display_drivers.layout import SYSTEM_LAYOUT
from traceml.loggers.error_log import get_error_logger
from traceml.renderers.base_renderer import BaseRenderer
from traceml.utils.formatting import fmt_mem_new, fmt_percent

from .computer import SystemMetricsComputer


class SystemRenderer(BaseRenderer):
    """
    Renderer for system-level telemetry.

    Driver expectations:
    - CLI driver calls get_panel_renderable()
    - Dashboard driver calls get_dashboard_renderable()
    """

    NAME = "System"

    def __init__(self, db_path) -> None:
        super().__init__(name=self.NAME, layout_section_name=SYSTEM_LAYOUT)
        self.db_path = db_path
        self._logger = get_error_logger(self.NAME + "Renderer")
        self._computer = SystemMetricsComputer(db_path=self.db_path)

    def _compute_cli(self) -> Dict[str, Any]:
        return self._computer.compute_cli()

    def _compute_dashboard(self, window_n: int = 100) -> Dict[str, Any]:
        return self._computer.compute_dashboard(window_n=window_n)

    def get_panel_renderable(self) -> Panel:
        """Return a Rich Panel for CLI display (latest sample)."""
        data = self._compute_cli()

        grid = Table.grid(padding=(0, 1))
        grid.add_column(justify="left", style="bright_white", no_wrap=True)
        grid.add_column(justify="left", style="bright_white", no_wrap=True)

        if data.get("view") == "cluster":
            return self._cluster_panel(data, grid)
        return self._single_node_panel(data, grid)

    def _single_node_panel(self, data: Dict[str, Any], grid: Table) -> Panel:
        """Render the single-node terminal system view."""
        ram_pct_str = ""
        if data["ram_total"]:
            ram_pct = data["ram_used"] * 100.0 / data["ram_total"]
            ram_pct_str = fmt_percent(ram_pct)
        else:
            ram_pct_str = "N/A"

        grid.add_row(
            f"[bold green]CPU[/bold green] {fmt_percent(data['cpu'])}",
            f"[bold green]RAM[/bold green] {ram_pct_str}",
        )

        if not data["gpu_available"]:
            grid.add_row(
                "[bold green]GPU[/bold green]", "[red]Not available[/red]"
            )
        else:
            util_total = data.get("gpu_util_total")
            avg = (
                util_total / max(data["gpu_count"], 1)
                if util_total is not None
                else None
            )
            util_str = fmt_percent(avg) if avg is not None else "N/A"

            grid.add_row(
                f"[bold green]GPU UTIL[/bold green] {util_str}",
                f"[bold green]GPU MEM[/bold green] "
                f"{self._format_percent_ratio(data['gpu_mem_used'], data['gpu_mem_total'])}",
            )

            temp = data.get("gpu_temp_max")
            temp_str = (
                f"[bold green]GPU TMP[/bold green] {temp:.1f}°C"
                if temp is not None
                else "[bold green]GPU TMP[/bold green] N/A"
            )

            headroom = data.get("gpu_mem_headroom_min")
            headroom_str = (
                f"[bold green]GPU HDRM[/bold green] {fmt_mem_new(headroom)}"
                if headroom is not None
                else "[bold green]GPU HDRM[/bold green] N/A"
            )
            grid.add_row(temp_str, headroom_str)

        return self._panel(grid, "[bold cyan]System Metrics[/bold cyan]")

    def _cluster_panel(self, data: Dict[str, Any], grid: Table) -> Panel:
        """Render the multi-node terminal system view."""
        metrics = data.get("metrics", {})
        grid.add_row(
            self._cluster_metric("CPU", metrics.get("cpu"), "%"),
            self._cluster_metric("RAM", metrics.get("ram"), "%"),
        )

        if not data.get("gpu_available"):
            grid.add_row(
                "[bold green]GPU[/bold green]", "[red]Not available[/red]"
            )
        else:
            grid.add_row(
                self._cluster_metric("GPU UTIL", metrics.get("gpu_util"), "%"),
                self._cluster_metric("GPU MEM", metrics.get("gpu_mem"), "%"),
            )
            grid.add_row(
                self._cluster_metric("GPU TMP", metrics.get("gpu_temp"), "C"),
                self._cluster_metric(
                    "GPU HDRM",
                    metrics.get("gpu_headroom"),
                    "bytes",
                ),
            )

        suffix = str(data.get("title_suffix") or "").strip()
        title = "[bold cyan]System Metrics[/bold cyan]"
        if suffix:
            title = f"[bold cyan]System Metrics {suffix}[/bold cyan]"
        return self._panel(grid, title)

    def _cluster_metric(
        self,
        label: str,
        metric: Any,
        unit: str,
    ) -> str:
        """Format one median/worst-node metric pair."""
        if not isinstance(metric, dict):
            return f"[bold green]{label}[/bold green] N/A"

        left = self._format_metric_value(metric.get("median"), unit)
        right = self._format_metric_value(metric.get("worst"), unit)
        node = str(metric.get("worst_node") or "n/a")
        return f"[bold green]{label}[/bold green] {left} / {right} {node}"

    def _format_metric_value(self, value: Any, unit: str) -> str:
        """Format one terminal metric value."""
        if value is None:
            return "N/A"
        try:
            numeric = float(value)
        except Exception:
            return "N/A"
        if unit == "%":
            return fmt_percent(numeric)
        if unit == "C":
            return f"{numeric:.1f}°C"
        if unit == "bytes":
            return fmt_mem_new(numeric)
        return f"{numeric:.1f}"

    def _format_percent_ratio(self, numerator: Any, denominator: Any) -> str:
        """Format a ratio as a percentage for compact terminal display."""
        try:
            den = float(denominator)
            if den <= 0.0:
                return "N/A"
            return fmt_percent(float(numerator) * 100.0 / den)
        except Exception:
            return "N/A"

    def _panel(self, grid: Table, title: str) -> Panel:
        """Build the common Rich panel wrapper."""

        cols, _ = shutil.get_terminal_size()
        panel_width = min(max(100, int(cols * 0.75)), 100)

        return Panel(
            grid,
            title=title,
            title_align="center",
            border_style="cyan",
            width=panel_width,
        )

    def get_dashboard_renderable(self) -> Dict[str, Any]:
        """
        Return a compact dashboard payload.

        This matches the semantics of your existing system dashboard card, but
        avoids shipping the raw system table to the UI.
        """
        return self._compute_dashboard(window_n=100)
