"""
High-level entry point for step-memory compute.

This mirrors process/system structure:
- compute_cli()
- compute_dashboard()
"""

from typing import Optional

from .cli_compute import StepMemoryCLIComputer
from .dashboard_compute import StepMemoryDashboardComputer
from .schema import StepMemoryCombinedResult


class StepMemoryMetricsComputer:
    """Unified step-memory compute facade for renderer use."""

    def __init__(
        self,
        db_path: str,
        *,
        stale_ttl_s: Optional[float] = 30.0,
        cli_window_size: int = 100,
        dashboard_window_size: int = 200,
    ) -> None:
        self._cli = StepMemoryCLIComputer(
            db_path=db_path,
            window_size=cli_window_size,
            stale_ttl_s=stale_ttl_s,
        )
        self._dashboard = StepMemoryDashboardComputer(
            db_path=db_path,
            window_size=dashboard_window_size,
            stale_ttl_s=stale_ttl_s,
        )

    def compute_cli(self) -> StepMemoryCombinedResult:
        """Return CLI payload."""
        return self._cli.compute()

    def compute_dashboard(self) -> StepMemoryCombinedResult:
        """Return dashboard payload."""
        return self._dashboard.compute()
