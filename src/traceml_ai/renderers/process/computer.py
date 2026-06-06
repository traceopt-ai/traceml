"""
High-level process telemetry compute entry point.

This class cleanly composes:
- terminal/live snapshot compute
- dashboard/UI compute

The split is intentional:
- `compute_cli()` is for terminal renderer output only
- `compute_dashboard()` is for UI/dashboard payload only
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .cli_compute import ProcessCLIComputer
from .dashboard_compute import ProcessDashboardComputer


class ProcessMetricsComputer:
    """
    Unified entry point for process telemetry compute.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.
    stale_ttl_s:
        Maximum age in seconds for stale fallback reuse.
    dashboard_max_rows:
        Maximum retained history rows for dashboard UI.
    """

    def __init__(
        self,
        db_path: str,
        stale_ttl_s: Optional[float] = 30.0,
        dashboard_max_rows: int = 200,
    ) -> None:
        self._cli = ProcessCLIComputer(
            db_path=db_path,
            stale_ttl_s=stale_ttl_s,
        )
        self._dashboard = ProcessDashboardComputer(
            db_path=db_path,
            dashboard_max_rows=dashboard_max_rows,
            stale_ttl_s=stale_ttl_s,
        )

    def compute_cli(self) -> Dict[str, Any]:
        """
        Return the latest terminal/live process snapshot.
        """
        return self._cli.compute()

    def compute_dashboard(self) -> Dict[str, Any]:
        """
        Return the dashboard/UI payload for process metrics.
        """
        return self._dashboard.compute()
